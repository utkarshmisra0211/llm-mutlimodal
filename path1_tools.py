import os
import sys
import torch
import whisper
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Automatically add ffmpeg to PATH if it exists
ffmpeg_path = os.path.expanduser("~/Library/Application Support/ffmpeg-downloader/ffmpeg")
if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")


class Path1Pipeline:
    
    def __init__(self, whisper_model="base", sd_model="runwayml/stable-diffusion-v1-5"):
        print("=" * 60)
        print("Loading Pipeline...")
        print("=" * 60)
        
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        print(f"\nLoading Whisper ({whisper_model})...")
        # Whisper has issues with MPS sparse tensors, use CPU for Whisper
        whisper_device = "cpu" if self.device == "mps" else self.device
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)
        if self.device == "mps":
            print("Note: Whisper running on CPU (MPS sparse tensor compatibility)")
        
        print("Loading prompt enhancer (GPT-2 based MagicPrompt)...")
        self.prompt_enhancer = pipeline(
            "text-generation",
            model="Gustavosta/MagicPrompt-Stable-Diffusion",
            tokenizer="gpt2",
            device=-1  # CPU
        )
        
        print(f"Loading Stable Diffusion...")
        dtype = torch.float32 if self.device == "mps" else torch.float16
        
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        self.sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipe.scheduler.config
        )
        self.sd_pipe = self.sd_pipe.to(self.device)
        self.sd_pipe.enable_attention_slicing()
        
        print("\nReady!")
        print("=" * 60)
    
    def transcribe_audio(self, audio_path):
        print(f"\n[1/3] Transcribing audio...")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Only use fp16 on CUDA
        fp16 = (self.device == "cuda")
        result = self.whisper_model.transcribe(audio_path, fp16=fp16)
        
        text = result["text"].strip()
        lang = result.get("language", "en")
        
        print(f"Transcription: {text}")
        return {"text": text, "language": lang}
    
    def enhance_prompt(self, text):
        print(f"\n[2/3] Enhancing prompt...")
        
        # MagicPrompt works best with shorter seed text
        seed = text.strip()
        if not seed.endswith('.'):
            seed = seed.rstrip('.')
        
        try:
            result = self.prompt_enhancer(
                seed,
                max_length=100,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            enhanced = result[0]['generated_text'].strip()
            
            # Clean up any weird artifacts
            enhanced = enhanced.replace('\n', ' ').replace('  ', ' ')
            
        except Exception as e:
            print(f"Warning: Enhancement failed ({e}), using original text")
            enhanced = f"{text}, highly detailed, beautiful composition, professional quality, fantasy art"
        
        print(f"Enhanced: {enhanced[:80]}...")
        return enhanced
    
    def generate_image(self, prompt, output_path="output.png", steps=30, seed=None):
        print(f"\n[3/3] Generating image...")
        
        negative = "blurry, bad quality, distorted, ugly, watermark, text"
        
        gen = None
        if seed:
            gen = torch.Generator(device=self.device).manual_seed(seed)
        
        if self.device == "mps":
            result = self.sd_pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=steps,
                guidance_scale=7.5,
                width=512,
                height=512,
                generator=gen
            )
        else:
            with torch.autocast(self.device):
                result = self.sd_pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                    generator=gen
                )
        
        img = result.images[0]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path)
        
        print(f"Saved: {output_path}")
        return output_path
    
    def run(self, audio_path, output_dir="outputs/path1"):
        print("\n" + "=" * 60)
        print("Running pipeline...")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result = self.transcribe_audio(audio_path)
        text = result["text"]
        
        prompt = self.enhance_prompt(text)
        
        output_img = os.path.join(output_dir, f"path1_audio_to_image_{ts}.png")
        self.generate_image(prompt, output_path=output_img)
        
        with open(os.path.join(output_dir, f"path1_transcription_{ts}.txt"), "w") as f:
            f.write(text)
        with open(os.path.join(output_dir, f"path1_enhanced_prompt_{ts}.txt"), "w") as f:
            f.write(prompt)
        
        print("\n" + "=" * 60)
        print("Done!")
        print(f"Output: {output_img}")
        print("=" * 60)
        
        return output_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        audio_file = "path1_audio.mp3"
        print(f"Using default audio file: {audio_file}")
    else:
        audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        print("Usage: python path1_tools.py <audio_file>")
        sys.exit(1)
    
    try:
        pipeline = Path1Pipeline()
        pipeline.run(audio_file)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
