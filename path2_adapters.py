import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import warnings
warnings.filterwarnings('ignore')

ffmpeg_path = os.path.expanduser("~/Library/Application Support/ffmpeg-downloader/ffmpeg")
if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")


class EncodingAdapter(nn.Module):
    def __init__(self, clip_dim=512, llm_dim=768):
        super().__init__()
        self.projection = nn.Linear(clip_dim, llm_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(llm_dim)
    
    def forward(self, clip_features):
        x = self.projection(clip_features)
        x = self.activation(x)
        x = self.norm(x)
        return x


class DecodingAdapter(nn.Module):
    def __init__(self, llm_dim=768, output_dim=512):
        super().__init__()
        self.projection = nn.Linear(llm_dim, output_dim)
        self.activation = nn.GELU()
    
    def forward(self, llm_features):
        x = self.projection(llm_features)
        x = self.activation(x)
        return x


class Path2Pipeline:
    
    def __init__(self, clip_model="openai/clip-vit-base-patch32", 
                 llm_model="gpt2", 
                 sd_model="runwayml/stable-diffusion-v1-5"):
        print("=" * 60)
        print("Path 2: LLM + Adapters Pipeline")
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
        
        print(f"\nLoading CLIP Image Encoder ({clip_model})...")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print("Initializing Encoding Adapter...")
        self.encoding_adapter = EncodingAdapter(clip_dim=512, llm_dim=768)
        self.encoding_adapter = self.encoding_adapter.to(self.device)
        self.encoding_adapter.eval()
        
        print(f"Loading LLM ({llm_model})...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.llm = self.llm.to(self.device)
        self.llm.eval()
        
        print("Initializing Decoding Adapter...")
        self.decoding_adapter = DecodingAdapter(llm_dim=768, output_dim=512)
        self.decoding_adapter = self.decoding_adapter.to(self.device)
        self.decoding_adapter.eval()
        
        print(f"Loading Stable Diffusion Image Decoder...")
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
    
    def encode_image(self, image_path):
        print(f"\n[1/5] Encoding image with CLIP...")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        print(f"Image encoded to {image_features.shape} tensor")
        return image_features, image
    
    def get_image_description(self, image):
        print(f"\n[2/5] Generating image description with CLIP...")
        
        labels = [
            "a photo of a dog",
            "a photo of a cat",
            "a photo of a person",
            "a photo of a landscape",
            "a photo of a building",
            "a photo of a vehicle",
            "a photo of animals in nature",
            "a photo of outdoor scenery",
            "a photo of an indoor scene",
            "a photo of food",
            "a photo of a city street",
            "a photo of mountains and sky"
        ]
        
        inputs = self.clip_processor(
            text=labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        top_probs, top_indices = probs[0].topk(3)
        description_parts = []
        for idx, prob in zip(top_indices, top_probs):
            if prob > 0.1:
                label = labels[idx].replace("a photo of ", "")
                description_parts.append(label)
        
        description = ", ".join(description_parts)
        print(f"Description: {description}")
        return description
    
    def apply_encoding_adapter(self, image_features):
        print(f"\n[3/5] Applying encoding adapter...")
        
        with torch.no_grad():
            adapted_features = self.encoding_adapter(image_features)
        
        print(f"Features projected to LLM space: {adapted_features.shape}")
        return adapted_features
    
    def process_with_llm(self, description, instruction):
        print(f"\n[4/5] Processing with LLM...")
        
        prompt = f"Image description: {description}. {instruction} New description:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "New description:" in generated_text:
            modified = generated_text.split("New description:")[-1].strip()
        else:
            modified = generated_text[len(prompt):].strip()
        
        if len(modified) < 10:
            modified = f"{description}, highly detailed, cinematic, professional quality"
        
        print(f"LLM output: {modified[:80]}...")
        return modified, outputs
    
    def apply_decoding_adapter(self, llm_output):
        print(f"\n[5/5] Applying decoding adapter...")
        
        last_hidden = llm_output
        
        with torch.no_grad():
            if len(last_hidden.shape) == 2:
                last_hidden = last_hidden.unsqueeze(0)
            
            adapted = self.decoding_adapter(last_hidden.float())
        
        print(f"Features adapted for image generation")
        return adapted
    
    def generate_image(self, prompt, output_path="output.png", steps=30, seed=None):
        print(f"\nGenerating image with Stable Diffusion...")
        
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
    
    def run(self, image_path, instruction="Make this more dramatic and cinematic", output_dir="outputs/path2"):
        print("\n" + "=" * 60)
        print("Running Path 2 Pipeline...")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        image_features, image = self.encode_image(image_path)
        
        description = self.get_image_description(image)
        
        adapted_encoding = self.apply_encoding_adapter(image_features)
        
        modified_prompt, llm_outputs = self.process_with_llm(description, instruction)
        
        self.apply_decoding_adapter(adapted_encoding)
        
        output_img = os.path.join(output_dir, f"path2_image_to_image_{ts}.png")
        self.generate_image(modified_prompt, output_path=output_img)
        
        with open(os.path.join(output_dir, f"path2_clip_description_{ts}.txt"), "w") as f:
            f.write(description)
        with open(os.path.join(output_dir, f"path2_llm_modified_prompt_{ts}.txt"), "w") as f:
            f.write(modified_prompt)
        
        print("\n" + "=" * 60)
        print("Done!")
        print(f"Input: {image_path}")
        print(f"Output: {output_img}")
        print("=" * 60)
        
        return output_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        image_file = "path2_dog.jpg"
        if not os.path.exists(image_file):
            print(f"Error: Default image not found: {image_file}")
            print("Usage: python path2_adapters.py <image_file> [instruction]")
            sys.exit(1)
        print(f"Using default image: A simple photo of a dog sitting in a park")
    else:
        image_file = sys.argv[1]
    
    instruction = sys.argv[2] if len(sys.argv) > 2 else "Make this dog look like a cyberpunk robot standing on Mars"
    
    if not os.path.exists(image_file):
        print(f"Error: Image file not found: {image_file}")
        print("Usage: python path2_adapters.py <image_file> [instruction]")
        sys.exit(1)
    
    try:
        pipeline = Path2Pipeline()
        pipeline.run(image_file, instruction=instruction)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

