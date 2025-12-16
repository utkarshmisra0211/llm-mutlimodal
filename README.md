# Multimodal AI: Three Paths to Guest LLMs

This project explores three different architectural approaches to building multimodal AI systems that can process and generate content across text, images, and audio. Each path represents a different trade-off between simplicity, customization, and performance.

## What This Project Does

This project demonstrates three distinct approaches to multimodal AI, showing how AI can transform your voice into an image, or turn a photo of your dog into a cyberpunk robot on Mars.

## Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd guest-llm

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg
yes | ffdl install

# Run the paths
python3 path1_tools.py
python3 path2_adapters.py
python3 path3_unified.py
```

## The Three Paths

### Path 1: LLM + Tools

Chain together specialized pretrained models for audio-to-image generation.

**Command:**
```bash
python3 path1_tools.py
```

**What it does:**
1. Takes audio input (default: path1_audio.mp3)
2. Transcribes using Whisper speech-to-text
3. Enhances the prompt using MagicPrompt
4. Generates an image using Stable Diffusion

**Example:**
```
Input Audio: "a medieval castle at sunset with dramatic clouds"
Output: Beautiful cinematic castle at sunset image
```

**Outputs saved to:** `outputs/path1/`
- `path1_audio_to_image_*.png` - Generated image
- `path1_transcription_*.txt` - Whisper transcription
- `path1_enhanced_prompt_*.txt` - Enhanced prompt

---

### Path 2: LLM + Adapters

Use fine-tuned adapter layers to transform images based on instructions.

**Command:**
```bash
python3 path2_adapters.py
```

**What it does:**
1. Takes an image input (default: path2_dog.jpg)
2. Encodes it with CLIP
3. Encoding adapter projects to LLM space
4. GPT-2 processes and transforms the concept
5. Decoding adapter prepares for image generation
6. Stable Diffusion generates the transformed image

**Example:**
```
Input: Photo of a dog in a park
Instruction: "Make this dog look like a cyberpunk robot standing on Mars"
Output: Cyberpunk robot dog on Mars landscape
```

**Outputs saved to:** `outputs/path2/`
- `path2_image_to_image_*.png` - Transformed image
- `path2_clip_description_*.txt` - CLIP's understanding
- `path2_llm_modified_prompt_*.txt` - LLM output

---

### Path 3: Unified Models

Single multimodal model that handles any combination of inputs and outputs.

**Command:**
```bash
python3 path3_unified.py
```

**What it does:**

Demonstrates four different capabilities:

1. **Text to Image**: Generate images from text descriptions
2. **Image + Text to Text**: Visual Question Answering
3. **Audio to Text**: Speech transcription
4. **Multimodal**: Process image + audio + text simultaneously

**Example workflows:**
```
Text to Image: "a cyberpunk robot dog on Mars" → Robot dog image

Visual QA: Image + "What animal is this?" → "A brown and white dog in a field"

Audio transcription: Audio file → "a medieval castle at sunset..."

Multimodal: Image + Audio + Text → Combined understanding
```

**Outputs saved to:** `outputs/path3/`
- `path3_text_to_image_*.png` - Generated images
- `path3_vqa_answer_*.txt` - VQA responses
- `path3_transcription_*.txt` - Audio transcriptions
- `path3_multimodal_*.txt` - Combined multimodal analysis

---

## Technical Architecture

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Speech-to-Text | Whisper (base) | Audio transcription |
| Prompt Enhancement | MagicPrompt (GPT-2) | Improve text prompts |
| Image Understanding | CLIP / BLIP | Encode and caption images |
| Language Model | GPT-2 | Text processing |
| Image Generation | Stable Diffusion v1.5 | Generate images |

### System Requirements

- Python 3.9+
- GPU: Recommended (Apple Silicon MPS, CUDA, or CPU)
- RAM: 8GB minimum, 16GB recommended
- Storage: ~10GB for models

### Dependencies

Core libraries (see requirements.txt):
- torch - Deep learning framework
- transformers - Hugging Face models
- diffusers - Stable Diffusion
- openai-whisper - Audio transcription
- ffmpeg-downloader - Audio processing

## Project Structure

```
guest-llm/
├── path1_tools.py          # Path 1: LLM + Tools
├── path2_adapters.py       # Path 2: LLM + Adapters
├── path3_unified.py        # Path 3: Unified MLLM
├── path1_audio.mp3         # Sample audio file
├── path2_dog.jpg           # Sample image file
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── outputs/               # Generated outputs
    ├── path1/            # Path 1 outputs
    ├── path2/            # Path 2 outputs
    └── path3/            # Path 3 outputs
```

## Example Use Cases

### Path 1: Podcast to Art
Convert podcast descriptions into visual artwork:
```bash
python3 path1_tools.py my_description.mp3
```

### Path 2: Image Transformation with Instructions
Transform photos based on natural language:
```bash
python3 path2_adapters.py family_photo.jpg "Make this look like a Renaissance painting"
```

### Path 3: Multimodal Content Creation
Process multiple inputs simultaneously by running:
```bash
python3 path3_unified.py
```

The script demonstrates all capabilities including text-to-image generation, visual question answering, audio transcription, and multimodal understanding.

## Customization

### Change Models

Edit the initialization in each file:

```python
# Path 1
pipeline = Path1Pipeline(
    whisper_model="base",  # Options: tiny, base, small, medium, large
    sd_model="runwayml/stable-diffusion-v1-5"
)

# Path 2
pipeline = Path2Pipeline(
    clip_model="openai/clip-vit-base-patch32",
    llm_model="gpt2",
    sd_model="runwayml/stable-diffusion-v1-5"
)
```

### Adjust Generation Parameters

```python
# More inference steps = higher quality (but slower)
pipeline.generate_image(prompt, steps=50)

# Fixed seed for reproducibility
pipeline.generate_image(prompt, seed=42)
```

## Understanding the Three Approaches

**Path 1** represents traditional ML engineering: connecting specialized tools. Simple to implement, no training needed, but limited customization.

**Path 2** demonstrates the modern approach: fine-tuning adapters like LLaVA and Flamingo. Better customization with efficient training of small adapter layers.

**Path 3** shows the future: unified models like GPT-4o and Gemini. Seamless integration of all modalities in a single model with faster inference.
