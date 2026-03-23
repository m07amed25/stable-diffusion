# 🌌 Stable Diffusion PyTorch Implementation

A modular, clean, and easily understandable implementation of **Stable Diffusion** from scratch in PyTorch. This repository breaks down the magic of Stable Diffusion—Text-to-Image and Image-to-Image generation—into readable and hackable components.

## ✨ Features

- **Text-to-Image Generation**: Generate vivid images from natural language text prompts.
- **Image-to-Image Generation**: Provide a source image and a text prompt to perform guided image modifications.
- **Classifier-Free Guidance (CFG)**: Controls how closely the generated image aligns with the text prompt.
- **DDPM Sampler**: Denoising Diffusion Probabilistic Models (DDPM) scheduling implementation.
- **Modular Components**: Separate modules for CLIP text encoding, VAE (Encoder/Decoder), and the UNet Diffusion engine.

---

## 🛠️ Project Structure

The codebase is organized in `sd/` and follows a clean architectural separation:

- `pipeline.py`: The main inference logic handling the diffusion loop, conditioning, and image generation.
- `model_loader.py`: Handles loading and dispatching pre-trained model weights.
- `ddpm.py`: Custom DDPM sampler handling noise addition and timestep inference scheduling.
- `clip.py`: Implementation of the OpenAI CLIP Text Encoder.
- `encoder.py` & `decoder.py`: The Variational Autoencoder (VAE) modules.
- `diffusion.py`: The core Diffusion model (UNet).
- `attention.py`: Implementation of self-attention and cross-attention blocks.
- `demo.ipynb`: A complete Jupyter Notebook demonstrating the usage.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install torch torchvision tqdm numpy
```
*Note: A GPU with CUDA support is highly recommended for generation.*

### Downloading Model Weights
This implementation relies on the official Stable Diffusion v1.5 weights.
Place the `v1-5-pruned-emaonly.ckpt` checkpoint file inside the `data/` directory.

Additionally, add the required vocab files (`merges.txt`, `vocab.json`) inside the `data/` directory for the CLIP tokenizer.

### Usage Example

You can trace the standard workflow via the `demo.ipynb` notebook or run inference directly in Python:

```python
import torch
from sd import model_loader
from sd.pipeline import generate
from transformers import CLIPTokenizer
from PIL import Image

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Models
# Loads CLIP, Encoder, Decoder, and Diffusion from the standard checkpoint
models = model_loader.preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", device)

# 3. Setup Tokenizer
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")

# 4. Generate Image (Text-to-Image)
prompt = "A high-tech cyberpunk city at night, neon lights, highly detailed, 8k"

generated_image_np = generate(
    prompt=prompt,
    uncond_prompt="",               # Negative prompt can go here
    input_image=None,               # Set to a PIL Image for Image-to-Image
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,                  # Higher values force closer matching to the prompt
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    seed=42,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# 5. Save the Output
output_image = Image.fromarray(generated_image_np)
output_image.save("output/generation.png")
output_image.show()
```

---

## 🧠 How It Works

1. **Text Prompt Processing**: The prompt is processed through the CLIP Text Encoder to produce a `context` numerical embedding.
2. **Latent Space & VAE**: If an input image is provided, it is compressed into a smaller latent space via the `Encoder`. If not, random noise latents are generated.
3. **Diffusion UNet**: The model loops over `n_inference_steps`. Each step takes the current noisy latents, the timestep embedding, and the text prompt context to predict the noise injected.
4. **CFG & Denoising**: Using Classifier-Free Guidance, conditional and unconditional predictions are evaluated, and the `DDPMSampler` removes the predicted noise.
5. **Image Decoding**: After the final timestep, the `Decoder` inflates the denoised latents back into the beautiful 512x512 RGB pixel space.

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome! If you find any issues, feel free to open a ticket or a pull request.

## 📝 License

This project relies on the Stable Diffusion architecture. Be sure to follow the open-source licensing respective to the model weights.
