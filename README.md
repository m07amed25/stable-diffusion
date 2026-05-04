# Stable Diffusion — PyTorch Implementation from Scratch

A modular, clean, and thoroughly commented implementation of **Stable Diffusion v1.5** built from scratch in PyTorch. This repository deconstructs the full text-to-image and image-to-image generation pipeline into independent, readable, and hackable components — no black boxes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Weights and Tokenizer Files](#model-weights-and-tokenizer-files)
- [Usage](#usage)
  - [Text-to-Image](#text-to-image)
  - [Image-to-Image](#image-to-image)
  - [Jupyter Notebook Demo](#jupyter-notebook-demo)
- [Configuration Reference](#configuration-reference)
- [How It Works](#how-it-works)
  - [1. Text Encoding CLIP](#1-text-encoding-clip)
  - [2. Latent Space VAE Encoder](#2-latent-space-vae-encoder)
  - [3. Diffusion Loop UNet](#3-diffusion-loop-unet)
  - [4. Classifier-Free Guidance](#4-classifier-free-guidance)
  - [5. DDPM Sampler](#5-ddpm-sampler)
  - [6. Image Decoding VAE Decoder](#6-image-decoding-vae-decoder)
- [Module Reference](#module-reference)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Abbreviations and Glossary](#abbreviations-and-glossary)
  - [High-Level Concepts](#high-level-concepts)
  - [Dimensions and Tensor Notation](#dimensions-and-tensor-notation)
  - [Neural Network Layer Abbreviations](#neural-network-layer-abbreviations)
  - [DDPM / Scheduler Math](#ddpm--scheduler-math)
- [Detailed Layer Reference](#detailed-layer-reference)
  - [CLIP Text Encoder](#clip-text-encoder-clippy)
  - [VAE Encoder](#vae-encoder-encoderpy)
  - [VAE Decoder](#vae-decoder-decoderpy)
  - [UNet Diffusion Model](#unet-diffusion-model-diffusionpy)
  - [Attention Modules](#attention-modules-attentionpy)
  - [DDPM Sampler](#ddpm-sampler-ddpmpy)
- [References](#references)

---

## Overview

Stable Diffusion is a latent diffusion model that generates high-quality images conditioned on text prompts. Unlike pixel-space diffusion models, it operates in a compressed **latent space** (64x64) rather than directly on 512x512 pixels, dramatically reducing compute requirements. This implementation faithfully reproduces the original architecture including:

- The **CLIP** text encoder for prompt conditioning
- A **Variational Autoencoder (VAE)** for encoding images into latent space and decoding back to pixels
- A **UNet** with cross-attention for iterative noise prediction
- A **DDPM scheduler** (Denoising Diffusion Probabilistic Models) for noise management
- **Classifier-Free Guidance (CFG)** for prompt adherence control

---

## Features

| Feature                      | Description                                                      |
| ---------------------------- | ---------------------------------------------------------------- |
| **Text-to-Image**            | Generate 512x512 images from any text prompt                     |
| **Image-to-Image**           | Modify existing images guided by a text prompt                   |
| **Negative Prompts**         | Suppress unwanted concepts via unconditional guidance            |
| **CFG Scale Control**        | Tune how strictly the output follows the text prompt             |
| **Seed Reproducibility**     | Set a fixed seed for deterministic generation                    |
| **CPU Offloading**           | Offload idle models to CPU to reduce GPU memory usage            |
| **Modular Codebase**         | Each component (CLIP, VAE, UNet, Sampler) is a standalone module |
| **No External SD Libraries** | Does not depend on `diffusers` — built purely in PyTorch         |

---

## Architecture

```
Text Prompt
    |
    v
+------------------+
|  CLIP Text       |  --> context embedding  (batch, 77, 768)
|  Encoder         |
+------------------+
    |
    v (conditioning)
    |
    |   [Text-to-Image]         [Image-to-Image]
    |   Random Latent Noise     Image -> VAE Encoder -> Noisy Latents
    |          |                             |
    +----------+-----------------------------+
                          |
                          v
             +------------------------+
             |   Diffusion UNet       |  <- timestep embedding
             |   (with cross-attn)    |  <- context embedding
             +------------------------+
                          |
                N inference steps
                (noise prediction + DDPM step)
                          |
                          v
             +------------------------+
             |    VAE Decoder         |  --> 512x512 RGB image
             +------------------------+
```

---

## Project Structure

```
stable-diffusion/
|
+-- data/
|   +-- v1-5-pruned-emaonly.ckpt   # Stable Diffusion v1.5 weights (download separately)
|   +-- vocab.json                 # CLIP tokenizer vocabulary
|   +-- merges.txt                 # CLIP BPE merge rules
|
+-- images/                        # Place input images for img2img here
|
+-- output/                        # Generated images are saved here
|
+-- sd/
|   +-- pipeline.py                # Main inference pipeline (generate function)
|   +-- model_loader.py            # Loads and maps checkpoint weights to modules
|   +-- diffusion.py               # UNet diffusion model
|   +-- clip.py                    # CLIP text encoder
|   +-- encoder.py                 # VAE encoder (image -> latent)
|   +-- decoder.py                 # VAE decoder (latent -> image)
|   +-- attention.py               # Self-attention and cross-attention implementations
|   +-- ddpm.py                    # DDPM noise scheduler
|   +-- demo.ipynb                 # Interactive notebook walkthrough
|   +-- add_noise.ipynb            # Notebook exploring forward noise process
|
+-- README.md
```

---

## Prerequisites

- Python 3.9+
- PyTorch 2.0+ (CUDA build strongly recommended)
- A CUDA-capable GPU with at least **6 GB VRAM** (8-12 GB recommended)
  - CPU-only inference is supported but will be very slow (~10-30 min per image)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/m07amed25/stable-diffusion.git
cd stable-diffusion

# Install dependencies
pip install torch torchvision tqdm numpy transformers pillow
```

> **Note:** Install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) to match your CUDA version.

---

## Model Weights and Tokenizer Files

This implementation uses the official **Stable Diffusion v1.5** weights. You must download them separately due to licensing.

### 1. Model Checkpoint

Download `v1-5-pruned-emaonly.ckpt` from Hugging Face:

```
https://huggingface.co/runwayml/stable-diffusion-v1-5
```

Place the file at: `data/v1-5-pruned-emaonly.ckpt`

### 2. CLIP Tokenizer Files

Download from the OpenAI CLIP repository or Hugging Face:

- `vocab.json`
- `merges.txt`

These can be obtained from:

```
https://huggingface.co/openai/clip-vit-large-patch14
```

Place both files inside the `data/` directory.

**Final `data/` directory:**

```
data/
+-- v1-5-pruned-emaonly.ckpt
+-- vocab.json
+-- merges.txt
```

---

## Usage

### Text-to-Image

```python
import torch
from PIL import Image
from transformers import CLIPTokenizer
from sd import model_loader
from sd.pipeline import generate

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
idle_device = "cpu"  # Models not in use are offloaded here to save VRAM

# Load all model components from the checkpoint
models = model_loader.preload_models_from_standard_weights(
    "data/v1-5-pruned-emaonly.ckpt", device
)

# Load the CLIP tokenizer
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")

# Run text-to-image generation
output = generate(
    prompt="A futuristic cityscape at sunset, cinematic lighting, highly detailed, 8k",
    uncond_prompt="blurry, low quality, distorted, ugly",  # negative prompt
    input_image=None,           # None = text-to-image mode
    strength=0.8,               # unused for text-to-image
    do_cfg=True,                # enable classifier-free guidance
    cfg_scale=7.5,              # prompt adherence strength (typical range: 5-15)
    sampler_name="ddpm",
    n_inference_steps=50,       # more steps = higher quality, slower
    models=models,
    seed=42,                    # set for reproducibility
    device=device,
    idle_device=idle_device,
    tokenizer=tokenizer,
)

# Save the output
Image.fromarray(output).save("output/result.png")
```

### Image-to-Image

```python
from PIL import Image

# Load your source image
input_image = Image.open("images/my_photo.jpg").convert("RGB")

output = generate(
    prompt="A fantasy painting version, detailed brushwork, oil on canvas",
    uncond_prompt="",
    input_image=input_image,    # provide source image
    strength=0.6,               # 0.0 = copy source, 1.0 = ignore source completely
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models=models,
    seed=42,
    device=device,
    idle_device=idle_device,
    tokenizer=tokenizer,
)

Image.fromarray(output).save("output/img2img_result.png")
```

### Jupyter Notebook Demo

An interactive walkthrough is available at `sd/demo.ipynb`. Open it in Jupyter or VS Code and run cells top-to-bottom. It includes visualization of intermediate latents and side-by-side before/after comparisons for image-to-image.

A second notebook `sd/add_noise.ipynb` explores the forward diffusion (noise addition) process step by step.

---

## Configuration Reference

| Parameter           | Type        | Default  | Description                                                        |
| ------------------- | ----------- | -------- | ------------------------------------------------------------------ |
| `prompt`            | `str`       | required | Text describing the desired image                                  |
| `uncond_prompt`     | `str`       | `""`     | Negative prompt — concepts to suppress                             |
| `input_image`       | `PIL.Image` | `None`   | Source image for img2img; `None` for text-to-image                 |
| `strength`          | `float`     | `0.8`    | Img2img noise level (0.0-1.0); higher = more deviation from source |
| `do_cfg`            | `bool`      | `True`   | Enable Classifier-Free Guidance                                    |
| `cfg_scale`         | `float`     | `7.5`    | CFG scale — higher = more prompt-faithful, less diverse            |
| `sampler_name`      | `str`       | `"ddpm"` | Noise scheduler. Currently `"ddpm"` is supported                   |
| `n_inference_steps` | `int`       | `50`     | Denoising steps. 20-30 for fast; 50+ for quality                   |
| `seed`              | `int`       | `None`   | Random seed. `None` = different result each run                    |
| `device`            | `str`       | `"cuda"` | Compute device: `"cuda"` or `"cpu"`                                |
| `idle_device`       | `str`       | `"cpu"`  | Device to offload inactive models to                               |

---

## How It Works

### 1. Text Encoding CLIP

The text prompt is tokenized into a sequence of 77 tokens using a **Byte-Pair Encoding (BPE)** tokenizer. The `CLIP` module embeds these tokens and processes them through a stack of transformer layers (self-attention + MLP + LayerNorm), producing a context tensor of shape `(batch, 77, 768)`.

When CFG is active, both the conditional and unconditional (negative) prompts are encoded and concatenated into a single batched tensor `(2, 77, 768)` to allow a single forward pass through the UNet.

### 2. Latent Space VAE Encoder

For **image-to-image**: the input image is resized to 512x512, normalized to `[-1, 1]`, and compressed by the VAE encoder through a series of ResNet blocks and downsampling layers into a `(4, 64, 64)` latent. Controlled noise is added at a level determined by `strength`, placing the latent at the correct timestep for the denoising schedule.

For **text-to-image**: a random Gaussian noise tensor of shape `(1, 4, 64, 64)` is sampled directly.

### 3. Diffusion Loop UNet

Over `n_inference_steps` timesteps (descending from ~999 to 0), the **UNet** takes:

- The current noisy latents `(1, 4, 64, 64)`
- A timestep embedding (sinusoidal, similar to transformers)
- The CLIP context `(2, 77, 768)` (conditional + unconditional)

It processes them through encoder blocks → bottleneck → decoder blocks, with **residual connections**, **GroupNorm**, and **cross-attention** layers where latent features attend to text token embeddings. The output is the predicted noise `(1, 4, 64, 64)`.

### 4. Classifier-Free Guidance

With CFG, the UNet produces two noise predictions per step — one conditioned on the prompt, one on the negative/empty prompt. The final prediction is blended as:

```
noise = uncond_noise + cfg_scale * (cond_noise - uncond_noise)
```

Higher `cfg_scale` values push the result to be more faithful to the prompt at the cost of variety and naturalness.

### 5. DDPM Sampler

The `DDPMSampler` implements the reverse diffusion process from the DDPM paper. It uses a **quadratic beta schedule** (beta_start=0.00085, beta_end=0.012) over 1000 training steps. At each inference timestep it:

1. Uses the predicted noise and current latents to estimate the original clean latent `x_0`
2. Recomputes the posterior mean and variance using cumulative alpha products
3. Optionally adds a small noise injection for non-final steps

For fast inference, steps are subsampled evenly from the 1000-step training schedule.

### 6. Image Decoding VAE Decoder

After the final denoising step, the `(4, 64, 64)` latent is upsampled by the VAE decoder through a symmetric stack of ResNet blocks and bilinear upsampling layers back to a `(3, 512, 512)` RGB image. The result is de-normalized from `[-1, 1]` to `[0, 255]` and returned as a NumPy `uint8` array.

---

## Module Reference

| File              | Class / Function                            | Description                                           |
| ----------------- | ------------------------------------------- | ----------------------------------------------------- |
| `pipeline.py`     | `generate(...)`                             | Top-level inference entry point                       |
| `model_loader.py` | `preload_models_from_standard_weights(...)` | Parses `.ckpt` and maps weights to model classes      |
| `clip.py`         | `CLIP`                                      | Full CLIP text transformer encoder                    |
| `encoder.py`      | `VAE_Encoder`                               | Convolutional VAE encoder (image to latent)           |
| `decoder.py`      | `VAE_Decoder`                               | Convolutional VAE decoder (latent to image)           |
| `diffusion.py`    | `Diffusion`                                 | UNet noise prediction model                           |
| `attention.py`    | `SelfAttention`, `CrossAttention`           | Scaled dot-product attention modules                  |
| `ddpm.py`         | `DDPMSampler`                               | DDPM beta schedule, timestep management, reverse step |

---

## Performance Tips

- **Use `idle_device="cpu"`** — moves CLIP, VAE encoder, and VAE decoder to CPU after use, keeping only the UNet on GPU. Essential for 6-8 GB VRAM GPUs.
- **Reduce `n_inference_steps` to 20-25** — produces decent quality at roughly double the speed of 50 steps.
- **Use half precision** — loading weights as `torch.float16` can halve VRAM usage (requires modifying `model_loader.py`).
- **Lower `cfg_scale` to 5-8** — values above 15 can over-saturate images and increase artifacts.
- **Always set `seed`** during parameter tuning to isolate the effect of one variable at a time.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please ensure new code is well-commented, explaining any non-obvious math or architectural decisions.

---

## License

The code in this repository is provided for educational and research use. The **Stable Diffusion v1.5 model weights** are governed by the [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). You are responsible for complying with this license when using or distributing the weights.

---

## Abbreviations and Glossary

Every acronym and shorthand used in this codebase and README, in one place.

### High-Level Concepts

| Abbreviation | Full Name                                | Meaning in this project                                                                       |
| ------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| **SD**       | Stable Diffusion                         | The overall latent diffusion model for text-to-image generation                               |
| **LDM**      | Latent Diffusion Model                   | The class of models that diffuse in a compressed latent space rather than pixel space         |
| **CLIP**     | Contrastive Language–Image Pre-training  | OpenAI's model used here purely as a text encoder to produce prompt embeddings                |
| **VAE**      | Variational Autoencoder                  | Neural network that compresses images to latents (Encoder) and reconstructs them (Decoder)    |
| **UNet**     | U-shaped Network                         | Encoder-decoder architecture with skip connections used as the noise predictor                |
| **DDPM**     | Denoising Diffusion Probabilistic Models | The mathematical framework and noise scheduler for the forward/reverse diffusion process      |
| **CFG**      | Classifier-Free Guidance                 | Technique to steer generation toward the prompt without a separate classifier                 |
| **BPE**      | Byte-Pair Encoding                       | Subword tokenization algorithm used by the CLIP tokenizer                                     |
| **EMA**      | Exponential Moving Average               | Averaging strategy used during training for more stable weights (reflected in `emaonly.ckpt`) |
| **ckpt**     | Checkpoint                               | A saved file containing all model weights, structured as a Python dict                        |

---

### Dimensions and Tensor Notation

| Symbol             | Meaning                                                                      |
| ------------------ | ---------------------------------------------------------------------------- |
| `B` / `Batch_Size` | Number of samples processed in parallel (usually 1 at inference, 2 with CFG) |
| `C` / `channels`   | Number of feature channels in a convolutional feature map                    |
| `H` / `Height`     | Spatial height of a feature map or image                                     |
| `W` / `Width`      | Spatial width of a feature map or image                                      |
| `Seq_Len`          | Sequence length of token embeddings (fixed at 77 for CLIP)                   |
| `Dim` / `d_embed`  | Embedding dimension (768 for CLIP, 320/640/1280 in UNet)                     |
| `d_head`           | Per-head dimension in multi-head attention = `d_embed / n_heads`             |
| `n_heads` / `H`    | Number of attention heads                                                    |
| `d_cross`          | Dimension of the cross-attention key/value context (768 from CLIP)           |
| `T` / `timestep`   | An integer in `[0, 999]` representing the noise level                        |
| `t`                | Current timestep during the reverse diffusion loop                           |

---

### Neural Network Layer Abbreviations

| Abbreviation  | Full Name                     | Description                                                                                  |
| ------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| **Conv2d**    | 2D Convolution                | Applies a learnable filter across spatial dimensions                                         |
| **Linear**    | Fully-Connected Linear Layer  | `y = xW + b` — maps between arbitrary embedding dimensions                                   |
| **GroupNorm** | Group Normalization           | Normalizes over groups of channels; used with `num_groups=32` throughout                     |
| **LayerNorm** | Layer Normalization           | Normalizes over the last dimension (embedding dim); used in CLIP and UNet attention blocks   |
| **SiLU**      | Sigmoid Linear Unit (= Swish) | `x * sigmoid(x)` — smooth non-linear activation used in VAE and UNet                         |
| **GELU**      | Gaussian Error Linear Unit    | `x * Φ(x)` — smooth activation used in the GeGLU feed-forward gate                           |
| **GeGLU**     | Gated Linear Unit with GELU   | `Linear(x) * GELU(gate)` — activation function in UNet attention blocks' FFN                 |
| **QuickGELU** | Fast GELU approximation       | `x * sigmoid(1.702 * x)` — used in CLIP layers as a fast approximation of GELU               |
| **FFN**       | Feed-Forward Network          | Two-layer MLP inside transformer blocks (expand 4x then contract back)                       |
| **MLP**       | Multi-Layer Perceptron        | Generic term for stacked linear layers with activations                                      |
| **ResBlock**  | Residual Block                | A block with a skip connection: `output = F(x) + x` to prevent vanishing gradients           |
| **QKV**       | Query, Key, Value             | The three projections computed from input in attention: Q asks, K holds keys, V holds values |
| **Attn**      | Attention                     | Short for the scaled dot-product attention operation                                         |
| **SA**        | Self-Attention                | Attention where Q, K, V all come from the same input sequence                                |
| **CA**        | Cross-Attention               | Attention where Q comes from latents and K, V come from the CLIP text context                |

---

### DDPM / Scheduler Math

| Symbol                  | Meaning                                                                         |
| ----------------------- | ------------------------------------------------------------------------------- |
| `β_t` (`beta`)          | Variance of noise added at timestep `t`; ranges from `0.00085` to `0.012`       |
| `α_t` (`alpha`)         | `1 - β_t` — fraction of signal kept at step `t`                                 |
| `ᾱ_t` (`alpha_cumprod`) | `∏ α_i` for `i=1..t` — cumulative product; represents total signal preservation |
| `x_t`                   | Noisy latent at timestep `t`                                                    |
| `x_0`                   | Original clean latent (what we are trying to recover)                           |
| `ε` (`epsilon`)         | The noise component; what the UNet is trained to predict                        |
| `σ_t`                   | Standard deviation of noise; `σ_t = sqrt(1 - ᾱ_t)`                              |
| `μ_t`                   | Posterior mean of `x_{t-1}` given `x_t` and `x_0`                               |

---

## Detailed Layer Reference

Full breakdown of every layer and block used in each model, with shapes and roles.

---

### CLIP Text Encoder (`clip.py`)

The CLIP encoder transforms a sequence of token IDs into rich contextual embeddings used to condition the UNet.

**Overall shape flow:**

```
(B, 77)  -->  CLIPEmbedding  -->  (B, 77, 768)
         -->  12x CLIPLayer  -->  (B, 77, 768)
         -->  LayerNorm      -->  (B, 77, 768)
```

#### `CLIPEmbedding`

| Sub-layer            | Type                       | Parameters                | Purpose                                                                    |
| -------------------- | -------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `token_embedding`    | `nn.Embedding(49408, 768)` | vocab_size=49408, dim=768 | Looks up a learned 768-d vector for each of the 77 input token IDs         |
| `position_embedding` | `nn.Parameter(77, 768)`    | learnable                 | Adds a position-specific bias to each token so the model knows token order |

#### `CLIPLayer` (repeated x12)

Each layer is a standard transformer encoder block with a causal mask:

| Sub-layer     | Type                           | Purpose                                                                  |
| ------------- | ------------------------------ | ------------------------------------------------------------------------ |
| `layernorm_1` | `LayerNorm(768)`               | Pre-norm before self-attention; stabilizes training                      |
| `attention`   | `SelfAttention(12 heads, 768)` | 12-head causal self-attention; each token attends to all previous tokens |
| Residual add  | —                              | `x = attention(norm(x)) + x` — preserves gradient flow                   |
| `layernorm_2` | `LayerNorm(768)`               | Pre-norm before the FFN                                                  |
| `linear_1`    | `Linear(768 → 3072)`           | First FFN layer — expands dimension by 4x                                |
| QuickGELU     | `x * sigmoid(1.702x)`          | Fast non-linear activation gating the expansion                          |
| `linear_2`    | `Linear(3072 → 768)`           | Second FFN layer — projects back to 768                                  |
| Residual add  | —                              | `x = FFN(norm(x)) + x`                                                   |

#### Final `LayerNorm(768)`

Applied once after all 12 layers to normalize the final output embeddings.

---

### VAE Encoder (`encoder.py`)

Compresses a 512x512 RGB image into a 64x64x4 latent. The factor-of-8 spatial reduction comes from 3 strided convolutions.

**Shape flow:**

```
(B, 3, 512, 512)
  --> Conv2d(3->128, k=3, p=1)           (B, 128, 512, 512)
  --> ResBlock(128, 128) x2               (B, 128, 512, 512)
  --> Conv2d(128->128, k=3, s=2)         (B, 128, 256, 256)   [downsample x1]
  --> ResBlock(128, 256)                  (B, 256, 256, 256)
  --> ResBlock(256, 256)                  (B, 256, 256, 256)
  --> Conv2d(256->256, k=3, s=2)         (B, 256, 128, 128)   [downsample x2]
  --> ResBlock(256, 512)                  (B, 512, 128, 128)
  --> ResBlock(512, 512) x2               (B, 512, 128, 128)
  --> Conv2d(512->512, k=3, s=2)         (B, 512, 64, 64)     [downsample x3]
  --> ResBlock(512, 512) x3               (B, 512, 64, 64)
  --> AttentionBlock(512)                 (B, 512, 64, 64)
  --> ResBlock(512, 512)                  (B, 512, 64, 64)
  --> GroupNorm(32, 512)                  (B, 512, 64, 64)
  --> SiLU                                (B, 512, 64, 64)
  --> Conv2d(512->8, k=3, p=1)           (B, 8, 64, 64)
  --> Conv2d(8->8, k=1)                  (B, 8, 64, 64)
  -- reparameterization (split into mean + logvar) --
  --> sample z = mean + std * noise       (B, 4, 64, 64)
  --> scale by 0.18215                    (B, 4, 64, 64)
```

#### `VAE_ResidualBlock(in_ch, out_ch)`

| Sub-layer                           | Purpose                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------- |
| `GroupNorm(32, in_ch)`              | Normalize across 32 groups of channels — more stable than BatchNorm for small batches |
| `SiLU`                              | Smooth activation function `x * σ(x)`                                                 |
| `Conv2d(in_ch → out_ch, k=3, p=1)`  | Spatial feature extraction; preserves spatial size                                    |
| `GroupNorm(32, out_ch)`             | Second normalization                                                                  |
| `SiLU`                              | Second activation                                                                     |
| `Conv2d(out_ch → out_ch, k=3, p=1)` | Second convolution                                                                    |
| Skip connection                     | If `in_ch == out_ch`: Identity. Else: `Conv2d(in_ch → out_ch, k=1)` to match channels |

#### `VAE_AttentionBlock(channels)`

| Sub-layer                           | Purpose                                                                                   |
| ----------------------------------- | ----------------------------------------------------------------------------------------- |
| `GroupNorm(32, channels)`           | Pre-norm                                                                                  |
| Reshape `(B,C,H,W)` → `(B, H*W, C)` | Flatten spatial dims into a sequence for attention                                        |
| `SelfAttention(1 head, channels)`   | Global self-attention over all spatial positions — lets every pixel attend to every other |
| Reshape back                        | Restore `(B,C,H,W)`                                                                       |
| Residual add                        | `output + input`                                                                          |

#### Reparameterization Trick

The final `Conv2d(512→8)` outputs 8 channels, split into `mean` (4ch) and `log_variance` (4ch). The latent is sampled as:

```
z = mean + sqrt(exp(log_var)) * noise
```

Then scaled by the training constant `0.18215` to normalize the latent distribution.

---

### VAE Decoder (`decoder.py`)

Mirrors the encoder: upsamples 64x64x4 latents back to 512x512x3 RGB.

**Shape flow:**

```
(B, 4, 64, 64)
  --> /= 0.18215                          undo latent scaling
  --> Conv2d(4->4, k=1)                  (B, 4, 64, 64)    post_quant_conv
  --> Conv2d(4->512, k=3, p=1)           (B, 512, 64, 64)  conv_in
  --> ResBlock(512,512)                   (B, 512, 64, 64)  mid block 1
  --> AttentionBlock(512)                 (B, 512, 64, 64)  mid attn
  --> ResBlock(512,512)                   (B, 512, 64, 64)  mid block 2
  -- Level 3: 64x64 -> 128x128 --
  --> ResBlock(512,512) x3                (B, 512, 64, 64)
  --> Upsample(scale=2)                   (B, 512, 128, 128)
  --> Conv2d(512->512, k=3, p=1)          (B, 512, 128, 128)
  -- Level 2: 128x128 -> 256x256 --
  --> ResBlock(512,512) x3                (B, 512, 128, 128)
  --> Upsample(scale=2)                   (B, 512, 256, 256)
  --> Conv2d(512->512, k=3, p=1)          (B, 512, 256, 256)
  -- Level 1: 256x256 -> 512x512 --
  --> ResBlock(512->256), ResBlock(256,256) x2   (B, 256, 256, 256)
  --> Upsample(scale=2)                   (B, 256, 512, 512)
  --> Conv2d(256->256, k=3, p=1)          (B, 256, 512, 512)
  -- Level 0: final --
  --> ResBlock(256->128), ResBlock(128,128) x2   (B, 128, 512, 512)
  --> GroupNorm(32, 128)                  (B, 128, 512, 512)
  --> SiLU                                (B, 128, 512, 512)
  --> Conv2d(128->3, k=3, p=1)           (B, 3, 512, 512)   RGB output
```

The `VAE_ResidualBlock` and `VAE_AttentionBlock` are identical to those in the encoder (described above).

---

### UNet Diffusion Model (`diffusion.py`)

The core noise predictor. Takes noisy latents + timestep + text context, outputs predicted noise with the same shape as the input latents.

**Overall interface:**

```
Inputs:
  x       : (B, 4, 64, 64)    noisy latents
  context : (B, 77, 768)      CLIP text embeddings
  time    : (1, 320)           sinusoidal timestep embedding

Output:
  noise   : (B, 4, 64, 64)    predicted noise
```

#### `TimeEmbedding(dim=320)`

Converts the 320-d sinusoidal time vector into a 1280-d conditioning signal:

| Sub-layer             | Shape                   | Purpose                                     |
| --------------------- | ----------------------- | ------------------------------------------- |
| `Linear(320 → 1280)`  | `(1, 320) → (1, 1280)`  | Expand time embedding                       |
| `SiLU`                | —                       | Non-linear activation                       |
| `Linear(1280 → 1280)` | `(1, 1280) → (1, 1280)` | Second projection for richer representation |

The 320-d sinusoidal input is computed in `get_time_embedding()` using 160 cosine + 160 sine frequencies (same formulation as positional encoding in transformers).

#### `UNet` encoder path (12 blocks, downsampling 3 times)

| Block index | Layers                                  | Input channels | Output channels | Spatial size       |
| ----------- | --------------------------------------- | -------------- | --------------- | ------------------ |
| 0           | Conv2d(4→320, k=3, p=1)                 | 4              | 320             | 64x64              |
| 1           | ResBlock(320,320) + AttnBlock(8h, 40)   | 320            | 320             | 64x64              |
| 2           | ResBlock(320,320) + AttnBlock(8h, 40)   | 320            | 320             | 64x64              |
| 3           | Conv2d(320→320, k=3, s=2, p=1)          | 320            | 320             | 32x32 — downsample |
| 4           | ResBlock(320,640) + AttnBlock(8h, 80)   | 320            | 640             | 32x32              |
| 5           | ResBlock(640,640) + AttnBlock(8h, 80)   | 640            | 640             | 32x32              |
| 6           | Conv2d(640→640, k=3, s=2, p=1)          | 640            | 640             | 16x16 — downsample |
| 7           | ResBlock(640,1280) + AttnBlock(8h,160)  | 640            | 1280            | 16x16              |
| 8           | ResBlock(1280,1280) + AttnBlock(8h,160) | 1280           | 1280            | 16x16              |
| 9           | Conv2d(1280→1280, k=3, s=2, p=1)        | 1280           | 1280            | 8x8 — downsample   |
| 10          | ResBlock(1280,1280)                     | 1280           | 1280            | 8x8                |
| 11          | ResBlock(1280,1280)                     | 1280           | 1280            | 8x8                |

All 12 encoder outputs are saved as **skip connections** for the decoder path.

#### `UNet` bottleneck

```
ResBlock(1280,1280) + AttnBlock(8h,160) + ResBlock(1280,1280)
Shape: (B, 1280, 8, 8) throughout
```

#### `UNet` decoder path (12 blocks, upsampling 3 times)

Each decoder block receives the previous output **concatenated** with the matching encoder skip connection (hence some input channels are doubled):

| Block index | Layers                                             | Input channels | Output channels | Spatial size  |
| ----------- | -------------------------------------------------- | -------------- | --------------- | ------------- |
| 0           | ResBlock(2560,1280)                                | 1280+1280      | 1280            | 8x8           |
| 1           | ResBlock(2560,1280)                                | 1280+1280      | 1280            | 8x8           |
| 2           | ResBlock(2560,1280) + Upsample                     | 1280+1280      | 1280            | 8x8 → 16x16   |
| 3           | ResBlock(2560,1280) + AttnBlock(8h,160)            | 1280+1280      | 1280            | 16x16         |
| 4           | ResBlock(2560,1280) + AttnBlock(8h,160)            | 1280+1280      | 1280            | 16x16         |
| 5           | ResBlock(1920,1280) + AttnBlock(8h,160) + Upsample | 1280+640       | 1280            | 16x16 → 32x32 |
| 6           | ResBlock(1920,640) + AttnBlock(8h, 80)             | 1280+640       | 640             | 32x32         |
| 7           | ResBlock(1280,640) + AttnBlock(8h, 80)             | 640+640        | 640             | 32x32         |
| 8           | ResBlock(960,640) + AttnBlock(8h, 80) + Upsample   | 640+320        | 640             | 32x32 → 64x64 |
| 9           | ResBlock(960,320) + AttnBlock(8h, 40)              | 640+320        | 320             | 64x64         |
| 10          | ResBlock(640,320) + AttnBlock(8h, 40)              | 320+320        | 320             | 64x64         |
| 11          | ResBlock(640,320) + AttnBlock(8h, 40)              | 320+320        | 320             | 64x64         |

#### `UNet_OutputLayer`

Final projection back to 4 latent channels:

| Sub-layer                   | Purpose                                           |
| --------------------------- | ------------------------------------------------- |
| `GroupNorm(32, 320)`        | Normalize features                                |
| `SiLU`                      | Activation                                        |
| `Conv2d(320 → 4, k=3, p=1)` | Project to predicted noise shape `(B, 4, 64, 64)` |

---

#### `UNET_ResidualBlock(in_ch, out_ch)`

Identical to the VAE residual block but also injects the timestep embedding:

| Sub-layer                           | Purpose                                         |
| ----------------------------------- | ----------------------------------------------- |
| `GroupNorm(32, in_ch)`              | Normalize input features                        |
| `SiLU`                              | Activation                                      |
| `Conv2d(in_ch → out_ch, k=3, p=1)`  | Feature extraction                              |
| `Linear(1280 → out_ch)`             | Project time embedding to feature channel count |
| `SiLU`                              | Activation on time signal                       |
| `feature + time[..., None, None]`   | Broadcast-add time embedding over spatial dims  |
| `GroupNorm(32, out_ch)`             | Normalize merged features                       |
| `SiLU`                              | Activation                                      |
| `Conv2d(out_ch → out_ch, k=3, p=1)` | Second conv                                     |
| Skip connection                     | Identity or `Conv2d(k=1)` to match channels     |

#### `UNET_AttentionBlock(n_heads, n_embd, d_context=768)`

A full transformer block that lets latent features attend to text context:

| Sub-layer                                          | Purpose                                                     |
| -------------------------------------------------- | ----------------------------------------------------------- |
| `GroupNorm(32, channels)`                          | Pre-norm before entering transformer                        |
| `Conv2d(ch → ch, k=1)`                             | Input projection (conv_input)                               |
| Reshape `(B,C,H,W)` → `(B, H*W, C)`                | Treat spatial positions as a sequence                       |
| **Sub-block 1 — Self-Attention:**                  |                                                             |
| `LayerNorm(channels)`                              | Pre-norm                                                    |
| `SelfAttention(n_heads, channels)`                 | Each spatial position attends to all others in the latent   |
| Residual add                                       |                                                             |
| **Sub-block 2 — Cross-Attention:**                 |                                                             |
| `LayerNorm(channels)`                              | Pre-norm                                                    |
| `CrossAttention(n_heads, channels, d_context=768)` | Latent positions attend to all 77 CLIP text tokens          |
| Residual add                                       |                                                             |
| **Sub-block 3 — Feed-Forward (GeGLU):**            |                                                             |
| `LayerNorm(channels)`                              | Pre-norm                                                    |
| `Linear(ch → 4*ch*2)`                              | Project to 8x dimension, split into `x` and `gate` halves   |
| `x * GELU(gate)`                                   | GeGLU gating — gate controls how much each neuron activates |
| `Linear(4*ch → ch)`                                | Project back to original dimension                          |
| Residual add                                       |                                                             |
| Reshape back to `(B,C,H,W)`                        | Restore spatial structure                                   |
| `Conv2d(ch → ch, k=1)`                             | Output projection (conv_output)                             |
| Long residual add                                  | `output + original_input` (skip over the entire block)      |

#### `SwitchSequential`

A custom `nn.Sequential` that routes the three inputs `(x, context, time)` to the correct layer:

- If layer is `UNET_AttentionBlock` → pass `(x, context)`
- If layer is `UNET_ResidualBlock` → pass `(x, time)`
- Otherwise → pass `(x)` only

#### `Upsample(channels)` (UNet)

| Sub-layer                                | Purpose                                                         |
| ---------------------------------------- | --------------------------------------------------------------- |
| `F.interpolate(scale=2, mode='nearest')` | Doubles spatial resolution using nearest-neighbor interpolation |
| `Conv2d(ch → ch, k=3, p=1)`              | Smooths upsampled features and removes checkerboard artifacts   |

---

### Attention Modules (`attention.py`)

#### `SelfAttention(n_heads, d_embed)`

Standard multi-head scaled dot-product self-attention:

| Step                 | Operation                                | Shape change                                 |
| -------------------- | ---------------------------------------- | -------------------------------------------- |
| Input projection     | `Linear(d_embed → d_embed*3)` then split | `(B, S, D) → Q,K,V each (B, S, D)`           |
| Split heads          | reshape + transpose                      | `(B, S, D) → (B, H, S, D/H)`                 |
| Attention scores     | `Q @ K^T / sqrt(d_head)`                 | `(B, H, S, S)`                               |
| Optional causal mask | set upper-triangle to `-inf`             | prevents future token leakage (used in CLIP) |
| Softmax              | normalize scores                         | `(B, H, S, S)`                               |
| Weighted sum         | `scores @ V`                             | `(B, H, S, D/H)`                             |
| Merge heads          | transpose + reshape                      | `(B, S, D)`                                  |
| Output projection    | `Linear(D → D)`                          | `(B, S, D)`                                  |

#### `CrossAttention(n_heads, d_embed, d_cross)`

Same as self-attention but Q comes from latents and K, V come from text context:

| Projection                             | Source                                     | Output shape  |
| -------------------------------------- | ------------------------------------------ | ------------- |
| `q_proj` `Linear(d_embed → d_embed)`   | latent `x`                                 | `(B, S_q, D)` |
| `k_proj` `Linear(d_cross → d_embed)`   | text context `y`                           | `(B, 77, D)`  |
| `v_proj` `Linear(d_cross → d_embed)`   | text context `y`                           | `(B, 77, D)`  |
| Attention                              | `Q @ K^T / sqrt(d_head)` → softmax → `@ V` | `(B, S_q, D)` |
| `out_proj` `Linear(d_embed → d_embed)` | —                                          | `(B, S_q, D)` |

This is how the image latents "read" the text: each spatial pixel queries all 77 text tokens and weights them by relevance.

---

### DDPM Sampler (`ddpm.py`)

Not a neural network layer, but the mathematical engine controlling noise levels.

| Attribute            | Value                                    | Meaning                                                           |
| -------------------- | ---------------------------------------- | ----------------------------------------------------------------- |
| `num_training_steps` | 1000                                     | Total noise levels used during training                           |
| `beta_start`         | 0.00085                                  | Minimum noise variance (very small noise at early steps)          |
| `beta_end`           | 0.012                                    | Maximum noise variance (heavily noisy at final training step)     |
| `betas`              | `linspace(sqrt(0.00085), sqrt(0.012))^2` | Quadratic schedule — noise increases slowly at first, then faster |
| `alphas`             | `1 - betas`                              | Signal retention per step                                         |
| `alphas_cumprod`     | `cumprod(alphas)`                        | Total signal retained from step 0 to step `t`                     |

**Key methods:**

| Method                           | Purpose                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `set_inference_timesteps(n)`     | Selects `n` evenly spaced steps from the 1000-step training schedule                                  |
| `set_strength(strength)`         | For img2img: skips the first `(1-strength)*n` steps, starting denoising from a partially noisy latent |
| `add_noise(latents, t)`          | Forward process: `x_t = sqrt(ᾱ_t)*x_0 + sqrt(1-ᾱ_t)*ε`                                                |
| `step(t, latents, model_output)` | Reverse process: estimates `x_{t-1}` from `x_t` and predicted noise `ε_θ`                             |
| `_get_variance(t)`               | Computes posterior variance `β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t`                                         |

---

## References

- Rombach et al. (2022) — [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Radford et al. (2021) — [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion v1.5 weights on Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5) — RunwayML
