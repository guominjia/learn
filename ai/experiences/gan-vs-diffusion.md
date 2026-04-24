# GAN vs Diffusion: A Deep Dive into Modern Generative AI

*April 2026*

---

## 1. The Two Paradigms: GAN and Diffusion

Generative AI has been dominated by two major model families over the past decade. Understanding their differences is foundational to understanding modern text-to-image and text-to-video systems.

### How They Work

**GAN (Generative Adversarial Network)** frames generation as a two-player game. A *generator* tries to produce realistic samples, while a *discriminator* tries to distinguish real from fake. They train adversarially until the discriminator can no longer tell the difference. Generation happens in a single forward pass.

**Diffusion models** take the opposite approach: they learn to reverse a gradual noising process. During training, Gaussian noise is incrementally added to real data. The model learns to denoise step by step. At inference time, it starts from pure noise and iteratively recovers a clean sample.

| Dimension | GAN | Diffusion |
|-----------|-----|-----------|
| Generation | Single forward pass | Hundreds of denoising steps |
| Mathematical basis | Game theory (minimax) | Markov chain + variational inference |
| Training objective | Fool the discriminator | Predict noise at each step (MSE) |

### Training Stability

This is where the two diverge most sharply in practice. GANs are notoriously difficult to train — they suffer from **mode collapse** (generating only a narrow subset of the data distribution), vanishing gradients, and extreme sensitivity to hyperparameters. Getting a GAN to converge well requires significant engineering effort.

Diffusion models, by contrast, have a simple and stable training objective. The loss is just mean squared error between the predicted and actual noise. They converge reliably and scale predictably with compute.

### Generation Quality and Diversity

GANs are fast — a single inference pass takes milliseconds. But they tend to undercover the full data distribution, missing modes. Diffusion models are slow (hundreds of neural network forward passes per sample), but they cover the distribution far more faithfully, producing greater diversity and higher fidelity.

### Controllability

Conditional generation is an afterthought in the GAN framework — it requires architectural changes like CGAN or complex techniques in StyleGAN. Diffusion models support **classifier-free guidance** natively, making text-conditioning, image-conditioning, and multi-modal control straightforward to implement.

### Representative Models

| | GAN | Diffusion |
|-|-----|-----------|
| Key models | StyleGAN, BigGAN, CycleGAN | DDPM, Stable Diffusion, DALL-E 3, Sora |
| Sweet spot | Real-time face synthesis, style transfer | Text-to-image, text-to-video, high-fidelity editing |

**Bottom line**: GAN is fast but fragile and diversity-limited. Diffusion is slow but stable, high-quality, and highly controllable. The mainstream has shifted decisively toward Diffusion.

---

## 2. Is Everything Text-to-Image and Text-to-Video Just Diffusion?

Not entirely — but Diffusion is dominant.

### Text-to-Image

| Model | Architecture |
|-------|-------------|
| Stable Diffusion, DALL-E 3, Imagen | Diffusion |
| DALL-E 1 | Transformer + VQVAE |
| Parti (Google) | Autoregressive Transformer |
| FLUX, Stable Diffusion 3 | Flow Matching (Diffusion variant) |
| GPT-4o image generation | Autoregressive (continuous tokens) |

### Text-to-Video

| Model | Architecture |
|-------|-------------|
| Sora (OpenAI) | Diffusion Transformer (DiT) |
| Wan (Alibaba), HunyuanVideo | Diffusion |
| CogVideo, Kling | Diffusion |
| VideoPoet (Google) | Autoregressive Transformer |
| Make-A-Video (Meta) | Diffusion |

### Emerging Alternatives

**Flow Matching** (used in FLUX, SD3) is a Diffusion variant that trains with straighter trajectories — same spirit, better efficiency.

**Autoregressive (AR) models** tokenize images or videos and generate them token-by-token, like a language model. This is the approach behind GPT-4o's native image generation and is gaining ground rapidly.

**VAR / MAR** are newer AR variants using multi-scale or continuous tokens, pushing the frontier further.

---

## 3. How Do You Tokenize an Image for Autoregressive Generation?

This is where AR image generation gets non-trivial. The naive approach — treating each pixel as a token with a vocabulary of $2^{24}$ colors — is completely intractable. There are two problems:

1. **Vocabulary too large**: $2^{24} \approx 16.7M$ possible tokens per position
2. **Sequence too long**: a 1024×1024 image has over 1 million pixels

### Solution 1: Compressed Latent Tokens

Instead of operating in pixel space, we first **compress the image with a VQ-VAE or VQGAN**:

- A 1024×1024 image is encoded to a $128 \times 128$ grid (8× spatial compression)
- Each position in the grid is assigned a discrete token from a codebook of size ~8192
- Sequence length: $128 \times 128 = 16{,}384$ tokens (manageable, though still long)

The codebook size of 8192 is orders of magnitude smaller than raw pixel space.

### Solution 2: Continuous Tokens (MAR)

**MAR (Masked Autoregressive)** sidesteps discrete quantization entirely. It predicts continuous latent vectors and uses a small diffusion head to decode each one. No codebook needed — vocabulary size is effectively infinite, but in a continuous sense.

### Solution 3: Multi-Scale Generation (VAR)

**VAR (Visual AutoRegressive)** generates images coarse-to-fine:

$$1\times1 \rightarrow 2\times2 \rightarrow 4\times4 \rightarrow \cdots \rightarrow 128\times128$$

Each scale conditions on all previous scales. Context length at any single step is short, and the model naturally captures global structure before local detail.

### The Full Pipeline

```
Original image (1024×1024×3)
    ↓ VAE / VQGAN encoder
Latent token grid (128×128, vocab ≈ 8192)
    ↓ AR Transformer (generates token by token)
Latent token grid
    ↓ VAE decoder
High-resolution image
```

The key insight: **never work in pixel space**. Always compress to a semantically meaningful latent space first.

---

## 4. What About Video? The Sequence Length Problem Gets Worse

If image tokenization is hard, video is exponentially harder. A 10-second clip at 24fps is 240 frames. At 1024×1024, that's an astronomical number of tokens.

### Spatial + Temporal Compression

The natural extension is to apply compression in **both space and time**:

```
Raw video: 240 frames × 1024×1024×3
    ↓ 3D VAE (8× spatial, 8× temporal compression)
Latent: 30 frames × 128×128
= 491,520 tokens
```

Almost half a million tokens is still far beyond what a standard Transformer can handle in its attention window.

### What the Industry Actually Does

**1. Lower resolution and frame rate than advertised**  
Most commercial text-to-video models generate at 720p or lower, at 16fps rather than 24fps. The "1080p" numbers in marketing often refer to upscaled output.

**2. Sparse attention patterns**  
Rather than full quadratic attention over all tokens, models use:
- Local spatial attention within each frame
- Temporal attention only to adjacent frames and keyframes
- Sora's DiT uses large **spacetime patches**, reducing token count significantly

**3. Chunked / sliding window generation**  
Generate short overlapping clips, then stitch them together using the last few frames of each chunk as conditioning for the next.

### Why Diffusion Beats AR for Video

This scalability gap is precisely why Diffusion remains dominant for video, even as AR catches up in images:

| | Autoregressive | Diffusion |
|-|---------------|-----------|
| Token sequence | Must fully unroll all tokens | Denoises in latent space holistically |
| Temporal consistency | Generated left-to-right, harder to maintain global coherence | Whole sequence optimized jointly |
| Long video | Extremely difficult to scale | Sliding window denoising works naturally |

### The Open Problem

There is no clean solution to long video generation today. The practical boundary in 2026 is roughly **20–60 seconds** for high-quality generation. Minute-long or hour-long video synthesis remains an open research problem.

The fundamental tension is:

$$\text{Quality} \times \text{Resolution} \times \text{Duration} = \text{Compute Budget}$$

Every current system makes significant sacrifices on at least one of these three axes.

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| GAN vs Diffusion | Diffusion wins on quality, stability, and controllability; GAN wins on speed |
| Text-to-image | Diffusion is mainstream; AR (GPT-4o style) is the fast-growing challenger |
| Text-to-video | Diffusion dominates; AR struggles with sequence length |
| Image tokenization | VQ-VAE/VQGAN compress pixels to manageable discrete tokens; continuous tokens (MAR) avoid quantization entirely |
| Video generation | 3D VAE + sparse attention + chunked generation; long video remains unsolved |

The trajectory is clear: **Diffusion is the present, AR is competing for the future**, and the battlefield is compute efficiency and sequence scalability.
