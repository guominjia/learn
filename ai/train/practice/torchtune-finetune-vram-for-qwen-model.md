# Why Does Fine-Tuning a 3B Model Consume 41GB of VRAM?

> A practical breakdown of GPU memory usage during LoRA fine-tuning with torchtune on Qwen2.5-3B-Instruct.

## The Problem

Running a LoRA fine-tune on Qwen2.5-3B-Instruct with a single A6000 (48GB) showed an alarming VRAM usage:

```
| GPU  Name                 | Memory-Usage |
|---------------------------|--------------|
|   0  NVIDIA RTX A6000     | 41969MiB / 49140MiB |
```

Nearly **42GB** for a model with only 3 billion parameters — on a card that theoretically should handle 20B+ models. What's going on?

---

## Breaking Down the 41GB

Training VRAM is not just "model weights". It's the sum of several components:

| Component | Estimated Size | Notes |
|---|---|---|
| Model weights (bf16) | ~6 GB | 3B params × 2 bytes |
| Forward activations | **~25–30 GB** | The main culprit — see below |
| LoRA params + optimizer states | ~2–4 GB | Only LoRA layers are trained |
| CUDA kernels / framework overhead | ~2–3 GB | PyTorch, NCCL, etc. |

The math adds up to ~35–43 GB, which matches the observed usage.

---

## The Root Cause: Unbounded Sequence Length on a Reasoning Dataset

The dataset used was `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` — a **reasoning chain dataset** where sequences can easily reach 8,000+ tokens. The torchtune config had:

```yaml
tokenizer:
  max_seq_len: null   # No truncation!
```

Forward pass activation memory scales roughly as:

$$\text{Activation Memory} \propto \text{batch\_size} \times \text{seq\_len} \times \text{hidden\_dim} \times \text{num\_layers}$$

With `max_seq_len: null`, long sequences are passed through untruncated, causing activation memory to balloon. A sequence of 8k tokens uses roughly **16× more activation memory** than one of 512 tokens.

torchtune even printed a warning about it:

```
INFO: Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't.
Enabling activation offloading should reduce memory further.
```

---

## The "48GB Can Train 20B" Myth

This common claim is based on **inference memory** — storing only model weights:

$$\text{Inference VRAM} \approx \text{params} \times \text{bytes\_per\_param}$$

Training is a different story. You also need:

- **Activations** stored for the backward pass
- **Gradients** for each trainable parameter
- **Optimizer states** (AdamW stores two momentum tensors per parameter)

In practice, training requires **4–8× more VRAM than inference** for the same model. A 3B model at ~6GB inference can easily consume 40GB during training with long sequences.

---

## How to Fix It

### 1. Cap the Sequence Length (Highest Impact)

```yaml
tokenizer:
  max_seq_len: 2048   # or 4096 depending on available VRAM
```

This alone can cut activation memory by 4–16×, depending on your dataset's average sequence length.

### 2. Enable Activation Offloading

```yaml
enable_activation_checkpointing: true
enable_activation_offloading: true   # offload activations to CPU RAM
```

Activation checkpointing recomputes activations during backward pass instead of storing them, trading compute for memory. Offloading goes further by moving stored tensors to CPU RAM.

### 3. Reduce Batch Size with More Gradient Accumulation

```yaml
batch_size: 1
gradient_accumulation_steps: 16   # effective batch size stays the same
```

Each sample in a batch is processed independently and its activation is held in VRAM simultaneously. Halving the batch size roughly halves activation memory.

---

## Expected VRAM After Fixes

With `max_seq_len: 2048` + `enable_activation_offloading: true` + `batch_size: 1`:

| Component | Before | After |
|---|---|---|
| Activations | ~28 GB | ~3–5 GB |
| Weights | ~6 GB | ~6 GB |
| Optimizer + LoRA | ~4 GB | ~4 GB |
| **Total** | **~41 GB** | **~15–18 GB** |

At 15–18GB, a 48GB A6000 has comfortable headroom to scale up to a 7B or even 14B Qwen model using the same LoRA approach.

---

## Key Takeaways

- **Training VRAM ≠ Inference VRAM.** The difference can be 4–8×.
- **Sequence length dominates activation memory**, especially with reasoning/chain-of-thought datasets.
- **Always set `max_seq_len`** when training on datasets with potentially long examples.
- `enable_activation_offloading: true` is essentially free if you have spare CPU RAM.
- torchtune's warning logs are worth reading — they often point directly at the problem.
