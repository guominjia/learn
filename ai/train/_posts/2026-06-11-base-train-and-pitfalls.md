---
layout: post
title: "Transformer Training Pitfalls: OOM, Attention Complexity, and Getting the Objective Right"
date: 2026-06-11
tags: [transformer, pytorch, oom, attention]
---

Building a Transformer from scratch is a great exercise — until the process gets `Killed` with no useful error message. This post walks through two common pitfalls encountered when training a small Transformer on CPU: OOM caused by quadratic attention, and training on the wrong objective.

---

## Pitfall 1: The Process Gets Killed — It's OOM, Not a Bug

### Symptom

Training starts, the first couple of encoder layers print their shapes fine, and then the process exits silently. No Python traceback, just:

```
torch.Size([8, 5000, 512])
torch.Size([8, 5000, 512])
Killed
```

The exit code is **137** — that's Linux's `SIGKILL`, sent by the **OOM killer** when the system runs out of memory.

### Root Cause: The $O(L^2)$ Attention Matrix

The problem is not in the encoder layers that printed correctly. The issue is in the **attention score tensor** that gets allocated right after.

With a configuration of:

| Parameter | Value |
|---|---|
| `batch_size` | 8 |
| `num_heads` | 8 |
| `max_seq_length` | 5000 |
| `d_model` | 512 |

Standard self-attention computes $QK^T$, which produces a tensor of shape:

$$
[B, H, L, L] = [8, 8, 5000, 5000]
$$

That's:

$$
8 \times 8 \times 5000 \times 5000 = 1.6 \times 10^9 \text{ elements}
$$

At `float32` (4 bytes each), just **one** attention score tensor is:

$$
1.6 \times 10^9 \times 4 \approx 6.4 \text{ GB}
$$

During training you're also holding `attn_probs`, `Q/K/V`, intermediate activations for backprop, and a large output logit tensor of shape `[8, 5000, 10000]` (~1.6 GB on its own). The system hits its memory ceiling and kills the process.

The early layers print successfully *because they're cheap*. The blow-up happens silently later when a new large tensor allocation fails.

---

## How Large Models (GPT, Claude, Qwen) Handle Memory

Large model training doesn't "brute-force" through memory. It stacks several complementary techniques:

### 1. Gradient Accumulation Instead of Large Batches

Each GPU uses a tiny **micro-batch**. Gradients are accumulated over several steps before an optimizer update:

$$
\text{effective batch} = \text{micro\_batch} \times \text{accum\_steps} \times \text{data\_parallel\_size}
$$

The large effective batch size is virtual; each individual step stays cheap.

### 2. Mixed Precision (`fp16` / `bf16`)

Parameters, activations, and most intermediate values are stored in 16-bit instead of 32-bit. This roughly halves memory usage across the board.

### 3. Activation Checkpointing (Gradient Checkpointing)

Instead of keeping all intermediate activations in memory for backprop, only **checkpoint tensors at layer boundaries** are saved. Activations in between are recomputed on demand during the backward pass.

- **Pro:** Significant memory reduction (often 4–8×)
- **Con:** ~30% more compute

### 4. Parameter and Optimizer State Sharding (ZeRO / FSDP)

The Adam optimizer stores not just parameters, but also two momentum buffers — often larger than the parameters themselves. ZeRO and PyTorch FSDP shard all of these across GPUs so no single device holds a full copy.

### 5. Fused Attention Kernels (FlashAttention)

Rather than materializing the full $[B, H, L, L]$ attention matrix in HBM:

```python
# Naive — stores the full L×L matrix
attn_scores = Q @ K.transpose(-2, -1)   # [B, H, L, L]  ← the memory killer
attn_probs  = attn_scores.softmax(-1)
out         = attn_probs @ V
```

FlashAttention (and PyTorch's `scaled_dot_product_attention`) fuses these into a tiled kernel that operates in SRAM, **never writing the $L \times L$ matrix to main memory**. Memory footprint drops from $O(L^2)$ to $O(L)$.

### 6. Controlled Sequence Length

Large models don't train at their full context window from day one. Since attention cost scales as $O(L^2)$, pushing $L$ too high too early is expensive and often unnecessary.

A common curriculum:

1. Start training at a short length (e.g., 512) until the loss plateau stabilizes
2. Progressively extend context length in stages
3. Use **sequence packing** — fill each batch slot with multiple short documents instead of padding to a fixed length
4. Use **length-bucketed batching** — group similar-length sequences together to minimize wasted padding compute
5. Treat very long context as a separate fine-tuning stage

For comparison, naively setting `batch_size=8`, `seq_len=5000`, `d_model=512`, and 6 encoder layers is extremely heavy for a standard implementation. Large models only reach such configurations after careful scaling and with all the other techniques in this list in place.

---

### 7. Sparse / Approximate Attention for Long Sequences

If you genuinely need very long sequences, standard full $L \times L$ attention is not the right tool. The common alternatives all share the same core idea: **avoid materializing the full attention map**.

| Approach | Core Idea |
|---|---|
| Sliding window attention | Each token only attends to a local window of size $w$; cost becomes $O(L \cdot w)$ |
| Block sparse attention | Attention is computed only within and between a fixed set of blocks |
| Grouped / local attention | Mix of local windows and a small set of global tokens |
| Linear attention / Performer | Approximate softmax attention with kernel functions; $O(L)$ complexity |
| Ring attention | Distributes the sequence across devices, allowing very long sequences across multiple GPUs |

PyTorch's `scaled_dot_product_attention` supports an `attn_mask` argument and dispatches to FlashAttention when available, making it the simplest first upgrade before reaching for these more advanced schemes.

---

## The Fix: Reduce `max_seq_length`

Setting `max_seq_length = 256` is sufficient to resolve the OOM:

```
[2026-05-13 12:52:52] Epoch 0,  Batch 9,  Loss: 1.5416
[2026-05-13 12:52:55] Epoch 0,  Batch 19, Loss: 1.0492
[2026-05-13 12:53:01] Epoch 1,  Batch 9,  Loss: 0.0815
[2026-05-13 12:53:04] Epoch 1,  Batch 19, Loss: 0.1235
...
[2026-05-13 12:54:16] Epoch 9,  Batch 19, Loss: 0.0039
[2026-05-13 12:55:45] Epoch 19, Batch 19, Loss: 0.0021
```

Training is stable and loss drops smoothly. The memory difference is dramatic:

$$
5000^2 = 25{,}000{,}000 \quad \text{vs} \quad 256^2 = 65{,}536
$$

The attention matrix shrinks by a factor of **~381×**.

However, notice how fast the loss drops — near zero by epoch 1. That's a red flag, explained next.

---

## Pitfall 2: Training on the Wrong Objective

### The Bug

The initial training loop passed the same tensor as both input and target:

```python
pred = model(src)
loss = criterion(pred.view(-1, vocab_size), src.view(-1))
```

This is an **identity mapping** task — the model just learns to copy its input, which is trivially easy. Loss near zero within the first epoch is expected and meaningless here.

### The Fix: Autoregressive Next-Token Prediction

The standard language model objective is:

$$
p(x_t \mid x_1, x_2, \dots, x_{t-1})
$$

Implemented as a one-position shift between input and target:

```python
pred = model(src[:, :-1])                              # input: tokens 0..T-2
loss = criterion(
    pred.view(-1, vocab_size),
    src[:, 1:].contiguous().view(-1)                   # target: tokens 1..T-1
)
```

The loss curve now looks like real learning:

```
Epoch 0,  Batch 9,  Loss: 3.7436   ← near random initialization
Epoch 0,  Batch 19, Loss: 2.9196
Epoch 1,  Batch 9,  Loss: 2.0779
Epoch 2,  Batch 9,  Loss: 1.1077
Epoch 4,  Batch 19, Loss: 0.1576
...
Epoch 19, Batch 19, Loss: 0.5158   ← still has variance
```

Loss starts high (close to $\ln(\text{vocab\_size})$, which is the random-chance baseline) and gradually decreases. The remaining variance between batches is normal for a small dataset with a small batch size.

---

## What This Model Actually Is

The architecture has an encoder and a decoder defined, but only the encoder is used in the forward pass:

```python
# In Transformer.forward()
enc_out = self.encoder(src, mask)
logits  = self.fc(enc_out)           # encoder output → vocab logits
# decoder is never called
```

Combined with the causal (lower-triangular) mask applied inside the encoder, this is functionally a **causal language model** — closer to GPT than to BERT or a seq2seq Transformer.

| Property | This model |
|---|---|
| Architecture | Encoder-only |
| Attention mask | Causal (lower triangular) |
| Training objective | Next-token prediction |
| Closest analog | GPT-1 / decoder-only LM |

---

## Key Takeaways

1. **Exit code 137 on Linux = OOM kill.** Look at the attention matrix shape, not the encoder output shape.
2. **Attention complexity is $O(L^2)$.** Reducing sequence length from 5000 to 256 shrinks the attention matrix by ~381×.
3. **Loss near zero in epoch 1 is a red flag.** Check whether the task is trivially easy (e.g., input == target).
4. **Next-token prediction requires a one-position shift** between input and target sequences.
5. **Low training loss does not imply good generalization.** Always validate on a held-out set and test actual generation quality.

---

## Recommended Next Steps

- Add a validation split and track val loss per epoch
- Implement a `generate()` function to inspect actual model outputs
- Replace naive attention with `torch.nn.functional.scaled_dot_product_attention` for better memory efficiency
- Add gradient checkpointing (`torch.utils.checkpoint`) if scaling up sequence length
