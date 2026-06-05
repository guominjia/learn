# DeepSeek V4 Attention vs Standard Multi-Head Attention

Transformer attention started from a clean, symmetric design: independent $Q$, $K$, and $V$ projections per head, full-dimensional rotary encoding, and a single output projection. DeepSeek V4 keeps the same high-level objective but re-engineers almost every internal detail to reduce inference cost, especially KV-cache bandwidth, while preserving model quality on long-context workloads.

This post explains those differences from first principles and why they matter in practice.

## 1) Baseline: What Standard MHA Looks Like

In classic multi-head attention (MHA), we typically use:

- $d_{model} = n_{heads} \times d_{head}$
- independent projections: $W_Q, W_K, W_V, W_O$

For input $X$:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
	ext{Attn}(Q,K,V)=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}+\text{mask}\right)V
$$

Then concatenate heads and project back with $W_O$.

This design is elegant and expressive, but autoregressive decoding becomes memory-bandwidth bound because KV cache grows with sequence length, number of layers, and number of KV heads.

## 2) DeepSeek V4: Core Design Changes

DeepSeek V4 attention is not a small tweak; it is a heavily engineered variant with several coupled ideas.

### A. Shared-KV MQA style: one KV head and shared K/V tensor

- V4 uses `num_key_value_heads = 1`.
- A single `kv_proj` output is used as both key and value.

This dramatically shrinks KV cache and decode-time memory traffic.

### B. Two-stage low-rank Q projection (LoRA-style factorization)

Instead of one giant projection from hidden size to all query heads, V4 uses:

- `q_a_proj`: down-project
- normalization
- `q_b_proj`: up-project to full query dimension

Why: internal attention dimension is very large (for example, $64 \times 512 = 32768$), so direct projection is expensive.

### C. Factorized / grouped output projection

Output projection is also decomposed (grouped low-rank path) rather than a single dense giant matrix, reducing parameters and compute.

### D. Partial RoPE and inverse rotation on output

V4 applies rotary position encoding to part of head channels and later applies an inverse-style rotation on attention output. This compensates for the shared K=V path so positional behavior remains well-formed.

### E. Learnable attention sink

Before softmax, V4 appends one extra learnable sink logit per head. Intuition: when no token should receive high probability, softmax still must sum to 1; sink absorbs that excess mass.

### F. Internal compressor for long context

For selected layers, V4 augments sliding-window KV with compressed long-range memory entries, improving long-context retrieval without storing all raw historical KV equally.

## 3) Side-by-Side: MHA vs DeepSeek V4

| Aspect | Standard MHA | DeepSeek V4 Attention |
|---|---|---|
| Q projection | Single dense projection | Two-stage low-rank projection |
| K/V heads | Usually many (MHA/GQA style) | Single KV head |
| K and V relation | Independent tensors | Shared tensor (K=V source) |
| RoPE usage | Usually full head channels | Partial channels + inverse step at output |
| Softmax logits | Token logits only | Token logits + learnable sink |
| Long context handling | Raw KV cache growth | Sliding KV + compressed memory path |
| Output projection | Single dense matrix | Grouped/factorized projection |

## 4) Why This Trade-off Exists

The key pressure is decode-time efficiency.

- Standard MHA maximizes representational freedom per head, but KV cache is large.
- Extreme KV sharing improves throughput and memory use, but can hurt expressivity.
- V4 tries to recover quality using larger head dimensions, low-rank factorized projections, sink regularization, and compressed long-range memory.

So V4 is best viewed as a systems-aware attention design: optimize for modern LLM serving constraints first, then add mechanisms to retain quality.

## 5) Quick Ecosystem Positioning

- Many recent models (for example, Qwen/GLM/MiniMax softmax-attention paths) still follow mainstream GQA-style attention with independent K and V projections.
- DeepSeek V3 already moved toward latent/low-rank ideas.
- DeepSeek V4 pushes further with shared-KV, output factorization, sink logits, and built-in compression.

In short, this is not “better MHA” in a pure theoretical sense; it is a different point in the quality-efficiency design space.

## 6) Practical Takeaway

If you implement textbook MHA, your understanding is correct for classic Transformer architecture. DeepSeek V4 intentionally breaks several textbook equalities and symmetries to make large-scale, long-context inference cheaper.

The design pattern is clear:

1. reduce KV memory and bandwidth aggressively,
2. re-inject representational capacity where needed,
3. compensate with architecture-level tricks (sink, partial/inverse RoPE, compression).

That combination is what makes DeepSeek V4 attention look unusual compared with the original Multi-Head Attention formulation.