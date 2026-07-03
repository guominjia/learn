---
layout: post
title: "Rotary Position Embedding in GLM, DeepSeek-V3, and Qwen3"
date: 2026-07-03
tags: [transformer, attention, rope, positional-encoding, llm]
---

Modern decoder-only language models usually do not use the original Transformer's input-level absolute positional encoding anymore. Instead, many of them use **RoPE**, short for **Rotary Position Embedding**.

In Hugging Face Transformers, the implementations in these model files are highly related:

- `src/transformers/models/glm/modular_glm.py`
- `src/transformers/models/deepseek_v3/modular_deepseek_v3.py`
- `src/transformers/models/qwen3/modular_qwen3.py`

They differ in model-specific details, but the core idea is the same: compute sinusoidal position-dependent factors, then apply them as rotations to the query and key vectors in attention.

This post explains what these implementations have in common, where they differ, and why RoPE is more than a small engineering trick.

## 1. Background: From Positional Encoding to RoPE

The original Transformer paper, [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762), adds positional encodings directly to token embeddings:

$$
h_i = x_i + p_i
$$

Then queries, keys, and values are projected from the same position-aware hidden state:

$$
q_i = h_i W_Q, \quad k_i = h_i W_K, \quad v_i = h_i W_V
$$

So in the original design, positional information indirectly enters `Q`, `K`, and `V`.

RoPE changes the insertion point. Instead of adding position vectors to the input hidden states, it rotates `Q` and `K` inside attention:

$$
q_i' = R_i q_i
$$

$$
k_j' = R_j k_j
$$

The value vector is usually left unrotated:

$$
v_j' = v_j
$$

This is an important design difference. In RoPE, position mainly affects **which tokens attend to which other tokens**, rather than directly modifying the content carried by `V`.

## 2. The Core RoPE Operation

RoPE does not rotate the whole head vector by a single angle. Instead, it splits the hidden dimensions into two-dimensional subspaces.

For example, if a query head is:

```text
[x0, x1, x2, x3, x4, x5, x6, x7]
```

RoPE treats it as pairs:

```text
(x0, x1), (x2, x3), (x4, x5), (x6, x7)
```

Each pair is rotated by a position-dependent angle. For one pair:

$$
\begin{bmatrix}
x_0' \\
x_1'
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
$$

That gives:

$$
x_0' = x_0\cos(m\theta) - x_1\sin(m\theta)
$$

$$
x_1' = x_1\cos(m\theta) + x_0\sin(m\theta)
$$

In code, this is usually expressed as:

```python
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

where `rotate_half()` transforms:

```text
[x0, x1, x2, x3]
```

into:

```text
[-x1, x0, -x3, x2]
```

This is not random mixing. It is a structured, norm-preserving rotation applied independently in many two-dimensional planes.

## 3. The Complex Number View

Another useful way to understand RoPE is to combine each even/odd dimension pair into a complex number:

$$
(x_0, x_1) \rightarrow x_0 + i x_1
$$

At position $m$, RoPE multiplies it by a complex phase:

$$
e^{im\theta}
$$

So:

$$
(x_0 + i x_1)e^{im\theta}
$$

Expanding it gives:

$$
(x_0\cos m\theta - x_1\sin m\theta) + i(x_1\cos m\theta + x_0\sin m\theta)
$$

This is exactly the two-dimensional rotation above.

So RoPE can be seen as multiplying complex-valued query/key coordinates by position-dependent phases.

## 4. Why Relative Position Appears Naturally

The most important property of RoPE is not just that it rotates vectors. The key property is what happens inside the attention score.

Attention scores are computed from query-key dot products:

$$
QK^T
$$

With RoPE:

$$
(R_i q_i)^T(R_j k_j)
$$

Because rotation matrices are orthogonal and compose by angle differences, this becomes:

$$
(R_i q_i)^T(R_j k_j) = q_i^T R_{j-i} k_j
$$

The attention score now depends on $j-i$, the relative distance between the query position and the key position.

This is the central reason RoPE works well: **relative position is built into the mathematical structure of attention itself**.

## 5. Comparing the Hugging Face Implementations

The three model implementations follow the same RoPE idea, but each model adapts it to its own architecture.

### GLM

In `src/transformers/models/glm/modular_glm.py`, GLM defines its own `rotate_half()` and `apply_rotary_pos_emb()`.

The implementation uses even/odd interleaving:

```python
x1 = x[..., 0::2]
x2 = x[..., 1::2]
return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

This means the pairs are:

```text
(x0, x1), (x2, x3), (x4, x5), ...
```

GLM also converts the cosine and sine tensors into the interleaved layout:

```python
cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
```

If the original cosine vector is conceptually:

```text
[c0, c1, c2, c3, c0, c1, c2, c3]
```

this converts it into:

```text
[c0, c0, c1, c1, c2, c2, c3, c3]
```

That matches the even/odd dimension pairing.

In many GLM configurations, `rotary_dim` equals the whole attention head dimension, so all query/key head dimensions participate in RoPE. But the implementation still keeps `q_pass` and `k_pass` to support the general case where only part of the head is rotated.

### Qwen3

In `src/transformers/models/qwen3/modular_qwen3.py`, Qwen3 reuses the RoPE utility from Qwen2:

```python
from ..qwen2.modeling_qwen2 import apply_rotary_pos_emb
```

Inside attention, Qwen3 computes query, key, and value projections, applies RMSNorm to query and key heads, and then applies RoPE to query and key:

```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

The important pattern is the same as GLM:

```text
RoPE applies to Q and K, not V.
```

Qwen3 adds model-specific details such as query/key normalization and sliding-window attention support, but the positional mechanism remains standard RoPE.

### DeepSeek-V3

DeepSeek-V3 is more specialized.

In `src/transformers/models/deepseek_v3/modular_deepseek_v3.py`, it imports the Llama-style RoPE utilities:

```python
from ..llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, rotate_half
```

It also defines `DeepseekV3RotaryEmbedding` as a subclass of `LlamaRotaryEmbedding`.

The important architectural difference is that DeepSeek-V3 splits query/key dimensions into two parts:

```text
q_pass: dimensions without RoPE
q_rot:  dimensions with RoPE
```

and similarly for keys:

```text
k_pass: dimensions without RoPE
k_rot:  dimensions with RoPE
```

Only the rotary part is passed through RoPE:

```python
q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
```

Then the model concatenates the non-rotary and rotary parts back together:

```python
query_states = torch.cat((q_pass, q_rot), dim=-1)
key_states = torch.cat((k_pass, k_rot), dim=-1)
```

This is still the same idea: relative position enters through the rotated part of `Q` and `K`. DeepSeek-V3 simply makes the split explicit, which fits its multi-latent attention style and its separation between positional and non-positional query/key subspaces.

DeepSeek-V3 also supports an interleaved RoPE path through `apply_rotary_pos_emb_interleave()`, which reshapes dimensions before applying the same rotation formula.

## 6. What They Have in Common

Despite implementation details, GLM, DeepSeek-V3, and Qwen3 share the same RoPE pattern:

1. Compute or receive `cos` and `sin` from token positions.
2. Broadcast them to match query/key tensor shapes.
3. Rotate query and key vectors in two-dimensional subspaces.
4. Leave value vectors unrotated.
5. Compute attention with position-aware `QK^T`.

The shared conceptual flow is:

```text
hidden_states
	-> q_proj, k_proj, v_proj
	-> apply RoPE to Q and K
	-> attention score from rotated Q and K
	-> weighted sum over unrotated V
```

## 7. Why RoPE Is Better Than Simple Positional Addition

RoPE has several practical and theoretical advantages over the original input-level positional addition.

### 7.1 Position Directly Enters $QK^T$

Original absolute positional encoding adds position to token embeddings before projection. The model can learn position relationships, but relative distance is not explicitly encoded in the attention score.

RoPE directly modifies the query-key dot product:

$$
(R_i q_i)^T(R_j k_j) = q_i^T R_{j-i} k_j
$$

That means the attention score naturally depends on relative position.

This is valuable because language modeling often cares more about relative relationships than absolute indexes:

- previous token,
- nearby phrase,
- matching bracket,
- local code block,
- long-range dependency.

### 7.2 RoPE Does Not Directly Pollute `V`

With original input-level positional encoding:

$$
v_i = (x_i + p_i)W_V
$$

So `V` also contains positional information.

With RoPE, the usual pattern is:

$$
q_i' = R_i q_i, \quad k_i' = R_i k_i, \quad v_i' = v_i
$$

This keeps a clean division of responsibility:

| Tensor | Role | RoPE behavior |
|---|---|---|
| `Q` | What am I looking for? | Rotated |
| `K` | What can I match? | Rotated |
| `V` | What content do I return? | Not directly rotated |

Position affects matching, while content retrieval remains less directly distorted.

### 7.3 Better Length Extrapolation

Learned absolute position embeddings are tied to a maximum training length:

```text
pos_0, pos_1, ..., pos_2047
```

Positions beyond that range may be missing or poorly trained.

RoPE uses trigonometric functions:

$$
\cos(m\theta), \quad \sin(m\theta)
$$

These can be evaluated for positions beyond the training window.

This does not mean RoPE gives perfect infinite-length generalization. In practice, long-context models often combine RoPE with methods such as NTK scaling, YaRN, LongRoPE, or other frequency-scaling strategies. But RoPE gives a better mathematical starting point for extrapolation than learned absolute embeddings.

### 7.4 Rotation Is Norm-Preserving and Reversible

A two-dimensional rotation matrix is orthogonal:

$$
R^T R = I
$$

Therefore it preserves vector norms:

$$
\lVert Rx \rVert = \lVert x \rVert
$$

It is also reversible:

$$
R^{-1} = R^T
$$

So RoPE is not arbitrary noise injected into the representation. It is a stable, structured transformation that changes direction while preserving magnitude.

## 8. The RoFormer Paper

RoPE was proposed in the paper **RoFormer: Enhanced Transformer with Rotary Position Embedding**.

The authors include:

- Jianlin Su
- Yu Lu
- Shengfeng Pan
- Bo Wen
- Yunfeng Liu

The arXiv version appeared around April 2021 as [`arXiv:2104.09864`](https://arxiv.org/pdf/2104.09864).

The paper compared RoPE against multiple positional strategies, including:

- no positional encoding,
- absolute positional encoding,
- relative positional encoding,
- rotary positional encoding.

Experimentally, RoPE usually performed better, especially on tasks where relative distance, long text, or sequential structure matters.

However, the most important contribution is not only the benchmark result. The deeper value is the algebraic property:

$$
(R_i q_i)^T(R_j k_j) = q_i^T R_{j-i} k_j
$$

This property shows that relative position is not merely learned indirectly. It is built into the attention score by construction.

## 9. Summary

The GLM, DeepSeek-V3, and Qwen3 implementations in Hugging Face Transformers all use Rotary Position Embedding as their core positional mechanism.

Their details vary:

| Model | RoPE implementation detail |
|---|---|
| GLM | Defines an interleaved even/odd `rotate_half()` and converts `cos/sin` layout |
| Qwen3 | Reuses Qwen2 RoPE utilities and applies RoPE after query/key normalization |
| DeepSeek-V3 | Splits query/key into non-RoPE and RoPE subspaces, then rotates only the RoPE part |

But conceptually they all follow the same rule:

```text
rotate Q and K by position-dependent angles; do not directly rotate V
```

The reason this works so well is that attention is fundamentally about query-key matching. RoPE injects position exactly where matching happens: in $QK^T$.

In one sentence:

> RoPE is powerful because it turns absolute token positions into relative query-key geometry inside attention.

## References

- Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Hugging Face Transformers: [models source tree](https://github.com/huggingface/transformers/tree/main/src/transformers/models).
