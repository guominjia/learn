---
layout: post
title: "Understanding PyTorch Scaled Dot-Product Attention"
date: 2026-06-12
tags: [pytorch, attention, transformer, gqa, flashattention]
---

`scaled_dot_product_attention` is one of the most important primitives in modern Transformer training and inference. In this post, we walk through how PyTorch exposes this API, how tensor shapes are interpreted, why causal masks use `tril`, and how GQA (Grouped Query Attention) is implemented.

Reference: [scaled_dot_product_attention](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py#L6341)

## 1) Why `_add_docstr` appears around `scaled_dot_product_attention`

In PyTorch, a common pattern is:

```python
scaled_dot_product_attention = _add_docstr(
    torch._C._nn.scaled_dot_product_attention,
    r"""scaled_dot_product_attention(...) -> Tensor
    ...
    """,
)
```

This does two things:

1. `torch._C._nn.scaled_dot_product_attention` is the real backend implementation (C++/CUDA).
2. `_add_docstr(...)` attaches a Python docstring so `help()` and `__doc__` show readable API docs.

The function object is still the same operator; only documentation is attached at the Python level.

This pattern appears in many ops as well:

| Python symbol | Backend entry |
|---|---|
| `conv1d` | `torch.conv1d` |
| `avg_pool2d` | `torch._C._nn.avg_pool2d` |
| `linear` | `torch._C._nn.linear` |
| `gelu` | `torch._C._nn.gelu` |
| `one_hot` | `torch._C._nn.one_hot` |

By contrast, some APIs are written with a Python wrapper (`def ...`) for argument checks/dispatch, then call into backend ops for the heavy compute.

## 2) Interpreting SDPA tensor shapes

PyTorch documentation describes shapes like:

```text
query: (N, ..., Hq, L, E)
key:   (N, ..., H,  S, E)
value: (N, ..., H,  S, Ev)
```

From the right side:

| Negative index | `query` | `key` |
|---|---|---|
| `-1` | `E` (embedding dim) | `E` (embedding dim) |
| `-2` | `L` (target length) | `S` (source length) |
| `-3` | `Hq` (query heads) | `H` (KV heads) |

That is why code often uses:

```python
L, S = query.size(-2), key.size(-2)
```

Using negative indices makes the code robust when there are arbitrary leading batch dimensions (`N, ...`).

## 3) `tril()` and causal masking

`tril()` means lower triangle (not upper triangle):

```python
temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
```

For `L = S = 4`, this gives:

```text
[[True,  False, False, False],
 [True,  True,  False, False],
 [True,  True,  True,  False],
 [True,  True,  True,  True ]]
```

- `True`: allowed attention
- `False`: masked attention

This is exactly the causal rule for autoregressive decoding: each position can attend only to itself and previous positions, never future tokens.

Typical usage:

```python
attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
```

Masked positions receive `-inf`, and after softmax their probability becomes `0`.

Quick memory aid:

| Function | Meaning | Kept region |
|---|---|---|
| `.tril()` | triangle lower | diagonal and below |
| `.triu()` | triangle upper | diagonal and above |

## 4) Why `repeat_interleave` is used for GQA

GQA changes the number of heads:

```text
MHA: Hq = H       (one KV head per query head)
GQA: Hq > H       (multiple query heads share one KV head)
MQA: H = 1        (extreme case of GQA)
```

Example (LLaMA-style): `Hq = 32`, `H = 8`.

The implementation aligns head counts by repeating KV heads:

```python
if enable_gqa:
    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
```

So KV heads expand from 8 to 32, matching query heads for direct batched attention math.

Conceptually:

```text
before: [kv0, kv1, ..., kv7]
after:  [kv0,kv0,kv0,kv0, kv1,kv1,kv1,kv1, ..., kv7,kv7,kv7,kv7]
```

## 5) Why GQA matters in practice

GQA primarily reduces KV-cache memory in autoregressive inference.

| Metric | MHA | GQA |
|---|---|---|
| KV head count | same as query heads | fewer than query heads |
| KV-cache size | larger | smaller |
| Inference bandwidth pressure | higher | lower |
| Quality | baseline | usually close to MHA |

Since KV cache grows linearly with sequence length, reducing KV heads can significantly cut memory and improve throughput. That is why many production LLMs (for example LLaMA-family, Mistral-family, and Gemma-family models) adopt GQA.

## Final takeaway

For `scaled_dot_product_attention`, Python usually provides API/doc surface, while high-performance kernels run in C++/CUDA. Understanding shape indexing, causal masking, and GQA head alignment helps you reason about both correctness and performance when working with Transformer internals.