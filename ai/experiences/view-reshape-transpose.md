---
title: "Understanding view(), reshape(), and transpose() in PyTorch"
date: 2026-04-28
tags: [pytorch, tensor, transformer, deep-learning]
---

# Understanding `view()`, `reshape()`, and `transpose()` in PyTorch

When implementing multi-head attention in Transformers, you will inevitably encounter a line like this:

```python
self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
```

This single chain combines three distinct operations — linear projection, shape splitting, and axis swapping. If you have ever wondered why `transpose` is needed here, or why `reshape` alone is not enough, this post is for you.

---

## 1. The Problem: Splitting Heads in Multi-Head Attention

Suppose `Q` has shape `[8, 512, 512]` — a batch of 8 sequences, each with 512 tokens, each token represented as a 512-dimensional vector (`d_model = 512`).

We want 8 attention heads, so `d_k = d_model / num_heads = 64`.

The goal is to transform `Q` from `[8, 512, 512]` into `[8, 8, 512, 64]`, where the dimensions represent `[batch, heads, seq_len, d_k]`.

Here is how the chain works step by step:

### Step 1 — Linear projection: `self.W_q(Q)`

A `Linear(512, 512)` layer. The shape remains `[8, 512, 512]`.

### Step 2 — Splitting heads: `.view(batch_size, -1, self.num_heads, self.d_k)`

The last dimension (`512`) is reshaped into two dimensions: `num_heads × d_k = 8 × 64`.

$$
[8, 512, 512] \rightarrow [8, 512, 8, 64]
$$

At this point the axes mean: `[batch, seq_len, heads, d_k]`.

### Step 3 — Swapping axes: `.transpose(1, 2)`

We swap dimension 1 (`seq_len`) and dimension 2 (`heads`):

$$
[8, 512, 8, 64] \rightarrow [8, 8, 512, 64]
$$

Now the axes mean: `[batch, heads, seq_len, d_k]`. Each attention head gets its own `[512, 64]` matrix to work with independently.

---

## 2. Why Not Just `reshape`?

A common question: *"Can I skip `transpose` and just `reshape` directly into `[8, 8, 512, 64]`?"*

**No.** Here is why.

`reshape` only reinterprets the flat memory layout — it cannot swap axes. If you write:

```python
reshape(batch_size, self.num_heads, -1, self.d_k)
```

The output shape would be `[8, 8, 512, 64]`, but the **data is wrong**. Elements that should belong to different heads end up mixed together, because `reshape` simply re-chunks the underlying memory in order.

### A Concrete 2×6 Example

Consider a small tensor to make this crystal clear:

$$
X =
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 \\
7 & 8 & 9 & 10 & 11 & 12
\end{bmatrix}
$$

#### Route A: `reshape(2,3,2)` then `transpose(0,1)`

First, `reshape` splits each row into 3 groups of 2:

```python
X.reshape(2, 3, 2)
# [
#   [ [1, 2], [3, 4], [5, 6] ],
#   [ [7, 8], [9,10], [11,12] ]
# ]
```

The underlying memory order is unchanged: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12`.

`reshape(2,3,2)` simply tells PyTorch: "divide into 2 blocks, each block has 3 groups of 2 elements." It is pure re-chunking — no axis reordering.

Then `transpose(0, 1)` swaps axis 0 (size 2) and axis 1 (size 3), producing shape `[3, 2, 2]`:

```python
# [
#   [ [1, 2], [7, 8]  ],
#   [ [3, 4], [9, 10] ],
#   [ [5, 6], [11,12] ]
# ]
```

Notice how elements from **different rows** are now grouped together at each position. The operation works like a matrix transpose on the outer two axes: it gathers the i-th group from every block.

#### Route B: `reshape(3,2,2)` directly

```python
X.reshape(3, 2, 2)
# [
#   [ [1, 2], [3, 4] ],
#   [ [5, 6], [7, 8] ],
#   [ [9,10], [11,12] ]
# ]
```

The shape is the same `[3, 2, 2]`, but the data arrangement is **completely different**. Raw memory is simply sliced into consecutive chunks of 4 elements each.

**Key insight:** `reshape` re-chunks memory sequentially. `transpose` swaps the meaning of axes without moving data in memory. Even though both produce the same shape, the contents differ.

---

## 3. Mapping Back to Transformers

In the Transformer context:

$$
[batch, seq\_len, d\_model] = [8, 512, 512]
$$

After `view(batch_size, seq_len, num_heads, d_k)`:

$$
[8, 512, 8, 64]
$$

Each of the 512 tokens has its 512-dimensional vector split into 8 heads of 64 dimensions. The layout is still **token-major**: for each token, you see all of its head components side by side.

After `transpose(1, 2)`:

$$
[8, 8, 512, 64]
$$

The layout becomes **head-major**: for each head, you see all 512 tokens. This is exactly the format needed for computing attention — each head independently attends over the full sequence.

This mirrors our 2×6 example:

- `reshape` splits a large dimension into smaller groups
- `transpose` rearranges which grouping comes first

---

## 4. `view()` vs `reshape()` — Are They the Same?

Not exactly. Both change the shape of a tensor, but they differ in how they handle memory layout.

| | `view()` | `reshape()` |
|---|---|---|
| **Memory requirement** | Tensor must be contiguous | No requirement |
| **Data copy** | Never copies | Copies only when necessary |
| **Returns** | Always a view (shared memory) | View if possible, copy otherwise |

In short:

$$
\texttt{reshape} \approx \text{a more forgiving version of } \texttt{view}
$$

### When does `view()` fail?

After `transpose()`, a tensor is typically **non-contiguous** — the underlying memory order no longer matches the logical index order. Calling `view()` on such a tensor raises a `RuntimeError`:

```python
x = torch.arange(12).view(3, 4)
y = x.transpose(0, 1)  # shape [4, 3], non-contiguous
y.view(2, 6)            # RuntimeError!
```

`reshape()` handles this gracefully by copying the data into contiguous memory first:

```python
y.reshape(2, 6)          # works fine
```

### The `contiguous()` pattern

In many Transformer implementations you will see:

```python
attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
```

This is the explicit version of what `reshape` does automatically:

1. `transpose(1, 2)` — swaps axes (tensor becomes non-contiguous)
2. `.contiguous()` — copies data into a new contiguous memory layout
3. `.view(...)` — reinterprets the now-contiguous memory

Using `reshape` is equivalent and more concise:

```python
attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
```

### Why do people still use `view()`?

- **Intentionality.** Writing `view()` asserts that the tensor is already contiguous. It is a form of documentation.
- **Performance awareness.** `view()` guarantees zero copies. A `reshape()` might silently allocate new memory.
- **Legacy.** Older PyTorch code heavily uses `view()` because `reshape()` was added later (PyTorch 0.4).

In the original multi-head attention code:

```python
self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k)
```

The output of a `Linear` layer is always contiguous, so `view()` is safe here.

---

## 5. Summary Cheat Sheet

| Operation | What it does | Moves data in memory? |
|---|---|---|
| `view(shape)` | Reinterprets memory layout as a new shape | **No** (fails if non-contiguous) |
| `reshape(shape)` | Same as `view`, but copies if needed | Only when necessary |
| `transpose(d0, d1)` | Swaps two dimensions | **No** (just changes strides) |
| `permute(dims)` | Reorders all dimensions arbitrarily | **No** (just changes strides) |
| `contiguous()` | Copies data so memory matches logical order | **Yes** (always allocates) |

### Rules of Thumb

1. **Need to split a dimension** (e.g., `512 → 8 × 64`): use `view()` or `reshape()`.
2. **Need to swap axes**: use `transpose()` or `permute()`. `reshape` alone **cannot** do this correctly.
3. **Need `view()` after `transpose()`**: call `.contiguous()` first, or just use `reshape()`.
4. **Want zero-copy guarantee**: use `view()`.
5. **Want it to just work**: use `reshape()`.

---

## 6. Full Runnable Example

```python
import torch

X = torch.arange(1, 13).view(2, 6)
print("Original shape [2, 6]:")
print(X)
# tensor([[ 1,  2,  3,  4,  5,  6],
#         [ 7,  8,  9, 10, 11, 12]])

# Route A: reshape then transpose
A = X.reshape(2, 3, 2).transpose(0, 1)
print("\nreshape(2,3,2).transpose(0,1) -> shape", list(A.shape))
print(A)
# tensor([[[ 1,  2], [ 7,  8]],
#         [[ 3,  4], [ 9, 10]],
#         [[ 5,  6], [11, 12]]])

# Route B: reshape directly (DIFFERENT result!)
B = X.reshape(3, 2, 2)
print("\nreshape(3,2,2) -> shape", list(B.shape))
print(B)
# tensor([[[ 1,  2], [ 3,  4]],
#         [[ 5,  6], [ 7,  8]],
#         [[ 9, 10], [11, 12]]])

print("\nAre they equal?", torch.equal(A, B))  # False
```

Same shape, different data — proof that `reshape` and `transpose` are fundamentally different operations.

---

## 7. Conclusion

The `view` / `reshape` + `transpose` pattern in multi-head attention is not redundant — each operation serves a distinct purpose:

- **`view` / `reshape`** splits a flat dimension into structured sub-dimensions (e.g., splitting `d_model` into `num_heads × d_k`).
- **`transpose`** reorders those dimensions so that each attention head gets a contiguous slice of the sequence to process.

Skipping `transpose` and relying on `reshape` alone will give you the right shape but the **wrong data**, silently corrupting your model's computations. Understanding this distinction is essential for anyone implementing or debugging attention mechanisms.
