# PyTorch Gradient Pitfalls: Lessons from Building a Word2Vec Training Loop

> A practical debugging journey through common PyTorch gradient issues, from `NoneType` errors to vanishing leaf nodes.

---

## Background

While implementing a Word2Vec-style embedding model from scratch using raw PyTorch (without `nn.Module`), several non-obvious gradient-related bugs surfaced. This post documents each issue, explains the root cause, and provides the correct fix.

The training loop in question looked like this:

```python
for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):
        target_samples = [review[target_i]] + concatenated[(torch.rand(negative) * len(concatenated)).long().tolist()].tolist()
        left_context = review[max(0, target_i-windows):target_i]
        right_context = review[target_i+1:min(len(review),target_i+windows)]
        input_context = left_context + right_context
        embed_target = torch.mean(embed[input_context], dim=0)
        l2 = torch.sigmoid(embed_target.matmul(W1[target_samples].T))
        loss = (l2 - layer_2_target).pow(2).sum()
        loss.backward()
        with torch.no_grad():
            embed -= alpha * embed.grad
            W1 -= alpha * W1.grad
            embed.grad.zero_()
            W1.grad.zero_()
    if rev_i % 100 == 0:
        print(f"Iteration {rev_i} completed.")
        print("Similar to 'terrible':", similar("terrible"))
```

---

## Bug 1: `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`

### Error

```
embed -= alpha * embed.grad
TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'
```

### Root Cause

`embed.grad` is `None` because `embed` was created **without** `requires_grad=True`. Without this flag, PyTorch never tracks operations on the tensor, so no gradient is ever computed during `loss.backward()`.

### Fix

```python
# ❌ Before — no gradient tracking
embed = torch.randn(vocab_size, embed_size)
W1 = torch.randn(vocab_size, embed_size)

# ✅ After — gradients will be computed
embed = torch.randn(vocab_size, embed_size, requires_grad=True)
W1 = torch.randn(vocab_size, embed_size, requires_grad=True)
```

### Correct Update Pattern

```python
loss.backward()                          # must be OUTSIDE no_grad
with torch.no_grad():
    embed -= alpha * embed.grad          # safe in-place update
    W1   -= alpha * W1.grad
    embed.grad.zero_()                   # clear for next iteration
    W1.grad.zero_()
```

---

## Bug 2: Why Does `(torch.rand(..., requires_grad=True) - 0.5) * 0.2` Lose Its Gradient?

### The Surprising Behavior

```python
w1 = (torch.rand((5, 5), requires_grad=True) - 0.5) * 0.2
w2 = torch.rand((5, 5), requires_grad=True)

print(w1.requires_grad)  # True  — misleading!
print(w1.is_leaf)        # False ← the real problem
print(w2.is_leaf)        # True  ✅
```

Even though `w1.requires_grad` is `True`, **PyTorch only retains gradients for leaf tensors** by default. `w1` is the *result* of arithmetic operations, making it an intermediate node in the computation graph — not a leaf.

### Why Only Leaf Tensors?

PyTorch's autograd engine is designed to save memory. Intermediate activations in a deep network would consume enormous amounts of memory if gradients were retained for all of them. Only leaf tensors (parameters you actually want to update) retain `.grad` automatically.

You can force gradient retention on non-leaf tensors with `.retain_grad()`, but that is rarely what you want for weight initialization.

### Initialization Approaches Compared

| Approach | `is_leaf` | `.grad` available |
|---|---|---|
| `torch.rand((5,5), requires_grad=True)` | ✅ True | ✅ |
| `(torch.rand((5,5), requires_grad=True) - 0.5) * 0.2` | ❌ False | ❌ |
| `((torch.rand((5,5)) - 0.5) * 0.2).requires_grad_(True)` | ✅ True | ✅ |
| `torch.nn.Parameter((torch.rand((5,5)) - 0.5) * 0.2)` | ✅ True | ✅ |

### Correct Ways to Initialize with Scaling

```python
# Method 1: compute first, then attach gradient tracking
embed = ((torch.rand(vocab_size, embed_size) - 0.5) * 0.2)
embed.requires_grad_(True)   # now a leaf node ✅

# Method 2: use detach() to sever from computation graph
embed = ((torch.rand(vocab_size, embed_size) - 0.5) * 0.2).detach().requires_grad_(True)

# Method 3: wrap with nn.Parameter (preferred in nn.Module)
embed = torch.nn.Parameter((torch.rand(vocab_size, embed_size) - 0.5) * 0.2)
```

---

## Bug 3: Empty `input_context` Causes a Runtime Error

When `target_i` is at the very start or end of a review, both `left_context` and `right_context` can be empty lists, making `input_context = []`. Indexing an embedding matrix with an empty list crashes:

```python
embed_target = torch.mean(embed[input_context], dim=0)  # ❌ if input_context == []
```

### Fix

```python
input_context = left_context + right_context
if len(input_context) == 0:
    continue
```

---

## Bug 4: `layer_2_target` Dimension Must Match `target_samples`

```python
loss = (l2 - layer_2_target).pow(2).sum()
```

`l2` has shape `(1 + negative,)` — one positive sample plus `negative` noise samples. `layer_2_target` must be the same shape with the label `1` at index 0 and `0` elsewhere:

```python
layer_2_target = torch.zeros(1 + negative)
layer_2_target[0] = 1.0   # positive sample label
```

Getting this wrong produces a silent broadcasting error or a shape mismatch crash.

---

## Summary Checklist

```
✅ Tensors you want to differentiate must be created with requires_grad=True
✅ Apply scaling BEFORE setting requires_grad, or use requires_grad_() after
✅ Check is_leaf=True before assuming .grad will be populated
✅ Guard against empty context windows with an early continue
✅ Ensure layer_2_target shape matches the number of (positive + negative) samples
✅ Call loss.backward() outside torch.no_grad()
✅ Perform weight updates and grad.zero_() inside torch.no_grad()
```

---

## Key Takeaway

The single most important rule when working with raw PyTorch autograd:

> **Only leaf tensors retain gradients. A tensor produced by any arithmetic operation is not a leaf, even if `requires_grad=True`.**

When in doubt, print `tensor.is_leaf` alongside `tensor.requires_grad` — they tell very different stories.
