# Fixing PyTorch dtype Mismatch: `Long` vs `Float` in Matrix Multiplication

**Published:** April 2, 2026

---

## The Error

If you've spent any time training neural networks with PyTorch, you've likely hit this wall:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float
```

It's one of those errors that feels cryptic at first but makes complete sense once you understand what's happening under the hood.

---

## Why Does This Happen?

PyTorch is strict about **dtype consistency** during matrix multiplication (`matmul`). When you write something like:

```python
data = torch.tensor([[1, 0], [0, 1], [1, 1]])
```

PyTorch infers the dtype from the Python literals you pass in. Since `1` and `0` are **integer** values, PyTorch creates a tensor with `dtype=torch.int64` — also known as `Long`.

Meanwhile, neural network layers like `nn.Linear` initialize their weights as `torch.float32` (i.e., `Float`) by default. When a forward pass tries to multiply your `Long` input tensor against `Float` weight matrices, PyTorch throws a dtype mismatch error.

---

## The Fix

The solution is straightforward: explicitly specify `dtype=torch.float` when creating your input tensors.

**Before (broken):**

```python
import torch
import torch.nn as nn

data   = torch.tensor([[1, 0], [0, 1], [1, 1]])
target = torch.tensor([[1], [0], [1]])
```

**After (fixed):**

```python
import torch
import torch.nn as nn

data   = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float)
target = torch.tensor([[1], [0], [1]],           dtype=torch.float)

model     = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn   = nn.MSELoss()

output = model(data)
loss   = loss_fn(output, target)
loss.backward()
optimizer.step()
```

---

## Alternative Approaches

There is more than one way to solve this, depending on your preference:

### 1. Use float literals directly

```python
data = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
```

Python treats `1.` as a `float`, so PyTorch infers `torch.float32`.

### 2. Cast after creation with `.float()`

```python
data = torch.tensor([[1, 0], [0, 1], [1, 1]]).float()
```

`.float()` is shorthand for `.to(torch.float32)` and is convenient when you don't control how the tensor was originally created.

### 3. Use `.to(dtype=...)`

```python
data = data.to(dtype=torch.float32)
```

This is the most explicit form and is often preferred in production code for clarity.

---

## Understanding PyTorch's Default dtype Inference

| Python literal | Inferred `torch.dtype` |
|---|---|
| `1`, `0` (int) | `torch.int64` (`Long`) |
| `1.0`, `0.5` (float) | `torch.float32` (`Float`) |
| `True`, `False` (bool) | `torch.bool` |
| `1+2j` (complex) | `torch.complex128` |

Neural network parameters (`nn.Linear`, `nn.Conv2d`, etc.) default to `torch.float32`. Keeping your input tensors in `float32` avoids the mismatch entirely.

---

## Checking dtypes at Runtime

When debugging dtype issues, two properties are your best friends:

```python
print(data.dtype)              # torch.int64  <- the culprit
print(model[0].weight.dtype)   # torch.float32
```

You can also build a small guard function to handle this defensively:

```python
def safe_forward(model, x):
    if x.dtype != torch.float32:
        x = x.float()
    return model(x)
```

---

## Key Takeaways

- PyTorch infers `int64` (`Long`) from integer Python literals, not `float32`.
- `nn.Linear` and most neural network modules expect `float32` inputs.
- Use `dtype=torch.float`, float literals (`1.`), or `.float()` to ensure correct types.
- Always verify `.dtype` when a matrix multiplication fails unexpectedly.

---

*Tags: PyTorch, Deep Learning, Debugging, dtype, RuntimeError*
