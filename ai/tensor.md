# Why `np.array` Breaks Autograd Backpropagation: A Subtle Python Reference Bug

## Introduction

When building a minimal autograd engine from scratch, you might encounter a puzzling bug: gradient values silently become incorrect — not because of wrong math, but because of how Python handles object references. This post walks through a custom `Tensor` class with basic backpropagation, and explains why passing a raw `np.array` as the initial gradient produces wrong results while wrapping it in a `Tensor` works perfectly.

## The Tensor Class

Here is a minimal autograd `Tensor` implementation that supports addition and backpropagation:

```python
import numpy as np

class Tensor:
    def __init__(self, data, creators=None, creation_op=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        if id is None:
            id = np.random.randint(0, 100000)
        self.id = id
        self.children = {}
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_grad_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot backpropagate more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if grad is None:
                grad = 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and \
               (grad_origin is None or self.all_grad_accounted_for()):
                if self.creation_op == "add":
                    self.creators[0].backward(grad, self)
                    self.creators[1].backward(grad, self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          creators=[self, other],
                          creation_op="add",
                          autograd=True)
        return Tensor(self.data + other.data)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()
```

## The Computation Graph

```python
a = Tensor([1, 2, 3], autograd=True)
b = Tensor([4, 5, 6], autograd=True)
c = a + b
d = b + c
```

This builds the following graph:

```
a ──┐
    ├──> c ──┐
b ──┤        ├──> d
    └────────┘
```

Node `b` contributes to both `c` and `d`, so it should receive gradients from two paths.

## Working Case: `d.backward(Tensor(np.array([1,1,1])))`

```python
d.backward(Tensor(np.array([1, 1, 1])))

print(a.grad.data == np.array([1, 1, 1]))  # [ True  True  True]
print(b.grad.data == np.array([2, 2, 2]))  # [ True  True  True]
print(c.grad.data == np.array([1, 1, 1]))  # [ True  True  True]
print(d.grad.data == np.array([1, 1, 1]))  # [ True  True  True]
```

All gradients are correct.

## Broken Case: `d.backward(np.array([1,1,1]))`

```python
d.backward(np.array([1, 1, 1]))

print(a.grad == np.array([1, 1, 1]))  # [False False False]
print(b.grad == np.array([2, 2, 2]))  # [ True  True  True]
print(c.grad == np.array([1, 1, 1]))  # [False False False]
print(d.grad == np.array([1, 1, 1]))  # [False False False]
```

`a.grad` and `c.grad` and `d.grad` are wrong — they are `[2, 2, 2]` instead of `[1, 1, 1]`.

No runtime error occurs. The code runs silently but produces incorrect gradient values.

## Root Cause: In-Place Mutation via Shared References

The bug is in the `backward` method:

```python
if self.grad is None:
    self.grad = grad    # (1) assigns the reference, not a copy
else:
    self.grad += grad   # (2) in-place mutation for np.array
```

### Trace through the broken case

Let's call the initial `np.array([1,1,1])` object **obj_A** and trace what happens:

| Step | Action | Result |
|------|--------|--------|
| 1 | `d.backward(obj_A)` | `d.grad = obj_A` (reference) |
| 2 | `d` propagates to `b`: `b.backward(obj_A, d)` | `b.grad = obj_A` (same reference) |
| 3 | `d` propagates to `c`: `c.backward(obj_A, d)` | `c.grad = obj_A` (same reference) |
| 4 | `c` propagates to `a`: `a.backward(obj_A, c)` | `a.grad = obj_A` (same reference) |
| 5 | `c` propagates to `b`: `b.backward(obj_A, c)` | `b.grad += obj_A` → **in-place modifies obj_A to `[2,2,2]`** |

After step 5, **every** variable pointing to `obj_A` — including `d.grad`, `c.grad`, and `a.grad` — now sees `[2,2,2]`.

`b.grad` happens to show `[2,2,2]` which is the correct expected value, so it appears correct. But `d.grad`, `c.grad` were supposed to remain `[1,1,1]`.

## Why `Tensor` Works

When `grad` is a `Tensor` object:

```python
self.grad += grad
```

The `Tensor` class does **not** define `__iadd__`. Python falls back to:

```python
self.grad = self.grad + grad      # calls Tensor.__add__
```

`__add__` creates a **new `Tensor` object** with fresh data. This means each node gets its own independent gradient object — no shared references, no silent mutation.

## The Fix

Ensure that gradients are never shared by reference. The simplest fix is to copy on assignment:

```python
def backward(self, grad=None, grad_origin=None):
    if self.autograd:
        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception("Cannot backpropagate more than once")
            else:
                self.children[grad_origin.id] -= 1

        if grad is None:
            grad = Tensor(np.ones_like(self.data))
        if self.grad is None:
            self.grad = Tensor(np.copy(grad.data) if isinstance(grad, Tensor) else np.copy(grad))
        else:
            self.grad += grad

        if self.creators is not None and \
           (grad_origin is None or self.all_grad_accounted_for()):
            if self.creation_op == "add":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)
```

By calling `np.copy()`, each node stores its own gradient array, and in-place `+=` can no longer corrupt other nodes.

## Key Takeaway

| `+=` behavior | `np.array` | `Tensor` (no `__iadd__`) |
|---------------|-----------|--------------------------|
| Operation | `__iadd__` → in-place mutation | Falls back to `__add__` → new object |
| Shared references? | Yes — all point to same array | No — each `+=` creates a new `Tensor` |
| Silent corruption? | **Yes** | No |

This is a classic Python gotcha: **mutable objects shared via reference assignment can be silently corrupted by in-place operations.** In autograd implementations, always ensure gradient tensors are independent copies to avoid this trap.
