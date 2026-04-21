# Understanding PyTorch `transforms.Normalize` and `ToTensor`: Common Pitfalls

A walkthrough of three frequently misunderstood behaviors in `torchvision.transforms`.

---

## 1. Why Does `Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])` Fail While `Normalize(mean=[0.5], std=[0.5])` Works?

### The Rule

`transforms.Normalize` applies normalization **per channel** using the formula:

$$y_c = \frac{x_c - \text{mean}_c}{\text{std}_c}$$

The **length of `mean` and `std` must exactly match the number of channels `C`** in the input tensor.

### Reproducing the Error

```python
import numpy as np
from torchvision import transforms

a = transforms.ToTensor()(np.array([[1., 2], [3, 4]]))
print(a.shape)  # torch.Size([1, 2, 2])  ->  C=1

# Works: mean/std length == C
transforms.Normalize(mean=[0.5], std=[0.5])(a)

# Fails: mean/std length (2) != C (1)
transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(a)
```

The input is a 2-D grayscale array. After `ToTensor()` the shape becomes `[1, 2, 2]` — **1 channel**. Passing two values in `mean`/`std` tells PyTorch to expect 2 channels, causing a dimension mismatch error.

### Rule of Thumb

| Image type | Expected `mean`/`std` length |
|---|---|
| Grayscale | `[v]` — 1 value |
| RGB | `[v, v, v]` — 3 values |

Always check `tensor.shape` and match the number of values accordingly.

---

## 2. Why Is `mean=0.5, std=0.5` Hardcoded Instead of Being Computed from the Data?

### Two Different Meanings of "Normalization"

There is an important distinction between **statistical normalization** and the **fixed linear rescaling** that `transforms.Normalize` performs.

**Statistical normalization** (computed from data):

$$z = \frac{x - \mu}{\sigma}, \quad \mu = \text{mean}(X),\ \sigma = \text{std}(X)$$

**`transforms.Normalize(mean, std)`** (fixed constants you supply):

$$y = \frac{x - \text{mean}}{\text{std}}$$

The transform does **not** compute anything from your data. It applies exactly the constants you pass in.

### Why `0.5, 0.5` Is So Common

After `ToTensor()`, pixel values are typically in $[0, 1]$. Substituting `mean=0.5, std=0.5`:

$$\frac{x - 0.5}{0.5} = 2x - 1$$

This linearly maps $[0, 1] \to [-1, 1]$, a range that many training pipelines (especially GANs) prefer. It is a convenient choice, not a statistically derived one.

### Using Real Dataset Statistics

To perform true statistical normalization you must compute mean and std offline over the entire training set and supply those values. Well-known precomputed constants:

| Dataset | `mean` | `std` |
|---|---|---|
| MNIST | `[0.1307]` | `[0.3081]` |
| CIFAR-10 | `[0.4914, 0.4822, 0.4465]` | `[0.2470, 0.2435, 0.2616]` |
| ImageNet | `[0.485, 0.456, 0.406]` | `[0.229, 0.224, 0.225]` |

#### Computing mean/std from a DataLoader

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

loader = DataLoader(
    datasets.MNIST("data", train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=512, shuffle=False,
)

mean = torch.zeros(1)
var  = torch.zeros(1)
n    = 0

for images, _ in loader:
    # images: [B, C, H, W]
    b = images.size(0)
    images_flat = images.view(b, images.size(1), -1)  # [B, C, H*W]
    mean += images_flat.mean(2).sum(0)
    var  += images_flat.var(2).sum(0)
    n    += b

mean /= n
std   = torch.sqrt(var / n)
print(f"mean={mean.item():.4f}, std={std.item():.4f}")
# mean=0.1307, std=0.3081
```

---

## 3. Does `ToTensor()` Always Scale Values to `[0, 1]`?

### Short Answer

No. **The scaling behavior depends on the dtype of the input, not whether the variable "looks like" an image.**

### What `ToTensor()` Actually Does

```python
a = transforms.ToTensor()(np.array([[0., 0, 0, 0, 0, 6, 7, 8, 9, 10]] * 10))
print(a)
# tensor([[[ 0.,  0.,  0.,  0.,  0.,  6.,  7.,  8.,  9., 10.],
#          ...]])
# Values are still 0, 6, 7, 8, 9, 10 — NOT rescaled
```

Because `np.array([...])` defaults to `float64`, `ToTensor()` only:

- Reorders dimensions from `[H, W]` or `[H, W, C]` to `[C, H, W]`
- Converts dtype to `torch.float32`

It does **not** divide by 255.

### When Does `ToTensor()` Scale to `[0, 1]`?

The `/255` rescaling happens **only when the input dtype is `uint8`**, which is what PIL images produce.

```python
# Scaled to [0, 1]: uint8 input
arr_uint8 = np.array([[0, 128, 255]], dtype=np.uint8)
t = transforms.ToTensor()(arr_uint8)
print(t)  # tensor([[[0.0000, 0.5020, 1.0000]]])

# NOT scaled: float input
arr_float = np.array([[0., 128., 255.]])
t = transforms.ToTensor()(arr_float)
print(t)  # tensor([[[  0., 128., 255.]]])
```

### Summary

| Input dtype | `ToTensor()` behavior |
|---|---|
| `uint8` (PIL image, `np.uint8`) | Divides by 255, output in `[0, 1]` |
| `float32` / `float64` | Dimension reorder + type cast only, **no scaling** |

### Forcing Rescaling When You Need It

```python
# Option 1: cast to uint8 first
t = transforms.ToTensor()(arr.astype(np.uint8))

# Option 2: manually divide before converting
t = transforms.ToTensor()(arr.astype(np.float32) / 255.0)
```

---

## Key Takeaways

1. **`Normalize` is per-channel** — the length of `mean`/`std` must equal the number of channels `C`.
2. **`Normalize` does not compute statistics** — you supply fixed constants. `0.5` is just a convenient rescaling from `[0, 1]` to `[-1, 1]`, not a dataset statistic.
3. **`ToTensor` scales to `[0, 1]` only for `uint8` inputs** — float arrays are passed through without rescaling.