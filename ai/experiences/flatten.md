# Understanding Tensor Shape Changes: Conv2d, BatchNorm2d, and Flatten in PyTorch

When building convolutional neural networks in PyTorch, one of the most common sources of confusion is tracking how tensor shapes transform as data flows through each layer. This post walks through the shape arithmetic for `Conv2d`, `BatchNorm2d`, and `Flatten` — the bridge between convolutional feature extraction and fully connected classification.

## Conv2d Output Shape

Given an input tensor of shape $[N, 3, 224, 224]$ (a batch of RGB images at 224×224 resolution), consider the following convolution layer:

```python
nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
```

The output spatial dimensions are computed using:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1
$$

where:
- $H_{in} = 224$ — input height
- $P = 1$ — padding
- $K = 3$ — kernel size
- $S = 1$ — stride

Substituting:

$$
H_{out} = \left\lfloor \frac{224 + 2 \times 1 - 3}{1} \right\rfloor + 1 = 224
$$

Width follows the same formula, so the output shape is **$[N, 64, 224, 224]$**. The spatial dimensions are preserved because `padding=1` with a 3×3 kernel and `stride=1` is the classic "same" convolution configuration.

## BatchNorm2d Preserves Shape

```python
nn.BatchNorm2d(64)
```

`BatchNorm2d` normalizes each channel independently across the batch, height, and width dimensions. It does **not** alter the tensor shape at all. The parameter `64` simply tells the layer how many channels to expect.

- Input: $[N, 64, 224, 224]$
- Output: $[N, 64, 224, 224]$

## The Flatten Bridge

After convolution and normalization, if you want to pass features into a `Linear` (fully connected) layer, you need to collapse the multi-dimensional feature maps into a single vector per sample. This is where `nn.Flatten()` comes in.

```python
nn.Flatten()  # default: start_dim=1, end_dim=-1
```

By default, `Flatten()` collapses all dimensions **except** the batch dimension:

- Input: $[N, C, H, W]$
- Output: $[N, C \times H \times W]$

For our running example:

- Input: $[N, 64, 224, 224]$
- Output: $[N, 64 \times 224 \times 224] = [N, 3{,}211{,}264]$

This means the subsequent linear layer must be:

```python
nn.Linear(64 * 224 * 224, num_classes)
```

### A Note on Practicality

A feature vector of length 3,211,264 is **extremely large**. In practice, networks insert pooling layers (e.g., `MaxPool2d`, `AdaptiveAvgPool2d`) between convolution blocks to progressively reduce spatial dimensions before flattening. For example, ResNet uses `AdaptiveAvgPool2d((1, 1))` to reduce any spatial size down to $1 \times 1$ before flattening, resulting in a feature vector of length equal to the channel count alone.

## Putting It All Together

Here is a minimal model illustrating the full pipeline:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # [N, 3, 224, 224] -> [N, 64, 224, 224]
    nn.BatchNorm2d(64),                                                 # [N, 64, 224, 224] -> [N, 64, 224, 224]
    nn.ReLU(),                                                          # shape unchanged
    nn.AdaptiveAvgPool2d((1, 1)),                                       # [N, 64, 224, 224] -> [N, 64, 1, 1]
    nn.Flatten(),                                                       # [N, 64, 1, 1]     -> [N, 64]
    nn.Linear(64, 10),                                                  # [N, 64]            -> [N, 10]
)
```

## Key Takeaways

| Layer | Effect on Shape |
|---|---|
| `Conv2d(in, out, k, s, p)` | Changes channels; spatial size governed by $\lfloor(H + 2P - K) / S\rfloor + 1$ |
| `BatchNorm2d(C)` | No shape change — normalizes per channel |
| `Flatten()` | Collapses $[N, C, H, W]$ into $[N, C \cdot H \cdot W]$ |
| `Linear(in, out)` | Maps $[N, \text{in}]$ to $[N, \text{out}]$ |

Understanding these shape transformations is fundamental to debugging dimension mismatch errors — the most frequent runtime error when prototyping new architectures. When in doubt, insert a quick `print(x.shape)` in your `forward()` method to trace the flow.