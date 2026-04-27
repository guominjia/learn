# Understanding Conv2d: How Multi-Channel Convolution Really Works

> A deep dive into PyTorch's `nn.Conv2d` — parameter semantics, multi-channel mechanics, and why convolution outputs look like grayscale images.

---

## Conv2d Parameters Explained

`nn.Conv2d` is the workhorse of convolutional neural networks. Its constructor signature:

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
```

| Parameter | Meaning |
|-----------|---------|
| `in_channels` | Number of channels in the input feature map (e.g., 3 for an RGB image) |
| `out_channels` | Number of channels in the output — equivalently, the number of independent convolutional filters |
| `kernel_size` | Spatial size of each filter (e.g., `3` means 3×3) |
| `stride` | Step size of the sliding window; controls the spatial resolution of the output |
| `padding` | Number of zero-padded pixels added to each edge of the input; commonly used to preserve spatial dimensions |
| `bias` | Whether to add a learnable scalar bias per output channel (default: `True`) |

---

## Walkthrough: 3-Channel Input → 6-Channel Output with a 3×3 Kernel

Consider a concrete example:

```python
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
```

Input shape: `(batch, 3, 32, 32)` — a batch of 32×32 RGB images.

### Shape of the Weight Tensor

Each filter must span **all input channels**, so a single filter has shape:

$$\text{filter shape} = (\text{in\_channels},\ k_H,\ k_W) = (3, 3, 3)$$

Since we want 6 output channels, there are **6 such filters**. The full weight tensor is:

```
conv.weight.shape = (6, 3, 3, 3)
                     ↑  ↑  ↑  ↑
              out_ch  in_ch  kH kW
```

### How a Single Filter Produces One Output Channel

A single filter (shape `3 × 3 × 3`) slides across the input and, at every spatial position, performs:

$$\text{output}(x, y) = \sum_{c=0}^{2}\sum_{i=0}^{2}\sum_{j=0}^{2} W_c(i,j) \cdot \text{input}_c(x+i,\ y+j) \;+\; b$$

Visually:

```
Input (3, 32, 32)
  ├─ Channel 0 (32×32)  ×  filter[0] (3×3)  →  partial sum₀
  ├─ Channel 1 (32×32)  ×  filter[1] (3×3)  →  partial sum₁
  └─ Channel 2 (32×32)  ×  filter[2] (3×3)  →  partial sum₂
                                                    │
                                        element-wise sum + bias
                                                    ↓
                                         1 output feature map (30×30)
```

The three per-channel partial results are **summed element-wise** into a single 2D map. That is why one filter, regardless of how many input channels there are, always produces exactly **one** output channel.

### Six Filters → Six Output Channels

Each of the 6 filters independently repeats the process above, yielding 6 feature maps that are stacked along the channel dimension:

```
Input: (batch, 3, 32, 32)
         │
         │  6 filters, each (3, 3, 3)
         ▼
Conv2d(3, 6, 3, stride=1, padding=0)
         │
         ▼
Output: (batch, 6, 30, 30)
```

### Output Spatial Size Formula

$$H_{out} = \frac{H_{in} - k + 2p}{s} + 1$$

With our parameters ($H_{in}=32,\ k=3,\ p=0,\ s=1$):

$$H_{out} = \frac{32 - 3 + 0}{1} + 1 = 30$$

Setting `padding=1` would keep the spatial size unchanged:

$$H_{out} = \frac{32 - 3 + 2}{1} + 1 = 32$$

### Parameter Count

$$\text{params} = \underbrace{6 \times 3 \times 3 \times 3}_{\text{weights} = 162} + \underbrace{6}_{\text{biases}} = 168$$

---

## Why Does a Convolution Output Look Like a Grayscale Image?

After convolution, each output channel is a single-channel 2D map — it is **inherently grayscale**. This often surprises people who expect to see color after convolving an RGB image.

### Root Cause

A convolution filter **sums across all input channels** at every spatial location. The R, G, and B information is fused into one scalar value per position. The result no longer separates color — it encodes a learned **feature response**.

### Do 6 Output Channels Equal Two RGB Images?

No. The 6 channels are **not** two groups of (R, G, B). Each channel is an independent feature detector trained to respond to different patterns:

```
Input  (3 channels = R, G, B)         →  human-interpretable color
Output (6 channels = f₀, f₁, …, f₅)  →  abstract features with no color semantics
```

For example, after training, individual channels may respond to horizontal edges, vertical edges, color gradients, textures, and so on.

### How to Visualize Convolution Outputs

| Method | Description |
|--------|-------------|
| **Per-channel grayscale** | Display each of the 6 channels as a separate grayscale image (most common) |
| **Pseudo-color heatmap** | Apply a colormap like `jet` or `viridis` to a single channel to show activation intensity |
| **Pick 3 channels as RGB** | Map any 3 channels to R/G/B for a false-color composite (colors have no physical meaning) |

```python
import matplotlib.pyplot as plt

# output shape: (6, 30, 30)
fig, axes = plt.subplots(1, 6, figsize=(18, 3))
for i in range(6):
    axes[i].imshow(output[i].detach().numpy(), cmap='gray')
    axes[i].set_title(f'Channel {i}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **One filter spans all input channels** — its depth always equals `in_channels`.
2. **`out_channels` = number of filters** — each produces one output feature map.
3. **Cross-channel summation** is why the output is single-valued (grayscale) per channel; color information is encoded in weights, not preserved visually.
4. **`padding = kernel_size // 2`** (with `stride=1`) is the common recipe to keep spatial dimensions unchanged.
5. Visualize feature maps **one channel at a time** — stacking them into RGB is misleading unless done intentionally for analysis.