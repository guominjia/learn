# ResNet: The Architecture That Won ImageNet

*April 2026*

---

## 1. ImageNet and the Deep Learning Revolution

The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) was the annual benchmark that drove the modern deep learning revolution in computer vision. From AlexNet (2012) to VGGNet (2014), each year's winner went deeper — more layers, more parameters, better accuracy. But going deeper hit a wall.

By 2015, researchers observed a counterintuitive phenomenon: simply stacking more layers on a plain convolutional network didn't always help. In fact, **deeper plain networks performed worse** than their shallower counterparts — not just on test data, but on training data too. This is the **degradation problem**, and it is *not* caused by overfitting.

---

## 2. The Degradation Problem

Consider two networks: a shallow one with 20 layers and a deep one with 56 layers. You would expect the deeper network to be at least as good — after all, the extra layers could just learn identity mappings and reproduce the shallow network's behavior. In practice, though, the 56-layer plain network had *higher* training error than the 20-layer one.

This showed that current optimizers (SGD and its variants) struggle to find good solutions in very deep plain networks. The optimization landscape becomes too difficult to navigate as depth increases.

---

## 3. The Residual Learning Framework

He et al. (2015) proposed a simple but powerful fix. Instead of asking each stack of layers to learn the desired mapping $H(x)$ directly, let them learn the **residual**:

$$F(x) = H(x) - x$$

The output then becomes:

$$y = F(x) + x$$

This is implemented via a **shortcut (skip) connection** that bypasses one or more layers and adds the input directly to the output. No extra parameters, no extra computation — just an identity shortcut and an element-wise addition.

### Why This Works

1. **Easier optimization target.** If the optimal transformation is close to the identity, it is much easier for the network to push $F(x)$ toward zero than to learn a full identity mapping from scratch.
2. **Smooth gradient flow.** During backpropagation, the gradient can flow directly through the skip connection. This mitigates the vanishing gradient problem and helps even very deep networks train effectively.
3. **No-harm depth.** In the worst case, extra layers can "do nothing" (i.e., $F(x) \approx 0$), so adding layers never degrades performance — it can only help.

---

## 4. Building Block Design

ResNet uses two types of residual blocks:

### Basic Block (ResNet-18/34)

Two stacked $3 \times 3$ convolutions with batch normalization and ReLU:

```
x ──► Conv 3×3 ──► BN ──► ReLU ──► Conv 3×3 ──► BN ──► (+) ──► ReLU
 │                                                        ▲
 └────────────────── identity shortcut ──────────────────┘
```

### Bottleneck Block (ResNet-50/101/152)

A $1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ design that first reduces the channel dimension, applies the spatial convolution, and then restores it:

```
x ──► Conv 1×1 ──► BN ──► ReLU ──► Conv 3×3 ──► BN ──► ReLU ──► Conv 1×1 ──► BN ──► (+) ──► ReLU
 │                                                                                      ▲
 └───────────────────────────── identity shortcut ─────────────────────────────────────┘
```

The bottleneck design keeps computational cost manageable while allowing the network to go much deeper.

### Dimension Mismatch

When the spatial size or channel count changes between input and output (e.g., at stage boundaries), a $1 \times 1$ convolution with appropriate stride is used as the shortcut to match dimensions.

---

## 5. Overall Architecture (ResNet-50)

| Stage | Output Size | Layers |
|-------|------------|--------|
| Input | 224 × 224 | $7 \times 7$ conv, stride 2 |
| Pool | 112 × 112 | $3 \times 3$ max pool, stride 2 |
| Stage 1 | 56 × 56 | 3 × bottleneck (64-d) |
| Stage 2 | 28 × 28 | 4 × bottleneck (128-d) |
| Stage 3 | 14 × 14 | 6 × bottleneck (256-d) |
| Stage 4 | 7 × 7 | 3 × bottleneck (512-d) |
| Output | 1 × 1 | Global avg pool → 1000-d FC → softmax |

Total: 50 weighted layers. By changing the number of blocks per stage, you get ResNet-18, 34, 101, or 152.

---

## 6. ImageNet Results (ILSVRC 2015)

ResNet-152 achieved a **top-5 error rate of 3.57%** on the ImageNet test set with model ensembling — surpassing human-level performance (~5.1%) and winning 1st place in ILSVRC 2015 classification.

| Model | Depth | Top-5 Error |
|-------|-------|-------------|
| VGGNet (2014) | 19 | 7.3% |
| GoogLeNet (2014) | 22 | 6.7% |
| **ResNet (2015)** | **152** | **3.57%** |

ResNet also won the detection and localization tracks that year, demonstrating that the residual framework is broadly useful beyond classification.

---

## 7. Minimal PyTorch Residual Block

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # shortcut projection when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity          # <-- the residual connection
        return self.relu(out)
```

---

## 8. Legacy and Impact

ResNet's influence goes far beyond ImageNet:

- **Backbone standard.** ResNet-50/101 became the default feature extractor for object detection (Faster R-CNN), semantic segmentation (DeepLab), and many other vision tasks.
- **Inspired modern architectures.** DenseNet, ResNeXt, EfficientNet, and even Transformers (which use residual connections in every block) all build on the skip-connection idea.
- **Enabled extreme depth.** Follow-up work pushed to 1000+ layers with proper initialization and training strategies.
- **Cross-domain adoption.** Residual connections are now standard in NLP (Transformer), speech, reinforcement learning, and more.

The core insight — *make it easy for the network to learn the identity, so extra depth never hurts* — remains one of the most important ideas in deep learning.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Russakovsky, O. et al. (2015). *ImageNet Large Scale Visual Recognition Challenge.* IJCV.
