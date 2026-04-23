# The End of `T.Scale`: What Happened to `torchvision.transforms.Scale` and What to Use Instead

If you've ever upgraded a legacy PyTorch project and suddenly hit an `AttributeError: module 'torchvision.transforms' has no attribute 'Scale'`, you're not alone. This post explains exactly when `T.Scale` was removed, why it was removed, and how to migrate your code cleanly.

---

## What Was `T.Scale`?

`torchvision.transforms.Scale` was an early image-resizing transform in the `torchvision` library. It accepted either an integer (resize the shorter side to that value while preserving aspect ratio) or a tuple `(H, W)` (resize to an exact size). In short, it did exactly what `Resize` does today.

```python
import torchvision.transforms as T

# Old way (deprecated, then removed)
transform = T.Scale(256)
transform = T.Scale((224, 224))
```

---

## When Was It Removed?

| Milestone | Version | Notes |
|-----------|---------|-------|
| Deprecated | <= 0.8 | `Scale` marked as deprecated; `Resize` introduced as the replacement |
| Still present | 0.11 | Available but with deprecation warning |
| **Removed** | **0.12** | `Scale` fully deleted from the public API |

`torchvision 0.12` (released alongside PyTorch 1.11 in March 2022) performed a broad API cleanup, removing all transforms that had been deprecated since version 0.8 or earlier. `Scale` was one of those long-overdue removals.

---

## Why Was It Removed?

Two reasons drove the decision:

1. **It was a redundant alias.** `Scale` and `Resize` were functionally identical. Maintaining two names for the same operation added confusion and documentation overhead with no benefit.

2. **Scheduled deprecation cleanup.** Starting from 0.12, the torchvision team enforced a formal deprecation policy: APIs deprecated in 0.8 or earlier were removed. `Scale` fell squarely into this category.

---

## What to Use Instead

### Class-based transform (recommended for `transforms.Compose`)

```python
import torchvision.transforms as T

# Resize shorter side to 256, preserving aspect ratio
transform = T.Resize(256)

# Resize to an exact (H, W)
transform = T.Resize((224, 224))
```

### Functional API

```python
import torchvision.transforms.functional as F
from PIL import Image

img = Image.open("image.jpg")
img_resized = F.resize(img, 256)          # shorter side -> 256
img_resized = F.resize(img, (224, 224))   # exact size
```

### Quick migration reference

| Old code | New code |
|----------|----------|
| `T.Scale(256)` | `T.Resize(256)` |
| `T.Scale((224, 224))` | `T.Resize((224, 224))` |
| `transforms.Scale(n)` | `transforms.Resize(n)` |

---

## Checking for Other Removed APIs

If you are upgrading from an old codebase, it is worth auditing for other deprecated transforms removed in 0.12. A quick grep can help:

```bash
grep -rn "T\.Scale\|transforms\.Scale\|T\.RandomSizedCrop\|T\.CenterCrop" your_project/
```

`RandomSizedCrop` is another common one that was removed in the same cleanup wave - its replacement is `RandomResizedCrop`.

---

## Summary

- `torchvision.transforms.Scale` was **removed in torchvision 0.12** (March 2022).
- It was deprecated long before that (<= 0.8) because `Resize` is the canonical, better-named equivalent.
- The fix is a straight one-to-one replacement: **`T.Scale` -> `T.Resize`**.

Keeping dependencies up to date and replacing deprecated APIs early is always cheaper than debugging a broken pipeline after a major upgrade.