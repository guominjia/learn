# Installing PyTorch with the Correct CUDA Version

A common mistake when installing PyTorch is specifying the CUDA version incorrectly. This post clarifies the correct approach and walks through diagnosing a real-world CUDA driver mismatch error.

## The Wrong Way

You might instinctively try something like:

```bash
pip install torch==128
# or
pip install torch==cu128
```

Neither of these is correct. The `==` operator in pip specifies the **PyTorch package version** (e.g., `2.7.0`), not the CUDA toolkit version.

## The Right Way

CUDA version selection is done via `--index-url`, which points pip to a wheel repository built for a specific CUDA version:

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

To pin both the PyTorch version and the CUDA version:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

## A Real-World Error: CUDA Driver Too Old

After installing PyTorch without specifying `--index-url`, you may encounter:

```
python3.12/site-packages/torch/cuda/__init__.py:187: UserWarning:
CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080). Please update your GPU driver...
```

### What This Means

- `found version 12080` corresponds to **CUDA 12.8**.
- The default `pip install torch` pulled a PyTorch build compiled against a **newer** CUDA version (e.g., 12.9+).
- Your NVIDIA driver only supports up to CUDA 12.8, so the newer build fails to initialize.

### How to Fix It

**Option 1: Install PyTorch matching your driver (recommended)**

First, check your driver's CUDA version:

```bash
nvidia-smi
```

Look at the top-right corner for `CUDA Version: 12.8`. Then install the matching build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Option 2: Upgrade your NVIDIA driver**

Download the latest driver from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx). A newer driver will support higher CUDA versions, allowing the default PyTorch build to work.

## Key Takeaway

| What you want to specify | How to specify it |
|---|---|
| PyTorch version | `pip install torch==2.7.0` |
| CUDA version | `--index-url https://download.pytorch.org/whl/cu128` |
| Both | `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128` |

Always run `nvidia-smi` first to know which CUDA version your driver supports, then pick the matching `--index-url`.
