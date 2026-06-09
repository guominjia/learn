---
layout: post
title: "Upgrading CUDA Toolkit Without Touching the Driver: A Practical Guide"
date: 2026-06-09
categories: [gpu, linux]
tags: [cuda, nvidia, ubuntu, python, torch]
---

## Background

When working with modern AI libraries like `flash-linear-attention` or `causal-conv1d`, you may encounter this warning or error:

```
[transformers] The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

Or worse, a hard crash:

```
RuntimeError: causal_conv1d is only supported on CUDA 11.6 and above.
```

Both point to the same root cause: your installed CUDA Toolkit is too old.

---

## Diagnosing the Problem

There are two different CUDA version concepts that often cause confusion:

| Command | What it shows |
|---------|---------------|
| `nvidia-smi` | Maximum CUDA version the **driver** supports |
| `nvcc -V` | The **CUDA Toolkit** version actually installed |

For example:

```
# nvidia-smi output
CUDA Version: 12.8   ← driver supports up to 12.8

# nvcc -V output
Cuda compilation tools, release 11.5   ← toolkit is only 11.5
```

The driver and the toolkit are **independent components**. Having a driver that supports CUDA 12.8 does not mean your toolkit is at 12.8. Libraries like `causal-conv1d` compile against the toolkit version (`nvcc`), not the driver version.

---

## Driver vs Toolkit: What Is the Difference?

| Component | Role | Installed by |
|-----------|------|--------------|
| **NVIDIA Driver** | Communicates with GPU hardware; required for `nvidia-smi` | System/admin, rarely changed |
| **CUDA Toolkit** | Compiler toolchain (`nvcc`) and runtime libraries for building CUDA programs | Can be installed independently, multiple versions can coexist |

> Only `sudo apt install cuda` (without a version suffix) installs both. `cuda-toolkit-12-8` installs the toolkit only and does **not** touch your driver.

---

## Upgrading the CUDA Toolkit on Ubuntu

Ubuntu's default apt repository only ships an old version (e.g., 11.5). To get newer versions, you need to add NVIDIA's own apt repository first.

### Step 1: Add NVIDIA's Repository

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

Replace `ubuntu2204` with `ubuntu2004` if you are on Ubuntu 20.04.

### Step 2: Install the Toolkit

```bash
sudo apt install -y cuda-toolkit-12-8
```

No restart required — the toolkit is userspace tooling, unlike a driver which loads a kernel module.

### Step 3: Fix the `nvcc` Path

After installation, running `nvcc -V` may still show the old version. This is because `/usr/bin/nvcc` (installed by the Ubuntu package `nvidia-cuda-toolkit`) is a real file, not a symlink, and takes precedence over the newly installed toolkit.

The cleanest solution is to prepend the new toolkit's `bin` directory to `PATH`:

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
which nvcc          # should be /usr/local/cuda-12.8/bin/nvcc
nvcc -V             # should show release 12.8
```

Alternatively, remove the old Ubuntu package entirely:

```bash
sudo apt remove nvidia-cuda-toolkit
```

### Why PATH and Not `update-alternatives`?

`update-alternatives` manages symlinks registered in its database. Because `/usr/bin/nvcc` is a plain file installed by the apt package — not registered with `update-alternatives` — the tool has no effect here. Controlling `PATH` order is the correct and non-destructive approach.

---

## Multiple CUDA Versions Coexist by Design

NVIDIA installs each toolkit version into its own directory:

```
/usr/local/cuda-11.5/
/usr/local/cuda-12.8/
/usr/local/cuda -> /usr/local/cuda-12.8  (symlink)
```

The system does not automatically switch the active version because users may intentionally run different projects against different toolkits. You control which version is active through `PATH`.

If you prefer a managed approach:

```bash
sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 128
sudo update-alternatives --config cuda   # interactive selector
```

This only works if `/usr/local/cuda` is a symlink, which it is when installed via NVIDIA's repo.

---

## Installing PyTorch and CUDA Extensions

Once the toolkit is at 12.8, install PyTorch built against the matching CUDA version:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
```

Then install the extensions:

```bash
uv pip install causal-conv1d flash-linear-attention
```

Verify everything is aligned:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc -V
```

Both should report CUDA 12.x. If they match, `causal-conv1d` will compile and load correctly, and the `transformers` fast path warning will be gone.

---

## Summary

- `nvidia-smi` CUDA version = driver ceiling, not your toolkit version.
- Upgrade the toolkit independently with `cuda-toolkit-12-8` from NVIDIA's apt repo.
- No restart needed for a toolkit-only install.
- Fix `nvcc` path by prepending `/usr/local/cuda-12.8/bin` to `PATH` in `~/.bashrc`.
- Match your `torch` CUDA build (`cu128`) with the installed toolkit.
