# GPU

## Troubleshooting: "Failed to initialize NVML: Driver/library version mismatch"

### The Problem

When running `nvidia-smi`, you may encounter the following error:

```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.126
```

This means the **NVIDIA kernel driver** loaded in memory and the **user-space NVML library** installed on disk are running different versions. The two components must be in sync for `nvidia-smi` and any GPU workload to function.

### Why Does This Happen?

| Cause | Details |
|-------|---------|
| Driver updated without reboot | The new library is on disk, but the old kernel module is still loaded in memory. |
| `apt upgrade` silently updated the driver | Package managers can pull in a newer driver as a dependency without an explicit reboot prompt. |
| Manual `.run` installer conflicts with package manager | Two installation methods leave mismatched files on the system. |
| Kernel update broke DKMS rebuild | A new Linux kernel was installed but DKMS failed to recompile the NVIDIA kernel module for it. |

### Quick Diagnosis

```bash
# Check the kernel module version currently loaded
cat /proc/driver/nvidia/version

# Check the user-space library version on disk
ls -la /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*
```

If the two versions differ, that confirms the mismatch.

### Fix 1 — Reboot (Simplest)

```bash
sudo reboot
```

A reboot forces the kernel to load the module that matches the on-disk library. **This resolves the issue in the vast majority of cases.**

### Fix 2 — Reload the Kernel Module Without Rebooting

Use this on production servers where a reboot is not immediately possible.

```bash
# 1. List processes using the GPU
sudo lsof /dev/nvidia*

# 2. Stop services that hold the GPU (examples)
sudo systemctl stop docker
sudo systemctl stop gdm          # display manager, if running a desktop

# 3. Unload the NVIDIA kernel modules in dependency order
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

# 4. Reload the matching module
sudo modprobe nvidia

# 5. Verify
nvidia-smi
```

> **Note:** If `rmmod` fails with "Module is in use", you still have a process holding the device. Kill or stop it first.

### Fix 3 — Reinstall the Driver

If the module and library are both corrupt or partially installed, a clean reinstall is the safest path.

```bash
# Remove existing packages
sudo apt-get purge nvidia-*

# If the driver was installed via a .run file instead:
# sudo /usr/bin/nvidia-uninstall

# Install the desired version
sudo apt-get update
sudo apt-get install nvidia-driver-580

# Reboot to load the new module
sudo reboot
```

### Fix 4 — Rebuild the DKMS Module

When a kernel update leaves the NVIDIA module uncompiled:

```bash
# Check DKMS status
dkms status

# Rebuild all modules for the running kernel
sudo dkms autoinstall

sudo reboot
```

### Prevention

Lock the driver package so routine upgrades do not silently change the version:

```bash
sudo apt-mark hold nvidia-driver-580
```

This keeps `apt upgrade` from touching the driver until you explicitly unhold it:

```bash
sudo apt-mark unhold nvidia-driver-580
```

### Key Takeaway

The error is almost always caused by an **updated library with a stale kernel module**. A simple `sudo reboot` fixes it in most cases. For environments that cannot reboot, manually unloading and reloading the kernel module (`rmmod` / `modprobe`) is the next best option.
