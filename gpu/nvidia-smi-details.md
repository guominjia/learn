# Understanding `nvidia-smi` Output

`nvidia-smi` (NVIDIA System Management Interface) is the go-to command-line tool for monitoring NVIDIA GPU status. This post breaks down every field in its output so you know exactly what your GPU is doing.

## Sample Output

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               Off |   00000000:3B:00.0 Off |                  Off |
| 30%   28C    P8             19W /  300W |   37211MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4120      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A           19658      C   ...SenseNova-U1/.venv/bin/python      37194MiB |
+-----------------------------------------------------------------------------------------+
```

## Header Line

| Field | Example | Description |
|-------|---------|-------------|
| NVIDIA-SMI | 570.211.01 | Version of the `nvidia-smi` tool itself |
| Driver Version | 570.211.01 | Installed NVIDIA kernel driver version |
| CUDA Version | 12.8 | Maximum CUDA version supported by this driver (not the installed CUDA toolkit version) |

## GPU Status Table

### Row 1: Identity & Error Correction

| Field | Example | Description |
|-------|---------|-------------|
| GPU | 0 | GPU index (starting from 0) |
| Name | NVIDIA RTX A6000 | GPU model name |
| Persistence-M | Off | **Persistence Mode.** When `On`, the driver stays loaded between GPU calls, reducing initialization latency. `Off` means the driver is unloaded when no process is using the GPU. Enable with `nvidia-smi -pm 1` for server workloads. |
| Bus-Id | 00000000:3B:00.0 | PCI bus address (`domain:bus:device.function`) |
| Disp.A | Off | Whether a display is connected to this GPU |
| Volatile Uncorr. ECC | Off | ECC (Error Correcting Code) memory error reporting. `Off` = disabled or not supported. When enabled, tracks uncorrectable memory errors since last driver load. |

### Row 2: Thermals, Power & Utilization

| Field | Example | Description |
|-------|---------|-------------|
| Fan | 30% | Fan speed as a percentage of maximum RPM |
| **Temp** | **28C** | **GPU core temperature in Celsius. Typical ranges: idle 25-40°C, load 65-85°C, throttle ~90°C. 28°C indicates the GPU is essentially idle.** |
| **Perf** | **P8** | **Performance State (P-State).** NVIDIA GPUs have states from **P0** (maximum performance, highest clocks) to **P12** (minimum performance, lowest clocks). P8 is a low-power idle state. The GPU dynamically shifts between P-States based on workload demand. Common states: P0 = full load, P2 = moderate, P5/P8 = idle, P12 = minimum. |
| Pwr:Usage/Cap | 19W / 300W | Current power draw vs. power limit. 19W out of 300W confirms idle state. |
| Memory-Usage | 37211MiB / 49140MiB | Used VRAM / Total VRAM. Here ~36GB of ~48GB is allocated (likely a large model loaded in memory). |
| GPU-Util | 0% | Percentage of time the GPU's compute cores were active. 0% means no kernels are running despite memory being occupied. |
| Compute M. | Default | Compute mode. `Default` = multiple processes can share the GPU. Other modes: `Exclusive_Thread`, `Exclusive_Process`, `Prohibited`. |
| MIG M. | N/A | Multi-Instance GPU mode. `N/A` means the GPU does not support MIG (only A100, H100, and newer data-center GPUs do). |

### Interpreting Temp, Perf, and Power Together

These three fields tell a consistent story:

| Scenario | Temp | Perf | Power |
|----------|------|------|-------|
| Idle | 25-35°C | P8 / P12 | 15-25W |
| Light work | 40-55°C | P5 / P2 | 50-150W |
| Full load | 65-85°C | P0 | 200-300W |
| Thermal throttle | >88°C | P0 (forced down) | Fluctuating |

In our example (28°C / P8 / 19W), the GPU is clearly idle — even though 36GB of VRAM is occupied by a loaded model, no inference or training computation is happening.

## Process Table

| Field | Example | Description |
|-------|---------|-------------|
| GPU | 0 | Which GPU the process is running on |
| **GI ID** | **N/A** | **GPU Instance ID.** Only meaningful when **MIG (Multi-Instance GPU)** is enabled. MIG partitions a single physical GPU into multiple isolated GPU instances, each with its own memory and compute resources. `N/A` means MIG is not active. |
| **CI ID** | **N/A** | **Compute Instance ID.** Within a MIG GPU Instance, you can further create Compute Instances. Each CI gets a slice of the GPU Instance's compute cores. `N/A` means MIG is not active. |
| PID | 19658 | Operating system process ID |
| Type | C / G | **C** = Compute process (CUDA workload), **G** = Graphics process (display/rendering) |
| Process name | ...SenseNova-U1/.venv/bin/python | Path to the executable |
| GPU Memory Usage | 37194MiB | VRAM consumed by this process |

### MIG Hierarchy (When Enabled)

On supported GPUs (A100, H100, etc.), MIG creates a two-level hierarchy:

```
Physical GPU
├── GPU Instance 0  (GI ID = 0)  ← own memory partition
│   ├── Compute Instance 0  (CI ID = 0)
│   └── Compute Instance 1  (CI ID = 1)
├── GPU Instance 1  (GI ID = 1)
│   └── Compute Instance 0  (CI ID = 0)
└── ...
```

Each GPU Instance (GI) gets isolated memory and compute slices. Within a GI, Compute Instances (CI) further divide the compute engines. This allows true multi-tenancy — different users or workloads run on the same physical GPU with hardware-level isolation.

## Quick Takeaway from the Example

- A Python process (`SenseNova-U1`) has loaded a large model consuming ~36GB VRAM
- Despite the memory allocation, GPU compute utilization is 0% — the model is loaded but not actively running inference
- The GPU is in idle power state (P8, 28°C, 19W) confirming no active computation
- MIG is not available on the RTX A6000, so GI/CI IDs show `N/A`