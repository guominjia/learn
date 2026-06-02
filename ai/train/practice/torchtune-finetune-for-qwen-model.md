# Fine-Tuning LLMs with Torchtune: A Practical Guide (Qwen & Llama2)

This post walks through fine-tuning large language models using [torchtune](https://github.com/pytorch/torchtune), covering common errors and memory constraints encountered in practice — especially when running inside a memory-limited Kubernetes container.

## Environment Setup

```bash
pip install torch torchtune torchao --index-url https://download.pytorch.org/whl/cu128
```

### Version Compatibility: torchtune + torchao

After installation, you may hit this import error when running any `tune` command:

```
ImportError: cannot import name 'int4_weight_only' from 'torchao.quantization'
```

This happens because `torchtune` and `torchao` have a **version mismatch**. Newer versions of `torchao` renamed `int4_weight_only` to `Int4WeightOnlyConfig`. Fix it by upgrading both packages together, or pinning a compatible `torchao` version:

```bash
# Option 1: Upgrade both
pip install torchtune torchao --upgrade

# Option 2: Pin compatible torchao
pip install "torchao<0.8"
```

## Downloading Models

```bash
# List all available built-in configs
tune ls

# Download Qwen 2.5 models
tune download Qwen/Qwen2.5-3B-Instruct --output-dir /tmp/Qwen2.5-3B-Instruct --hf-token $HF_TOKEN
tune download Qwen/Qwen2.5-7B-Instruct --output-dir /tmp/Qwen2.5-7B-Instruct --hf-token $HF_TOKEN

# Download Llama 2 7B
tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token $HF_TOKEN
```

If `tune download` fails (e.g., due to the `torchao` import issue above), you can bypass it entirely with `huggingface-cli`:

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /tmp/Llama-2-7b-hf --token $HF_TOKEN
```

## Fine-Tuning with LoRA

### Basic Command

```bash
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device epochs=1
```

### Error: bf16 Not Supported

```
RuntimeError: bf16 precision was requested but not available on this hardware.
```

`bf16` (bfloat16) requires NVIDIA Ampere or newer GPUs (A100, RTX 30xx/40xx). If your hardware doesn't support it, switch to `fp32` or `fp16`:

```bash
# Use fp32 (safe for all hardware, but 2x memory and slower)
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device epochs=1 dtype=fp32

# Use fp16 (most GPUs support this, half the memory of fp32)
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device epochs=1 dtype=fp16
```

## Handling OOM (Out of Memory) Kills

### Symptom

Training starts and immediately gets killed:

```
0%|  | 0/3235 [00:00<?, ?it/s]Killed
```

No error traceback — the Linux OOM Killer terminates the process silently.

### Identifying the Real Memory Limit in Containers

If you're running inside a **Kubernetes Pod / Docker container**, the `top` or `free` commands show the **host machine's** memory, not your container's limit. The actual limit is controlled by **cgroups**:

```bash
# Container memory limit (cgroup v2)
cat /sys/fs/cgroup/memory.max

# Current usage
cat /sys/fs/cgroup/memory.current

# For cgroup v1
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

In my case, the container limit was **32GB** while `top` showed 257GB (host memory):

| Metric | Value |
|--------|-------|
| Container memory limit | 34,359,738,368 bytes (**32 GiB**) |
| Usage before kill | 34,337,239,040 bytes (**~32 GiB**) |
| Headroom | **~21 MB** — almost zero |

### Memory Estimation for Model Weights

Model weights alone (fp32, 4 bytes per parameter):

- **3B model**: $(3 \times 10^9 \times 4) \div 1024^3 \approx 11.2\text{ GB}$
- **7B model**: $(7 \times 10^9 \times 4) \div 1024^3 \approx 26.1\text{ GB}$

On top of weights, you also need memory for optimizer states, gradients, activations, and data loading — easily adding 50–100% overhead. A 7B fp32 model realistically needs **40+ GB**.

### Solutions to Reduce Memory Usage

**1. Use fp16 with smaller batch size:**

```bash
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device \
  epochs=1 dtype=fp16 batch_size=1 gradient_accumulation_steps=16 \
  enable_activation_checkpointing=true tokenizer.max_seq_len=512
```

fp16 cuts model weight memory in half (~14 GB for 7B), leaving room for optimizer states and activations within 32 GB.

**2. Use QLoRA (4-bit quantization) for maximum savings:**

```bash
# If a built-in qlora config exists:
tune run lora_finetune_single_device --config qwen2_5/7B_qlora_single_device epochs=1

# Or manually specify the quantizer:
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device \
  epochs=1 dtype=fp32 \
  quantizer._component_=torchtune.training.quantization.Int4WeightOnlyQuantizer \
  quantizer.groupsize=128
```

4-bit quantization reduces the 7B model to **~3.5 GB**, making it very comfortable within a 32 GB container.

**3. Use a smaller model:**

```bash
tune download Qwen/Qwen2.5-3B-Instruct --output-dir /tmp/Qwen2.5-3B-Instruct --hf-token $HF_TOKEN
tune run lora_finetune_single_device --config qwen2_5/3B_lora_single_device epochs=1 dtype=fp32
```

Qwen2.5 available sizes: **0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B** (there is no 4B variant).

## Summary: Memory Strategy Cheat Sheet

| Strategy | 7B Weight Memory | Total Estimated | Fits 32GB? |
|----------|-----------------|-----------------|------------|
| fp32 | ~26 GB | ~40+ GB | No |
| fp16 | ~13 GB | ~20-24 GB | Yes |
| QLoRA (4-bit) | ~3.5 GB | ~8-12 GB | Yes |
| Use 3B fp32 | ~11 GB | ~18-22 GB | Yes |

## References

- [Torchtune First Fine-Tune Tutorial](https://meta-pytorch.org/torchtune/stable/tutorials/first_finetune_tutorial.html)
- [Torchtune GitHub](https://github.com/pytorch/torchtune)
- [Qwen2.5 Model Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)