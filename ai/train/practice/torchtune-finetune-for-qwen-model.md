# Torchtune finetune for Qwen model

```bash
pip install torch torchao torchao<0.8

# List available finetune model
tune ls

# Download Qwen 2.5 3B model
tune download Qwen/Qwen2.5-3B-Instruct --output-dir /tmp/Qwen2.5-3B-Instruct --hf-token $HF_TOKEN

# Finetune model
tune run lora_finetune_single_device --config qwen2_5/3B_lora_single_device epochs=1 dtype=fp32

# Download Qwen 2.5 7B model if have enough memory
tune download Qwen/Qwen2.5-7B-Instruct --output-dir /tmp/Qwen2.5-7B-Instruct --hf-token $HF_TOKEN

# Finetune model
tune run lora_finetune_single_device --config qwen2_5/7B_lora_single_device epochs=1 dtype=fp32

# Author deny my access so can't try this one
tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token $HF_TOKEN
```

If only have 32GB memory, it's difficult to finetune **7B** model according to below calculation:
- 3B fp32 memory occupation is $(3 \times 1,000,000,000 \times 4) \div (1024^3) = 11.18GB$
- 7B fp32 memory occupation is $(7 \times 1,000,000,000 \times 4) \div (1024^3) = 26.07GB$

## References

- <https://meta-pytorch.org/torchtune/stable/tutorials/first_finetune_tutorial.html>