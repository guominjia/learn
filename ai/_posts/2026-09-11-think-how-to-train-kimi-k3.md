---
layout: post
title: "Thinking About Retraining Kimi K3"
date: 2026-09-11
categories: [ai]
tags: [kimi, moe, transformer, training, finetuning]
---

I am thinking about retraining Kimi K3, and the first thing I notice is its unusually large architecture.

| Item | Kimi K3 |
|---|---:|
| Transformer layers | **93** |
| Hidden size | **7168** |
| Attention heads | **96** |
| Total parameters | **2.8T** |
| Activated parameters per token | **104B** |
| Number of MoE experts | **896** |
| Experts selected per token | **16** |
| MoE hidden size per expert | **3072** |
| Dense layers | **1** |
| Context window | **1,048,576 tokens** |
| Vision encoder | 27 layers, 1024 dimensions, about 401M parameters |

The attention stack in its 93 layers is not traditional full attention:

- **69 KDA layers** (Kimi Delta Attention, a linear-attention approach)
- **24 Gated MLA layers** (gated multi-head latent attention)
- The FFN uses MoE: each token is routed to 16 of 896 experts, with 2 additional shared experts.

This leads me to distinguish between two very different things I might mean by "retraining Kimi K3":

1. **Continue training the original Kimi K3 weights (fine-tuning):** I need to keep the number of layers, hidden size, number of attention heads, and MoE expert structure in the text backbone compatible with the original model.
2. **Change the number of layers, hidden size, or number of experts and train the model myself:** this would no longer be fine-tuning K3 weights. I would be training a **new model in the K3 architectural style**. The original weight tensors would no longer have matching shapes and generally could not be loaded.

I find the text-backbone configuration under `text_config` in the top-level `config.json`, for example:

```json
{
  "text_config": {
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    "num_attention_heads": 96,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "moe_intermediate_size": 3072,
    "routed_expert_hidden_size": 3584
  }
}
```

The official weights have 93 layers, a hidden size of 7168, and 896 experts. If I change the model to 24 layers, a hidden size of 1024, 8 heads, and 8 experts, a parameter shape changes from something like:

$$
W_q \in \mathbb{R}^{7168 \times 7168}
$$

to:

$$
W_q \in \mathbb{R}^{1024 \times 1024}
$$

Therefore, `from_pretrained()` cannot load the original K3 weights into my modified architecture.

## My First Option: Keep K3 Unchanged and Fine-Tune with LoRA or SFT

If I want to continue using the Kimi K3 model itself, this is the practical route I can take. I first load the official custom model code through Hugging Face:

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "moonshotai/Kimi-K3"

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

I can then use `peft` for LoRA instead of attempting full-parameter training:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

However, even with quantized weights, K3 is still a **2.8T-parameter MoE model with 104B activated parameters per token**. I cannot realistically train it on an ordinary single machine or a small number of GPUs. LoRA training and inference would still require substantial GPU memory and distributed deployment capabilities.

## My Second Option: Modify the Configuration and Train from Scratch

Alternatively, I can use the custom implementation from the official repository to create a smaller K3-style network. The key point is that I need to use **`from_config()` rather than `from_pretrained()`**.

```python
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "moonshotai/Kimi-K3"

config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
)

text_config = config.text_config

# My smaller text backbone
text_config.num_hidden_layers = 24
text_config.hidden_size = 1024
text_config.num_attention_heads = 8
text_config.num_key_value_heads = 8

# My smaller MoE layers
text_config.num_experts = 8
text_config.num_experts_per_token = 2
text_config.num_shared_experts = 1
text_config.moe_intermediate_size = 512
text_config.routed_expert_hidden_size = 1024

# Reinitialize without loading the original K3 weights
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)

model.save_pretrained("./kimi-k3-small-init")
```

Whether this code runs directly depends on whether K3's remote model implementation accepts arbitrarily small configurations. K3 uses custom components such as KDA, AttnRes, and LatentMoE. Some implementations may assume fixed dimensions, head counts, or layer layouts. If I get a shape or assertion error, I will need to modify the corresponding constraints in the official model implementation rather than changing only the JSON configuration.

## References

- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) - official architecture summary, parameter counts, context length, and deployment information.
- [Kimi K3 source repository](https://github.com/MoonshotAI/Kimi-K3) - official model repository and technical report.
- [Transformers model documentation](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_config) - `from_config()` and `from_pretrained()` behavior, including checkpoint size compatibility.
