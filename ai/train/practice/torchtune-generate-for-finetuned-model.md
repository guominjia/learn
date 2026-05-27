# Running Inference on a Torchtune LoRA Fine-Tuned Model

After fine-tuning a model with torchtune's LoRA recipe, the next step is running inference to test the results. This turns out to be less straightforward than expected — this post documents the full journey from training output to working generation, including every error encountered along the way.

## Prerequisites

```bash
pip install torch torchtune torchao --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate --index-url https://download.pytorch.org/whl/cu128
```

> `accelerate` is required if you use `device_map="auto"` in HuggingFace Transformers. Without it, you'll get:
> ```
> ValueError: Using a `device_map`, `tp_plan`, ... requires `accelerate`.
> ```

## The Training Command

```bash
tune run lora_finetune_single_device --config qwen2_5/3B_lora_single_device epochs=1
```

This produces a resolved config like:

```yaml
checkpointer:
  _component_: torchtune.training.FullModelHFCheckpointer
  checkpoint_dir: /tmp/Qwen2.5-3B-Instruct
  checkpoint_files:
    - model-00001-of-00002.safetensors
    - model-00002-of-00002.safetensors
  model_type: QWEN2
  output_dir: /tmp/torchtune/qwen2_5_3B/lora_single_device

model:
  _component_: torchtune.models.qwen2_5.lora_qwen2_5_3b
  apply_lora_to_mlp: true
  lora_alpha: 16
  lora_attn_modules: [q_proj, v_proj, output_proj]
  lora_dropout: 0.0
  lora_rank: 8

tokenizer:
  _component_: torchtune.models.qwen2_5.qwen2_5_tokenizer
  path: /tmp/Qwen2.5-3B-Instruct/vocab.json
  merges_file: /tmp/Qwen2.5-3B-Instruct/merges.txt
```

## Understanding the Training Output

After training completes, the output directory contains:

```
/tmp/torchtune/qwen2_5_3B/lora_single_device/
├── epoch_0/
│   ├── adapter_config.json
│   ├── adapter_model.pt
│   ├── adapter_model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model-00001-of-00002.safetensors   # ← merged full model weights
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── logs/
└── torchtune_config.yaml
```

The key insight: `FullModelHFCheckpointer` **merges LoRA weights back into the base model** and saves a complete HuggingFace-compatible checkpoint. This means:

- The `model-*.safetensors` files contain the **full merged weights** (base + LoRA already combined)
- You **must** load them with the base model (`qwen2_5_3b`), not the LoRA model (`lora_qwen2_5_3b`)
- The `adapter_model.pt` / `adapter_model.safetensors` files are also saved for reference, but they are **standalone adapter exports** — they cannot be loaded on top of the merged safetensors (that would apply LoRA twice)

> **Important:** You might think you can use `lora_qwen2_5_3b` + `adapter_checkpoint` to load base weights and adapter separately. This does **not** work with the `epoch_0/` output because the safetensors are already merged. Using `lora_qwen2_5_3b` creates a model that expects LoRA keys (`layers.*.attn.q_proj.lora_a.weight`, etc.), but those keys don't exist in the merged checkpoint — resulting in `Missing key(s) in state_dict` errors.

## Approach 1: `tune run generate` (torchtune CLI)

### Pitfall 1: Config Name Does Not Exist

```bash
# ❌ Wrong — this config doesn't exist in torchtune
tune run generate --config qwen2_5/3B_generation
```

```
FileNotFoundError: No such file or directory: '/Projects/finetune-llm/qwen2_5/3B_generation'
```

**Why?** When torchtune can't find a built-in config by name, it treats the argument as a **local file path** and tries to open it with OmegaConf. Always verify available configs first:

```bash
tune ls | grep -i gen
```

### Pitfall 2: Using Training Config for Generation

```bash
# ❌ Wrong — training config lacks generation-specific keys
tune run generate --config qwen2_5/3B_lora_single_device prompt="Hello"
```

```
omegaconf.errors.ConfigAttributeError: Missing key quantizer
```

The `generate` recipe expects fields like `quantizer` that don't exist in the training config.

### Pitfall 3: Using LoRA Model Architecture with Merged Weights

You might try to use `lora_qwen2_5_3b` — either with or without specifying `adapter_checkpoint`:

```bash
# ❌ Attempt 1: lora model without adapter
model._component_=torchtune.models.qwen2_5.lora_qwen2_5_3b

# ❌ Attempt 2: lora model + adapter file from epoch_0/
model._component_=torchtune.models.qwen2_5.lora_qwen2_5_3b
checkpointer.adapter_checkpoint=adapter_model.pt
```

Both fail with:

```
RuntimeError: Missing key(s) in state_dict:
    "layers.0.attn.q_proj.lora_a.weight",
    "layers.0.attn.q_proj.lora_b.weight", ...
```

**Why?** `FullModelHFCheckpointer` already merged LoRA into the base weights. The safetensors in `epoch_0/` are ordinary model weights with no LoRA keys. Using `lora_qwen2_5_3b` creates extra LoRA parameters that have no corresponding entries in the checkpoint.

**Fix:** Always use the base model `qwen2_5_3b` with the merged checkpoint. The LoRA model architecture (`lora_qwen2_5_3b`) is **only for training**, not for inference with merged checkpoints.

### Pitfall 4: `output_dir` Same as `checkpoint_dir`

```
ValueError: The output directory cannot be the same as or a subdirectory
of the checkpoint directory.
```

Always set `output_dir` to a different path than `checkpoint_dir`.

### Pitfall 5: Prompt Format

```bash
# ❌ Wrong — prompt is a string, but recipe expects a dict
prompt="What are the benefits of LoRA fine-tuning?"
```

```
TypeError: string indices must be integers, not 'str'
```

The `generate` recipe expects `prompt.user` and `prompt.system` fields.

### Working Command

```bash
tune run generate \
  --config generation \
  model._component_=torchtune.models.qwen2_5.qwen2_5_3b \
  checkpointer._component_=torchtune.training.FullModelHFCheckpointer \
  checkpointer.checkpoint_dir=/tmp/torchtune/qwen2_5_3B/lora_single_device/epoch_0 \
  checkpointer.checkpoint_files="[model-00001-of-00002.safetensors,model-00002-of-00002.safetensors]" \
  checkpointer.model_type=QWEN2 \
  checkpointer.output_dir=/tmp/torchtune/qwen2_5_3B/lora_single_device/generate_output \
  tokenizer._component_=torchtune.models.qwen2_5.qwen2_5_tokenizer \
  tokenizer.path=/tmp/torchtune/qwen2_5_3B/lora_single_device/epoch_0/vocab.json \
  tokenizer.merges_file=/tmp/torchtune/qwen2_5_3B/lora_single_device/epoch_0/merges.txt \
  device=cuda \
  dtype=bf16 \
  prompt.user="What are the benefits of LoRA fine-tuning?" \
  prompt.system="You are a helpful assistant."
```

Key points:
- Use `qwen2_5_3b` (not `lora_qwen2_5_3b`) since weights are already merged
- Point `checkpoint_dir` to `epoch_0/` where the merged safetensors live
- Set `output_dir` to a **different** directory than `checkpoint_dir`
- Use `prompt.user` / `prompt.system` instead of a plain `prompt` string

## Approach 2: HuggingFace Transformers (Recommended)

Since the output is a standard HuggingFace checkpoint, you can skip the `tune run generate` complexity entirely:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/tmp/torchtune/qwen2_5_3B/lora_single_device/epoch_0"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the benefits of LoRA fine-tuning?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

This works because:
- The `epoch_0/` directory contains `config.json`, `tokenizer.json`, and merged safetensors — everything HuggingFace needs
- No need to manually specify model architecture, tokenizer paths, or checkpoint files
- `apply_chat_template` handles the Qwen chat format automatically

## Summary

| Approach | Pros | Cons |
|----------|------|------|
| `tune run generate` | Stays within torchtune ecosystem | Many config pitfalls, verbose CLI |
| HuggingFace Transformers | Simple, standard API, auto-detects everything | Requires `transformers` + `accelerate` |

The HuggingFace approach is recommended for quick inference testing — the merged checkpoint is already in HF format, so there's no reason to fight with torchtune's generation config.
