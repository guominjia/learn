---
layout: post
title: "How to Analyze a Model from safetensors: Decoder Layers, Attention, and FFN"
date: 2026-06-16
tags: [safetensors, transformers, pytorch, llm, model-analysis]
---

When you first open a `.safetensors` file, it is tempting to think you can directly inspect the full model architecture from it.

You can inspect **weights and tensor shapes** very well, but there is an important detail:

- `safetensors` stores tensors (parameter values), not the full forward graph.
- Architectural behavior (for example, exact Self-Attention logic or Cross-Attention placement) is defined by model code + config.

This post shows two practical ways to analyze a model:

1. **Reliable method**: load via Hugging Face `transformers` and inspect modules.
2. **Fallback method**: inspect raw `.safetensors` keys and infer structure from naming patterns.

---

## 1) What `safetensors` gives you (and what it does not)

`safetensors` is great for:

- fast and safe tensor loading,
- listing parameter names,
- checking tensor shapes and dtypes,
- counting parameters.

It does **not** directly tell you:

- exact execution order,
- custom attention logic,
- activation implementation details,
- whether a block is decoder-only vs encoder-decoder (unless inferred from names/config).

So if your goal is: *How many DecoderLayer blocks? Is there Cross-Attention? What is FFN style?* then use config + model class whenever possible.

---

## 2) Recommended workflow: `transformers` + `safetensors`

Install dependencies:

```bash
pip install safetensors transformers torch
```

Then run:

```python
import re
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM

MODEL_DIR = r"/path/to/model_dir"  # contains config.json + *.safetensors

cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
print("model_type:", cfg.model_type)
print("is_encoder_decoder:", getattr(cfg, "is_encoder_decoder", False))
print("num_hidden_layers:", getattr(cfg, "num_hidden_layers", None))
print("num_attention_heads:", getattr(cfg, "num_attention_heads", None))
print("intermediate_size:", getattr(cfg, "intermediate_size", None))

if getattr(cfg, "is_encoder_decoder", False):
	model = AutoModelForSeq2SeqLM.from_pretrained(
		MODEL_DIR,
		trust_remote_code=True,
		torch_dtype="auto",
		device_map="cpu",
	)
else:
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_DIR,
		trust_remote_code=True,
		torch_dtype="auto",
		device_map="cpu",
	)

# Find decoder-layer-like modules (generic heuristic)
decoder_layers = []
for name, module in model.named_modules():
	cls = module.__class__.__name__.lower()
	if (
		"decoderlayer" in cls
		or re.search(r"(decoder\.layers\.\d+$|model\.layers\.\d+$|h\.\d+$)", name)
	):
		decoder_layers.append((name, module))

print("DecoderLayer count:", len(decoder_layers))

if decoder_layers:
	layer_name, layer = decoder_layers[0]
	print("Sample layer:", layer_name, layer.__class__.__name__)

	self_attn = []
	cross_attn = []
	ffn_like = []

	for sub_name, _ in layer.named_modules():
		if re.search(r"(^|\.)(self_attn|self_attention)(\.|$)", sub_name, re.I):
			self_attn.append(sub_name)
		if re.search(r"(^|\.)(cross_attn|encoder_attn|encdecattention)(\.|$)", sub_name, re.I):
			cross_attn.append(sub_name)
		if re.search(r"(mlp|ffn|feed_forward|wi|wo|gate_proj|up_proj|down_proj)", sub_name, re.I):
			ffn_like.append(sub_name)

	print("SelfAttention modules:", self_attn or "Not found")
	print("CrossAttention modules:", cross_attn or "Not found")
	print("FFN-like modules:", ffn_like or "Not found")

	print("\nParameters in sample decoder layer:")
	for pn, p in layer.named_parameters():
		print(f"{pn:50s} {tuple(p.shape)}")
```

### Why this method is best

Because it combines:

- **config-level truth** (declared hyperparameters), and
- **class-level truth** (actual module composition).

This is the closest thing to a reliable architectural introspection script.

---

## 3) Fallback workflow: only `.safetensors`

If you only have a raw `.safetensors` file and no config/model code, you can still infer useful structure.

```python
import re
from safetensors import safe_open

path = r"/path/to/model.safetensors"

keys = []
with safe_open(path, framework="pt", device="cpu") as f:
	for k in f.keys():
		keys.append(k)

# Infer layer ids from naming patterns
layer_ids = set()
for k in keys:
	m = re.search(r"(?:layers|h|block)\.(\d+)\.", k)
	if m:
		layer_ids.add(int(m.group(1)))

print("Inferred layer count:", len(layer_ids))

self_attn_keys = [
	k for k in keys if re.search(r"(^|\.)(self_attn|self_attention)(\.|$)", k)
]
cross_attn_keys = [
	k for k in keys if re.search(r"(^|\.)(cross_attn|encoder_attn|encdecattention)(\.|$)", k)
]
ffn_keys = [k for k in keys if re.search(r"(mlp|ffn|feed_forward|gate_proj|up_proj|down_proj|wi|wo)", k)]

print("self-attn key count:", len(self_attn_keys))
print("cross-attn key count:", len(cross_attn_keys))
print("ffn key count:", len(ffn_keys))

print("\nExample self-attn keys:")
for k in self_attn_keys[:20]:
	print(k)
```

### Caveat

This method is **pattern-based inference**. It can be very useful, but it is still less authoritative than loading the real model class.

---

## 4) Decoder-only vs Encoder-Decoder: quick interpretation

- **Decoder-only models** (many modern LLMs): usually have Self-Attention + FFN per layer, and no Cross-Attention.
- **Encoder-Decoder models**:
	- **Encoder layers** usually have Self-Attention + FFN, and no Cross-Attention.
	- **Decoder layers** usually have Self-Attention + Cross-Attention + FFN.
- This is the standard Transformer pattern; some specialized architectures may add extra attention modules.

So if your key/module scan shows no cross-attention names, that is often expected for decoder-only LLMs.

---

## 5) Practical checklist for model analysis

When auditing an unknown model package, I usually do this in order:

1. Read `config.json` (layer count, head count, hidden size, FFN/intermediate size).
2. Load model class with `transformers` (if available).
3. Print one representative layer and all parameter shapes.
4. Use safetensors key scan to cross-check naming and parameter coverage.
5. Compute parameter counts by module groups if needed.

This gives both high-level architecture and low-level tensor confidence.

---

## Conclusion

`safetensors` is excellent for secure, efficient tensor inspection, but architecture understanding is strongest when combined with `transformers` config and model classes.

If your goal is to answer questions like:

- How many DecoderLayer blocks are there?
- Does the model have Self-Attention and Cross-Attention?
- What does the FFN block look like?

then the most robust approach is:

**`config.json` + model class introspection + safetensors key/shape validation.**

