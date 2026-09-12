---
title: "Fixing CUDA Out-of-Memory Errors in Ovis2.5-2B Inference"
date: 2026-09-02
tags: [cuda, pytorch, transformers, ovis, multimodal]
---

# Fixing CUDA Out-of-Memory Errors in Ovis2.5-2B Inference

I tried to run the Ovis2.5-2B vision-language model on a GPU with 16 GiB of
memory. The model loaded, but inference failed while processing the image:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 576.00 MiB.
GPU 0 has a total capacity of 15.57 GiB of which 313.56 MiB is free
```

This post records what I ran, how I located the allocation, and the smallest
changes that made inference fit in memory. The same investigation also explains
why the script reported a slow image processor even though it passed
`use_fast=True` to the model loader.

## What I Did

The task was simple: give Ovis an image and ask it to calculate the sum of the
numbers in the middle box in figure (c).

The original script followed the model card's Transformers example, with
FlashAttention enabled and a remote model implementation:

```python
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

torch.cuda.empty_cache()

MODEL_PATH = "AIDC-AI/Ovis2.5-2B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    _attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    use_fast=True,
).cuda()

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": Image.open(requests.get(
                "https://cdn-uploads.hf-mirror.com/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png",
                stream=True,
            ).raw),
        },
        {
            "type": "text",
            "text": "Calculate the sum of the numbers in the middle box in figure (c).",
        },
    ],
}]

with torch.inference_mode():
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    outputs = model.generate(
        inputs=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        enable_thinking=False,
        enable_thinking_budget=True,
        max_new_tokens=3072,
        thinking_budget=2048,
    )

response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## The Problem

The failure was not caused by the text-generation budget. Thinking was disabled,
so `thinking_budget=2048` was inactive. The allocation happened earlier, in the
visual path, before the model could generate an answer.

The important part of the error was the requested allocation:

```text
Tried to allocate 576.00 MiB
```

That size was a useful clue. The model weights already occupied most of the
available VRAM, leaving too little room for a temporary tensor created while the
image was converted into visual tokens.

## Finding the Cause

### 1. Model memory is only the starting point

For a rough lower bound, two billion parameters stored as BF16 weights require:

$$
2 \times 10^9 \times 2\ \text{bytes} \approx 4\ \text{GB}
$$

That estimate excludes the vision encoder, the visual tokenizer, runtime
buffers, and the generation cache. It is therefore not a useful prediction of
the peak memory required by a multimodal model.

I also checked the actual parameter dtype instead of assuming that the loader
had honored the requested type:

```python
print(next(model.parameters()).dtype)
```

For Transformers versions that expect the documented argument name, the model
loading call should use `torch_dtype=torch.bfloat16` rather than `dtype=`.

### 2. The visual tokenizer creates a large temporary tensor

Ovis2.5's remote model code defines a visual vocabulary of 65,536 entries. Its
visual tokenizer projects each visual feature to almost the full vocabulary and
then applies softmax in float32:

```python
visual_vocab_size = 65536
head_dim = visual_vocab_size - len(INDICATOR_IDS)

logits = self.head(features)
tokens = torch.softmax(logits, dim=-1, dtype=torch.float32)
```

The default preprocessing limit in that implementation is:

```python
max_pixels = 1344 * 1792
```

In my run, the resized image produced roughly 2,300 visual tokens. The
float32 softmax tensor was therefore approximately:

$$
2300 \times 65532 \times 4\ \text{bytes}
\approx 576\ \text{MiB}
$$

This matches the allocation in the error message. The failure was in the
visual tokenizer's `logits -> softmax` path, not in the later text-generation
loop.

### 3. The custom `generate()` includes the visual forward pass

Ovis overrides `generate()`. Before delegating to the underlying language model,
it calls `merge_multimodal()`, which runs the visual tokenizer and merges the
visual embeddings with the text embeddings.

The script already used `torch.inference_mode()`, so autograd graph recording
was not the main cause of this particular failure. Keeping that context is still
important: PyTorch documents it as the inference-only mode that disables
autograd-related overhead such as view tracking and version-counter updates.

`torch.cuda.empty_cache()` also cannot make a live tensor smaller. It only helps
with cached blocks that are no longer referenced; it cannot remove the temporary
softmax allocation or the model weights.

## The Fix

I reduced the image resolution before it entered the visual tokenizer, used the
documented dtype argument, disabled evaluation-time features explicitly, and
reduced the output limit while debugging:

```python
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_PATH = "AIDC-AI/Ovis2.5-2B"
MIN_PIXELS = 448 * 448
MAX_PIXELS = 768 * 768

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    _attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda()
model.eval()

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": Image.open(requests.get(
                "https://cdn-uploads.hf-mirror.com/production/uploads/658a8a837959448ef5500ce5/TIlymOb86R6_Mez3bpmcB.png",
                stream=True,
            ).raw),
        },
        {
            "type": "text",
            "text": "Calculate the sum of the numbers in the middle box in figure (c).",
        },
    ],
}]

with torch.inference_mode():
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    outputs = model.generate(
        inputs=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        enable_thinking=False,
        max_new_tokens=512,
    )

response = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

The most important change is `max_pixels=768 * 768`. Fewer image pixels produce
fewer visual tokens, which reduces the size of the visual-tokenizer logits and
its float32 softmax temporary.

The smaller `max_new_tokens` value controls a different memory consumer: the
language model's generation cache. It is useful for keeping the total memory
budget predictable, but it was not the allocation that caused this failure.

If the reduced image size is not accurate enough, the next options are to move
some modules to CPU with `device_map="auto"`, use CPU offload, or load a
quantized model. Those options trade memory for latency or precision; reducing
the visual input size is the least invasive fix for this error.

## Why `use_fast=True` Did Not Remove the Warning

The script also printed:

```text
Using a slow image processor as `use_fast` is unset
```

There are two different loaders involved:

```python
# Outer model loader
AutoModelForCausalLM.from_pretrained(..., use_fast=True)
```

and, inside Ovis's remote model code:

```python
self.image_processor = AutoImageProcessor.from_pretrained(
    image_processor_name_or_path,
    do_center_crop=False,
)
```

The outer `use_fast=True` is not automatically forwarded to the inner
`AutoImageProcessor`. The warning describes that second call, not the model
loader in the application script.

This warning is independent of the CUDA OOM. A slow processor may make image
preprocessing slower, but it does not explain the 576 MiB GPU allocation made by
the visual tokenizer.

The practical choices are:

1. Ignore the warning if the model works and preprocessing speed is acceptable.
2. Pin a Transformers version compatible with the Ovis release.
3. If modifying the cached remote code is part of a controlled experiment, pass
   `use_fast=False` to make the intended slow processor explicit. Do not force a
   fast processor solely to silence the warning unless that processor is
   available and supported by the model.

## Conclusion

The useful debugging sequence was:

1. Read the allocation size in the CUDA error.
2. Trace the custom Ovis `generate()` into `merge_multimodal()`.
3. Inspect the visual tokenizer's vocabulary size, softmax dtype, and image
   resolution limit.
4. Reduce `max_pixels` before trying heavier solutions such as offload or
   quantization.

The filename, the `use_fast` warning, and the thinking budget were distractions.
The peak allocation came from high-resolution visual tokenization, so controlling
the image resolution fixed the problem at the point where the memory was being
used.

## References

- [Ovis2.5-2B model card](https://huggingface.co/ATH-MaaS/Ovis2.5-2B) - official inference example, model identifier, and image-input parameters.
- [Ovis2.5 remote model implementation](https://huggingface.co/ATH-MaaS/Ovis2.5-2B/raw/main/modeling_ovis2_5.py) - visual tokenizer, default pixel limits, softmax path, and custom multimodal `generate()` implementation.
- [Ovis2.5 configuration](https://huggingface.co/ATH-MaaS/Ovis2.5-2B/raw/main/configuration_ovis2_5.py) - visual vocabulary size configuration.
- [PyTorch `inference_mode` documentation](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html) - inference-mode behavior and its relationship to autograd.
- [Transformers `AutoImageProcessor` implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/image_processing_auto.py) - `use_fast` handling and processor backend selection.