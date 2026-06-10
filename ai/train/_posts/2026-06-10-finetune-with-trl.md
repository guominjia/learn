---
title: "Fine-Tuning with TRL: SFTTrainer, OOM, and LoRA"
date: 2026-06-10
tags: llm trl sft finetuning huggingface oom lora qlora
---

This post covers three practical issues that come up when fine-tuning a chat model with TRL's `SFTTrainer`: a removed API, CUDA out-of-memory errors during full fine-tuning, and how LoRA/QLoRA solves both the memory and the update-scope problems.

The original experiment used the [SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), the TRL course notebook, and a small SmolLM2 model. It ran into three common problems:

- `setup_chat_format` is no longer available in newer TRL releases.
- A base model without a chat template is a poor default for supervised fine-tuning on conversational data.
- Full fine-tuning on a large model exhausts GPU memory during the first optimizer step.

## What went wrong

### `setup_chat_format` no longer exists in TRL 1.5.1

Older examples often show this pattern:

```python
from trl import setup_chat_format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
```

In newer TRL versions, that helper is gone. If you try to import it, you will get:

```text
ImportError: cannot import name 'setup_chat_format' from 'trl'
```

The usual fixes are:

- use an instruct/chat model that already has a chat template
- clone a template from another tokenizer
- patch the tokenizer with TRL's training chat template helper when needed

### Base models are not the easiest starting point

The model below is a base model:

```python
model_name = "HuggingFaceTB/SmolLM2-135M"
```

For chat fine-tuning, this is less convenient than an instruct variant because it may not ship with a usable chat template.

For a first pass, this is a better choice:

```python
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
```

That usually removes the need for manual chat-format setup.

### Let TRL manage device placement

With `SFTTrainer`, it is better to let the trainer handle placement instead of forcing:

```python
model.to(device)
```

That becomes especially important once you move to LoRA, quantization, or multi-GPU setups.

## A minimal working TRL example

If the goal is simply to get training running, this is a cleaner baseline:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

trainer.train()
```

## If you want evaluation

Before enabling evaluation, check whether the dataset actually contains a `test` split:

```python
print(dataset)
print(dataset.keys())
```

If `test` exists, you can enable evaluation like this:

```python
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)
```

## CUDA out of memory

When running full fine-tuning on a large model, you may hit this error during the very first optimizer step:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 MiB.
GPU 0 has a total capacity of 47.41 GiB of which 8.69 MiB is free.
```

The forward and backward passes may complete, but Adam needs to allocate two extra state tensors (`exp_avg` and `exp_avg_sq`) for every parameter on the first `step`. That second allocation is what pushes the GPU over the limit.

**Why it happens**

Full fine-tuning keeps four copies of every parameter in memory: the weights, the gradients, and two Adam moment tensors. Add the activations from a large model and a 48 GB GPU fills up quickly.

**How to fix it (in order of impact)**

1. Switch to LoRA or QLoRA - freeze the base weights and only train a small adapter.
2. Set `per_device_train_batch_size=1` and use `gradient_accumulation_steps` to recover the effective batch size.
3. Enable `gradient_checkpointing=True` and `bf16=True`.
4. Use `optim="paged_adamw_8bit"` (bitsandbytes) to page optimizer state off the GPU.
5. Lower `max_length` to 1024 or 512 as a first step.
6. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (helps but does not solve the root cause).

## LoRA and QLoRA

LoRA (Low-Rank Adaptation) is the standard way to fine-tune large models without running out of memory. Instead of updating all parameters, it injects a pair of small trainable matrices into selected layers. The base model is frozen, so the memory footprint shrinks dramatically.

```text
Original weight W (frozen)
W + A x B   <- only A and B are trained (rank r << hidden dim d)
```

QLoRA combines LoRA with 4-bit quantization of the frozen base weights. The base model loads in NF4 format, reducing weight memory by roughly 4x, while LoRA adapters remain in bfloat16.

**When to use each**

| Setup | GPU memory | When to use |
|---|---|---|
| Full fine-tune | Highest | Small models only, or many GPUs |
| LoRA | Medium | Model fits in FP16/BF16 on one GPU |
| QLoRA | Lowest | Large model on a single consumer or mid-range GPU |

**Minimal QLoRA + SFTTrainer setup for a single 48 GB GPU**

Refer [code](https://github.com/guominjia/learn/blob/code_study/finetune/finetune-qwen3p5.py) for implementation

If OOM persists after switching to QLoRA, lower `max_length` to 768, then reduce `r` from 16 to 8.

## Practical recommendation

- Pick an instruct model that already has a chat template.
- Use `SFTTrainer` directly without legacy helpers.
- Check dataset splits before enabling evaluation.
- Start with QLoRA on any model larger than a few billion parameters.
- Keep `max_length` at 1024 until training is stable, then increase.

## Related resources

- [SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [TRL SFTTrainer course](https://huggingface.co/learn/llm-course/chapter11/3)
- [TRL documentation](https://huggingface.co/docs/trl)
- [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter11/section3.ipynb)
- [GitHub Code](https://github.com/huggingface/notebooks/blob/main/course/en/chapter11/section3.ipynb)
- [QLoRA script reference](https://github.com/guominjia/learn/blob/code_study/finetune/finetune-qwen3p5.py)