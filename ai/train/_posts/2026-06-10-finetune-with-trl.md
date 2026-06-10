---
title: Fine-Tuning with TRL: Common Pitfalls and a Minimal SFTTrainer Setup
date: 2026-06-10
tags: llm trl sft finetuning huggingface
---

This post summarizes a few practical issues that can appear when fine-tuning a chat model with TRL's `SFTTrainer`.

The original experiment used the [SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), the TRL course notebook, and a small SmolLM2 model. It ran into two common problems:

- `setup_chat_format` is no longer available in newer TRL releases.
- A base model without a chat template is a poor default for supervised fine-tuning on conversational data.

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

## Practical recommendation

If the model already has a chat template, do not force a legacy helper from old examples. Keep the setup simple:

- pick an instruct model first
- use `SFTTrainer` directly
- check dataset splits before enabling evaluation

That path is usually faster to debug and easier to maintain.

## Related resources

- [SmolTalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [TRL SFTTrainer course](https://huggingface.co/learn/llm-course/chapter11/3)
- [TRL documentation](https://huggingface.co/docs/trl)
- [Google Colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter11/section3.ipynb)
- [Github Code](https://github.com/huggingface/notebooks/blob/main/course/en/chapter11/section3.ipynb)