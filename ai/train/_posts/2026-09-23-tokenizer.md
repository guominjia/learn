---
layout: post
title: "Tokenizer"
date: 2026-09-23
tags: [transformer, huggingface, tokenizer]
---

A tokenizer splits text into tokens and maps them to ids. Load one with `AutoTokenizer`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Use `tokenizer.vocab_size` for the vocabulary size, and `tokenizer.get_vocab()` for the full `token -> id` map.

Convert between tokens and ids with `tokenizer.convert_tokens_to_ids()` and `tokenizer.convert_ids_to_tokens()`, or go straight from text to ids and back with `tokenizer.encode()` and `tokenizer.decode()`.

To avoid the warning **"Token indices sequence length is longer than the specified maximum sequence length for this model"**, truncate the input:
```python
inputs = tokenizer(
    text,
    truncation=True,
    max_length=1024,
    return_tensors="pt"
)
```

Plain truncation drops everything past the limit. To keep the whole text, split it into overlapping chunks instead:

```python
inputs = tokenizer(
    text,
    truncation=True,
    max_length=1024,
    stride=128,
    return_overflowing_tokens=True,
    return_tensors="pt"
)
```

With `stride=128` and `return_overflowing_tokens=True`, each chunk repeats the last 128 tokens of the previous one:

```text
chunk 1: token 0    ~ 1023
chunk 2: token 896  ~ 1919
chunk 3: token 1792 ~ 2199
```

Check the sequence limits with:

```python
print(tokenizer.model_max_length)
print(model.config.max_position_embeddings)
```

Add a special token like this:

```python
tokenizer.add_special_tokens({
    "pad_token": "[PAD]"
})
```

After adding tokens, use `len(tokenizer)` as the vocabulary size instead of `tokenizer.vocab_size`, because the latter does not count the added tokens. Remember to call `model.resize_token_embeddings(len(tokenizer))` so the embedding matrix matches.
