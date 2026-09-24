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
)
```

With `stride=128` and `return_overflowing_tokens=True`, each chunk repeats the last 128 tokens of the previous one:

```text
chunk 1: token 0    ~ 1023
chunk 2: token 896  ~ 1919
chunk 3: token 1792 ~ 2199
```

Note that `stride` on its own does nothing — it only takes effect when `return_overflowing_tokens=True`. Avoid `return_tensors="pt"` here as well: the last chunk is shorter than `max_length`, so the chunks cannot be stacked into a single tensor unless you also pass `padding="max_length"`.

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

Two things are called "batching", and they are easy to confuse:

| | `map(batched=True)` | `DataLoader(batch_size=N)` |
|---|---|---|
| When | Preprocessing, once | Training/inference, every step |
| What it does | Passes N examples to your function as a dict of lists | Groups N stored rows into tensors via `collate_fn` |
| Result | Still stored as **individual rows** | A batch tensor |

So `batched=True` does not produce batched data. It only changes the calling convention of your function, and it is the one mode where **the output row count is allowed to differ from the input row count**. Without it, `map()` expects exactly one example in and one example out, so the extra chunks from `return_overflowing_tokens=True` have nowhere to go.

After `map()`, the dataset is simply longer. The DataLoader then batches those chunk rows like any other rows, with no idea they came from overflow.

One consequence: once you shuffle, chunks from the same document are scattered across different DataLoader batches. That is fine for classification or language-model training, but if you need to reassemble per-document predictions (question answering, long-document inference), keep `overflow_to_sample_mapping` as a column so you can group the chunks again afterwards.

Putting tokenization and `map()` together:

```python
cols = raw_dataset.column_names

def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024,
        stride=128,
        return_overflowing_tokens=True,
    )
    idx = out.pop("overflow_to_sample_mapping")
    out["label"] = [batch["label"][i] for i in idx]
    return out

tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=cols)
```

`remove_columns` is required here. The original columns still hold one value per document, so once chunking makes the new columns longer, Arrow refuses to build the table and raises `Column 'text' expected length 1000 but got 1473`. Dropping the original columns lets the tokenized output define the new row count.