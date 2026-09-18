---
layout: post
title: "Training a Tokenizer with train_new_from_iterator"
date: 2026-09-17
categories: [ai]
tags: [tokenizer, transformers, bpe, vocabulary, qwen, deepseek]
---

## Training a New Tokenizer — "Whole Corpus" in Theory, "Sampled Subset" in Practice

BPE / WordPiece / Unigram are all **global statistics** algorithms:

- First scan the corpus to build a `word -> count` frequency table (after pre-tokenization)
- Then iteratively merge (BPE) or prune (Unigram) based on that global table

Memory usage therefore scales with the number of **unique words**, not the total byte size of the corpus — which is why tens of GB of text is still trainable.

```python
from transformers import AutoTokenizer

old = AutoTokenizer.from_pretrained("gpt2")

def corpus_iter(dataset, batch=1000):
    for i in range(0, len(dataset), batch):
        yield dataset[i : i + batch]["text"]

new_tok = old.train_new_from_iterator(corpus_iter(ds), vocab_size=32000)
```

The iterator feeds data in a **streaming** fashion, which looks like incremental construction, but the Rust backend accumulates every batch into one frequency table and only starts training after the last `yield`. There is no online/incremental vocabulary update.

## Why a Subset Is Usually Enough

| Consideration | Explanation |
|---|---|
| Fast convergence | Word frequency is long-tailed; a few hundred MB to a few GB of text already stabilizes a 32k vocab, and more data barely changes it |
| Time cost | Scanning TB-scale corpora just to train a tokenizer isn't worth it |
| Distribution coverage | The sample must cover all languages/domains/code/math symbols **proportionally**, or compression ratio degrades badly for some text types |

So the mainstream approach (LLaMA, GPT series, etc.) is **stratified sampling** of a few-GB representative subset from the full corpus, not the full corpus.

## Pitfalls

- **No incremental extension**: if the corpus changes and you want to update the vocab, you must retrain, then `resize_token_embeddings()` and retrain the model's embedding layer
- **Sampling bias is irreversible**: if the sample has no Chinese, Chinese gets shredded into UTF-8 byte-level fragments and sequence length explodes
- **Special tokens must be passed explicitly**: `train_new_from_iterator(..., new_special_tokens=[...])`

-------------

`old` here is just a **configuration template** — `train_new_from_iterator` serializes `old.backend_tokenizer` into `tokenizer.json`, **clears vocab/merges**, and retrains with the same pipeline config.

## What Gets Inherited

| Component | Explanation |
|---|---|
| `normalizer` | NFC/NFKC, lowercasing, accent stripping, etc. |
| `pre_tokenizer` | The critical one. GPT-2 uses `ByteLevel(add_prefix_space=False)`, BERT uses `BertPreTokenizer` — it decides how text is split into "words" |
| `model.type` | BPE / WordPiece / Unigram / WordLevel, which also selects `BpeTrainer` vs `WordPieceTrainer` |
| model config options | `continuing_subword_prefix` (`##`), `end_of_word_suffix` (`</w>`), `byte_fallback`, `dropout` |
| `decoder` | Rules for reassembling text |
| `post_processor` | Template is kept, but its token ids are **remapped to the new vocab** |
| special tokens | Taken from `old`'s `[unk]/[cls]/[sep]/[pad]/[mask]` etc. and passed as the trainer's `special_tokens` |
| Python class + `init_kwargs` | It returns `self.__class__(...)`, so `model_max_length`, `padding_side` and friends carry over |

## What Does Not Get Inherited

- **vocab** (`tokenizer_json["model"]["vocab"] = {}`)
- **merges** (BPE merge rules are cleared)
- **added_tokens**: `pop`ped first, and only special tokens are added back after training

So the old and new tokenizers have **completely unrelated vocabularies**; the same string will almost certainly encode to a different id sequence.

## Verify It Yourself

```python
import json
cfg = json.loads(old.backend_tokenizer.to_str())
print(cfg["normalizer"], cfg["pre_tokenizer"], cfg["decoder"], sep="\n")
print(cfg["model"]["type"], len(cfg["model"]["vocab"]))
```

Print it again after training and you'll see the structure is identical except for `vocab`/`merges`.

## What This Means in Practice

The only criterion for picking `old` is: **which splitting style do you want**.

- Training a code model → use `gpt2` or `Salesforce/codegen` as the template (ByteLevel BPE, preserves spaces/indentation)
- Training a Chinese BERT-style model → use `bert-base-chinese` (WordPiece + `##` prefix)
- Want `byte_fallback` with no UNK → use a `llama`-family template

**If `old`'s `pre_tokenizer` is wrong for your data (e.g. using a BERT template for code, where whitespace rules eat the indentation), no amount of corpus will save you — in that case build the pipeline from scratch with the `tokenizers` library instead of `train_new_from_iterator`.**

--------

Start by dumping its `tokenizer.json` to inspect the pipeline.

## How the Three Templates Actually Differ

From the official `tokenizer_config.json` files:

| | `gpt2` | `Qwen/Qwen3-8B` | `deepseek-ai/DeepSeek-V3` |
|---|---|---|---|
| `tokenizer_class` | `GPT2Tokenizer` | `Qwen2Tokenizer` | `LlamaTokenizerFast` |
| model | byte-level BPE | byte-level BPE | byte-level BPE |
| `unk_token` | none | `null` | `null` |
| vocab | 50257 | 151643 base + specials, padded to 151936 | 129280 |
| `model_max_length` | 1024 | 131072 | 131072 |
| bos | none | `add_bos_token: false` | `add_bos_token: true`, `<｜begin▁of▁sentence｜>` |
| pre_tokenizer | plain `ByteLevel` | `Split` (tiktoken-style regex) + `ByteLevel` | same kind of structure |

Note that even though DeepSeek's `tokenizer_class` says `LlamaTokenizerFast` and it still carries `legacy`/`sp_model_kwargs` fields, it actually runs the byte-level BPE from `tokenizer.json`, not SentencePiece.

## Technically It Works; the Gain Is in the pre_tokenizer

All three have `model.type == "BPE"`, squarely inside the branch `train_new_from_iterator` supports, so nothing will error out.

The real gain of using Qwen3/DeepSeek over `gpt2` as a template is inheriting that `Split` regex. GPT-2's `ByteLevel` regex is a product of the English-only era (digits aren't grouped, CJK falls back entirely to bytes), while Qwen/DeepSeek's regex groups digits into chunks of at most 3 and handles CJK and indentation explicitly. For Chinese/code corpora the difference is very visible.

## But There Are Traps

**Some added_tokens get lost.** `train_new_from_iterator` first does `pop("added_tokens")` and only adds special tokens back after training. Qwen3's `additional_special_tokens` lists only 13 entries (`<|im_start|>`, `<|im_end|>`, the vision batch), while `<tool_call>`, `</tool_call>`, `<think>`, `</think>`, `<|fim_prefix|>`, `<|repo_name|>` etc. are marked `"special": false` in `added_tokens_decoder` — **they will not come back automatically** and must be passed explicitly:

```python
new_tok = old.train_new_from_iterator(
    corpus_iter(ds), vocab_size=151936,
    new_special_tokens=["<think>", "</think>", "<tool_call>", "</tool_call>",
                        "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"],
)
```

**The chat_template carries over, but may become useless.** It lives in `init_kwargs` so it's inherited, but once the tokens above are missing from the new vocab, `<think>` gets split into a string of subwords and thinking mode simply breaks.

**DeepSeek's post_processor includes a bos.** With `add_bos_token: true` + `TemplateProcessing`, token ids get remapped on retraining, so you must make sure `<｜begin▁of▁sentence｜>` (note the full-width `｜` and the `▁`) is in the special tokens, or the mapping goes wrong.

**Don't casually shrink vocab_size.** Qwen3's 151936 exists to support multilingual text plus code. Training it down to 32k throws away its most valuable part — at which point you may as well use `gpt2` as the template and save yourself the trouble.

## The Bigger Question: Why Are You Retraining?

Picking `gpt2` as the template feels natural because the default assumption is **pretraining from scratch**, where the weights are discarded anyway.

But if you're looking at Qwen3 / DeepSeek, you most likely want to **reuse their pretrained weights**. Then `train_new_from_iterator` is the wrong tool — swap the vocab and every row of the embedding and lm_head no longer lines up, which means scrapping the whole model and retraining, at a cost starting in the tens of thousands of GPU hours.

What you actually want is to **extend the vocabulary**:

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tok.add_tokens(["your-domain-term", "..."])      # regular tokens
tok.add_special_tokens({"additional_special_tokens": ["<|my_tag|>"]})

model.resize_token_embeddings(len(tok))
# New rows are randomly initialized by default; initializing from the mean of existing
# token embeddings converges much faster
```

Qwen3 deliberately pads its vocab to 151936 (293 empty slots above the actual 151643) exactly for this kind of extension — adding a handful of tokens may not even require a resize.

In one line: **`train_new_from_iterator` is a tool for the "train a model from scratch" scenario; for models like Qwen3/DeepSeek whose weights you want to keep, extend the vocabulary instead of retraining the tokenizer.**