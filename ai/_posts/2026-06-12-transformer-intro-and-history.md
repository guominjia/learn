---
layout: post
title: "The Transformer: Introduction and History"
date: 2026-06-12
tags: [transformer, attention, nlp, history, llm]
---

The Transformer architecture is arguably the most influential neural network design of the last decade. From BERT to GPT to modern large language models (LLMs), virtually every state-of-the-art system in natural language processing — and increasingly in computer vision, audio, and multimodal AI — is built on or inspired by the Transformer. This post traces its origins, explains its core ideas, and follows the architectural milestones that shaped the field.

## Before the Transformer: The RNN Era

Prior to 2017, sequence modeling was dominated by **Recurrent Neural Networks (RNNs)** and their gated variants:

- **RNN (Elman, 1990)** — processes sequences step-by-step, maintaining a hidden state.
- **LSTM (Hochreiter & Schmidhuber, 1997)** — introduced gating to preserve long-term dependencies.
- **GRU (Cho et al., 2014)** — a simpler, faster gated unit.

These architectures were effective but suffered from two critical limitations:

1. **Sequential computation** — each token depends on the previous one, preventing parallelism during training.
2. **Long-range dependency degradation** — even LSTMs struggled to retain information across very long sequences.

The encoder-decoder framework with attention (Bahdanau et al., 2015) partially addressed the second problem by allowing the decoder to look back at all encoder states, but the sequential bottleneck remained.

## The Breakthrough: "Attention Is All You Need" (2017)

In June 2017, researchers at Google Brain and Google Research — Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin — published the paper:

> **"Attention Is All You Need"** — [arXiv:1706.03762](https://arxiv.org/pdf/1706.03762)

The central claim: *you don't need recurrence or convolution at all*. A model built entirely from attention mechanisms and feed-forward layers could outperform all prior approaches on machine translation while being significantly faster to train.

### Core Architecture

The Transformer follows an encoder-decoder structure:

| Component | Role |
|---|---|
| **Input Embedding + Positional Encoding** | Converts token IDs to vectors; injects sequence order |
| **Multi-Head Self-Attention** | Each token attends to every other token in the sequence |
| **Feed-Forward Network (FFN)** | Position-wise, applied independently to each token |
| **Layer Normalization + Residual Connections** | Stabilize training and enable deep stacking |
| **Encoder-Decoder Cross-Attention** | Decoder attends over the full encoder output |

The key insight is **self-attention**: for every token, compute a weighted sum over all other tokens in the sequence, where the weights represent relevance. This runs in parallel over the entire sequence in a single matrix operation.

The attention function is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where $Q$, $K$, $V$ are queries, keys, and values — each a linear projection of the input. **Multi-head attention** runs $h$ such attention functions in parallel, then concatenates their outputs, allowing the model to jointly attend to information at different positions and representation subspaces.

### Positional Encoding

Since self-attention has no inherent sense of order, positions are injected via **sinusoidal positional encodings** added to the input embeddings:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

This design allows the model to generalize to sequence lengths unseen during training.

## The Rapid Evolution: 2018–2020

### BERT (2018) — Bidirectional Pre-training

Google's **BERT** (Devlin et al., 2018) used only the Transformer encoder and introduced **masked language modeling (MLM)**: randomly mask tokens and train the model to predict them. This bidirectional pre-training strategy produced general-purpose representations that could be fine-tuned on downstream tasks, setting new records on GLUE, SQuAD, and many others.

### GPT-1, GPT-2 (2018–2019) — Autoregressive Language Modeling

OpenAI's **GPT** series used only the Transformer decoder with causal (left-to-right) self-attention. GPT-2 (2019, 1.5B parameters) demonstrated that large autoregressive language models could generate coherent long-form text, raising significant discussion about potential misuse.

### T5 (2019) — Text-to-Text Unification

Google's **T5** (Raffel et al., 2019) reframed all NLP tasks — classification, summarization, translation, question answering — as text-to-text problems, using a full encoder-decoder Transformer. The "scale matters" finding from T5 directly motivated the push toward ever-larger models.

### XLNet, RoBERTa (2019) — Improved Pre-training

- **XLNet** combined autoregressive modeling with permutation-based training to capture bidirectional context without masking.
- **RoBERTa** demonstrated that BERT was undertrained and that longer training with more data and no next-sentence prediction objective significantly improved performance.

## The Scale Era: 2020–Present

### GPT-3 (2020) — Emergent Few-Shot Abilities

OpenAI's **GPT-3** (175B parameters) was a landmark: a model of this scale displayed remarkable **in-context learning** — solving tasks with just a few examples in the prompt, without any gradient updates. This reframed thinking about what it means to "train" a model for a task.

### Scaling Laws (2020)

Kaplan et al. (OpenAI) published **Neural Scaling Laws**, showing that model performance scales predictably as a power law with compute, data, and parameters. This gave practitioners a principled framework for allocating training budgets.

### DALL-E, ViT — Transformers Beyond NLP (2020–2021)

- **ViT** (Dosovitskiy et al., 2020) applied patch-based Transformers directly to image classification, matching or outperforming CNNs at scale.
- **DALL-E** (OpenAI, 2021) generated images from text descriptions using a discrete VAE and a Transformer decoder.

### InstructGPT / ChatGPT (2022) — RLHF Alignment

**Reinforcement Learning from Human Feedback (RLHF)** — used in InstructGPT and ChatGPT — demonstrated that fine-tuning a large language model to follow instructions dramatically improves usability and safety alignment. This became the standard recipe for deploying LLMs.

### GPT-4, Gemini, LLaMA, Mistral (2023–2024)

The field has since diversified into:

- **Multimodal Transformers** — accepting text, images, audio, and video (GPT-4V, Gemini).
- **Open-weight models** — LLaMA (Meta), Mistral, and derivatives made capable models accessible for research and production.
- **Efficient architectures** — Mixture-of-Experts (MoE) Transformers (e.g., Mixtral) activate only a subset of parameters per token, enabling larger model capacity at lower inference cost.

## Why the Transformer Won

| Property | RNN / LSTM | Transformer |
|---|---|---|
| Parallelism during training | None (sequential) | Full (matrix ops) |
| Long-range dependencies | Difficult | Native (attention span = full sequence) |
| Scalability | Limited | Scales with compute and data |
| Hardware utilization (GPU/TPU) | Poor | Excellent |
| Adaptability to non-NLP tasks | Limited | Vision, audio, code, multimodal |

## Key Takeaways

- The Transformer eliminated recurrence entirely, replacing it with self-attention and enabling unprecedented parallelism.
- Scaling pre-trained Transformers with more data and compute consistently yields stronger models — a finding that has held from BERT to GPT-4.
- The architecture's generality has made it the default backbone across NLP, vision, audio, and multimodal AI.

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). arXiv:1706.03762.
- [tensor2tensor](https://github.com/tensorflow/tensor2tensor) — Original Transformer implementation by the paper's authors.
- Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Brown et al. (2020). Language Models are Few-Shot Learners (GPT-3).
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
