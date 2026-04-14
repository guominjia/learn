# Watching a Character-Level RNN Learn: Training Observations and the Loss Plateau Problem

Training a recurrent neural network from scratch reveals a lot about how these models actually work — and where they struggle. This post documents a hands-on experiment training a character-level RNN on Markdown text and dives into the most common issue: the loss plateau around 2.3.

---

## The Setup

- **Model**: A single-layer `RNNCell` with 128-dimensional hidden state
- **Activation**: `sigmoid`
- **Sequence length (BPTT)**: 16 time steps
- **Learning rate**: `alpha = 0.1` (fixed, SGD)
- **Vocabulary**: Character-level tokens from Markdown source files
- **Dataset**: 7 batches per epoch

---

## Training Loss Over Time

| Epoch | Loss |
|-------|------|
| 1 | 107.64 |
| 101 | 14.23 |
| 201 | 5.19 |
| 301 | 3.52 |
| 401 | 2.86 |
| 501 | 2.64 |
| 601 | 2.53 |
| 701 | 2.47 |
| 801 | 2.43 |
| 1001 | 2.38 |
| 1101 | 2.36 |
| 1201 | 2.34 |
| 1301 | 2.33 |
| 1401 | 2.32 |
| 1501 | 2.31 |

The loss dropped sharply from 107 down to around 2.3 in the first few hundred epochs, then nearly flattened out.

---

## What the Loss Numbers Actually Mean

After fixing a BPTT bug that previously only backpropagated through the last time step, the total loss is now the **sum of cross-entropy losses across all 16 BPTT steps**:

$$L_{\text{total}} = \sum_{t=1}^{16} L_t$$

So the per-step loss at Epoch 1 is roughly $107.6 / 16 \approx 6.7$, which is consistent with the pre-fix value of ~6.2. The higher raw number post-fix is expected and correct behavior.

---

## How Generated Text Evolved

As training progressed, the quality of sampled text improved significantly:

| Epoch | Sample Output |
|-------|---------------|
| 101 (buggy) | `rithan wwan ows pact...` |
| 101 (fixed) | `projicrosoft notes, and with notes, and with notes...` |
| 501 | `## Purpose` + partial Markdown structure |
| 801 | `small examples`, `README.md`, heading patterns |
| 1001+ | Repeating `Disk/README.md) — Disk/README.md) — ...` |

The model clearly learned Markdown vocabulary, list syntax, and heading conventions. However, repetitive loops (`and data analysing...`, `Disk/README.md...`) crept in at later epochs — a signature failure mode of vanilla RNNs.

---

## Why Does Loss Plateau Around 2.3?

This is the most important question. Three compounding factors explain it:

### 1. Vanishing Gradients (the Primary Cause)

The model uses `sigmoid` as its activation function. The derivative of sigmoid has a maximum value of only **0.25**. Over 16 BPTT steps, the gradient magnitude decays by:

$$0.25^{16} \approx 2.3 \times 10^{-10}$$

This means the model effectively **cannot learn from context more than a few characters back**. It memorizes local short-range patterns but fails to model longer dependencies, capping its ability to reduce loss further.

**Fix**: Switch to `tanh`, whose derivative peaks at 1.0, dramatically slowing gradient decay:

```python
model = RNNCell(128, 128, len(vocabs), activation="tanh")
```

### 2. Fixed Learning Rate with SGD

With `alpha = 0.1` and plain SGD, the optimizer can oscillate around a flat region rather than converging into it. When the loss landscape becomes shallow, a large fixed step size causes the parameters to bounce back and forth rather than descend.

**Fix**: Lower the learning rate, or switch to an adaptive optimizer:

```python
trainer = Trainer(..., alpha=0.01)
```

Alternatively, use momentum-based optimizers like Adam or RMSProp, which adapt the effective step size and handle flat regions much better.

### 3. Model Capacity Bottleneck

A 128-dimensional hidden state with sigmoid saturation limits how much information the model can encode. At character level, 128 hidden units is modest. The model learns that certain character sequences are common (e.g., ` and`, `## `, `README.md`) but cannot represent more complex compositional structure.

The repetitive loops in the output are a direct symptom: the model learns local n-gram probabilities but has no reliable mechanism to break out of repeating cycles.

### 4. Data Coverage Limit

The training corpus (a short Markdown README) covers only 7 batches × 16 BPTT steps = 112 time steps per epoch. The model may have already approached its **capacity ceiling** for this dataset — loss ≈ 2.3 could be close to the theoretical minimum achievable by this architecture on this data.

---

## Key Takeaways

| Issue | Symptom | Fix |
|-------|---------|-----|
| Vanishing gradients (sigmoid) | Early plateau, repetitive loops | Switch to `tanh` or use LSTM/GRU |
| Fixed learning rate (SGD) | Slow post-plateau convergence | Lower `alpha` or use Adam |
| Small hidden size | Limited vocabulary capture | Increase hidden dim (e.g., 256 or 512) |
| Short BPTT window | Cannot learn long-range dependencies | Increase sequence length, use LSTM |

---

## What's Next

The improvements that would have the biggest impact, in order:

1. **Replace sigmoid with tanh** — immediate gradient health improvement, zero cost
2. **Switch to LSTM or GRU** — purpose-built for long-range memory, eliminates most vanishing gradient issues
3. **Use Adam optimizer** — adaptive learning rate handles flat regions automatically
4. **Increase hidden size** — 256 or 512 units for richer character representations
5. **Increase BPTT length** — 32 or 64 steps to capture longer context

Even a vanilla RNN hitting 2.3 loss after this few epochs shows the model is learning meaningfully. The architecture changes above should push it significantly further.

---

## References

- [Understanding LSTM Networks — Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks — Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [On the difficulty of training recurrent neural networks (Pascanu et al., 2013)](https://arxiv.org/abs/1211.5063)