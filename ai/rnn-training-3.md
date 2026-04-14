# From Fast Loss Drop to Weird Tokens: Reading Character-Level LTSM Training Signals

In this experiment, a character-level language model is trained on Markdown text. The model shows a strong reduction in loss, but occasional nonsense tokens still appear in sampled output. This post explains what is going well, why glitches still happen, and what to improve next.

---

## Training Snapshot

The following checkpoints summarize both optimization progress and text quality:

| Iteration | Loss | Typical Output Quality |
|---|---:|---|
| 1 | 60.49 | Mostly random characters |
| 11 | 23.50 | Repeated fragments, weak structure |
| 21 | 5.76 | Real words and Markdown-like tokens appear |
| 31 | 1.40 | Better phrase continuity |
| 41 | 0.95 | Clear Markdown formatting patterns |
| 51 | 1.10 | Small instability (temporary rebound) |
| 61 | 0.44 | Mostly coherent domain terms |
| 71 | 0.27 | Near-training-style content |
| 81 | 0.25 | Stable structure, minor artifacts |
| 91 | 0.23 | Mostly coherent with occasional nonsense |

---

## What the Curve Tells Us

The overall trend is healthy: loss decreases from about `60` to about `0.23`, and generated text transitions from noise to structured Markdown-like output.

At the same time, three practical observations matter:

1. **Minor optimization instability** appears around iteration 51 (`0.95 -> 1.10`), which suggests the update step may be slightly aggressive for this stage.
2. **The model is still improving** at iteration 91, so it likely has not fully converged yet.
3. **Memorization risk** is visible because some sampled lines strongly resemble the training corpus.

---

## Why Nonsense Still Appears at Low Loss

At iteration 91, output includes strings like `forkfnal` and `Compla`. This is common in character-level generation and usually comes from a combination of decoding and context effects.

### 1) Cold-start hidden state during sampling

If sampling starts from a zero hidden state and a short prompt (for example only `#`), the model has almost no context and can drift in early steps.

```python
model.init_hidden(batch_size=1)  # zero state
```

### 2) Greedy decoding error accumulation

Greedy decoding always selects the highest-probability next token:

```python
pred_index = np.argmax(pred.data)
```

When several characters have similar probabilities, choosing the top one deterministically can start an incorrect branch. In autoregressive generation, one wrong character can snowball into several malformed words.

---

## Better Sampling: Temperature + Probabilistic Choice

A robust fix is to sample from a probability distribution instead of using greedy `argmax`.

```python
def softmax_sample(logits, temperature=0.8):
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```

`np.random.choice(len(probs), p=probs)` means:

- Choose one index from `[0, 1, ..., len(probs)-1]`
- The probability of selecting index `i` is exactly `probs[i]`

Example:

```python
probs = [0.1, 0.6, 0.3]
index = np.random.choice(3, p=probs)
```

Here, index `0` is selected with 10%, `1` with 60%, and `2` with 30%.

---

## Practical Improvements

Use these changes to stabilize training and improve generation quality:

```python
# 1) Gradient clipping
for p in parameters:
    p.grad = np.clip(p.grad, -5, 5)

# 2) Lower learning rate
optimizer = SGD(..., alpha=0.005)

# 3) Train longer and/or add more data
iterations = 300
```

And for sampling quality:

- Warm up hidden state with a longer prompt before free generation.
- Use temperature sampling (for example `temperature=0.8`) instead of pure greedy decoding.

---

## Conclusion

This training run is fundamentally successful: the model learns structure quickly and reaches low loss. The remaining nonsense fragments are not a contradiction; they are expected artifacts of character-level generation under limited context and greedy decoding.

In short: optimization is working, generation is improving, and the next gains will come from better decoding strategy, more stable updates, and broader training data.