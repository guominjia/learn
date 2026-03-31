# Why Perplexity Rises at the Start of RNN Training

> A practical look at the counterintuitive phenomenon where perplexity temporarily increases during early training — and how to fix it.

---

## 1. The Phenomenon

You launch RNN training, monitor the perplexity curve, and notice something unsettling: **perplexity goes up before it starts going down**. This is surprisingly common and usually not a bug — it is a natural consequence of how the training process begins.

---

## 2. Root Causes

### 2.1 Random Initialization Creates a "Uniform Baseline"

At initialization, model weights are random. The output distribution over the vocabulary is close to **uniform**, meaning every word is roughly equally likely.

$$\text{Perplexity}_{\text{uniform}} = |V|$$

where $|V|$ is the vocabulary size. For a 10,000-word vocabulary, initial perplexity hovers around 10,000.

The first few gradient updates can **break this uniformity in the wrong direction** — the model becomes overconfident about incorrect tokens before learning the right patterns. This temporarily pushes perplexity *above* the uniform baseline.

### 2.2 Learning Rate Too Large

When the initial learning rate is too aggressive:

| Step | What Happens |
|------|-------------|
| 0 | Weights are random; output ≈ uniform |
| 1–5 | Large gradient steps overshoot the loss surface |
| 5–50 | Model oscillates before settling into a descent direction |
| 50+ | Perplexity begins to decrease steadily |

The model "jumps" around on the loss landscape before finding a downhill direction, causing transient spikes.

### 2.3 Hidden State Instability

RNNs propagate information through hidden states across time steps:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)$$

During early training:
- The hidden-to-hidden weight matrix $W_{hh}$ has not yet learned stable dynamics
- Hidden states **amplify noise** rather than propagating useful sequential information
- This results in chaotic gradient signals that temporarily degrade predictions

### 2.4 Softmax Logit Scale Issues

The final linear layer produces logits that are fed into softmax:

$$p(w_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

After a few updates, the logit magnitudes $z_i$ can become more extreme (larger variance) before the model learns to calibrate them. This makes the model confidently wrong, which the cross-entropy loss penalizes severely:

$$\mathcal{L} = -\log p(w_{\text{target}})$$

Being *confidently wrong* yields a much higher loss than being *uniformly uncertain*.

### 2.5 Batch-Level Sampling Variance

If you are monitoring per-batch perplexity rather than epoch-level averages, the apparent "rise" may simply be **statistical noise** — different batches expose the model to different difficulty levels, producing high variance in the first few steps.

---

## 3. Mitigation Strategies

### 3.1 Learning Rate Warmup

Start with a very small learning rate and ramp it up over the first $N$ steps:

```python
from torch.optim.lr_scheduler import LambdaLR

warmup_steps = 100
scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup_steps)
)
```

This prevents large initial jumps and lets the model gently find a good descent direction.

### 3.2 Gradient Clipping

Constrain gradient norms to prevent individual updates from being too destructive:

```python
import torch.nn as nn

# Clip gradients to max norm of 5.0
nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

This is especially critical for vanilla RNNs, which are prone to exploding gradients.

### 3.3 Proper Weight Initialization

Avoid initializing weights with too large or too small a scale:

```python
# Xavier initialization for RNN layers
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param)
    elif 'bias' in name:
        nn.init.zeros_(param)
```

Xavier/Glorot initialization keeps the variance of activations roughly constant across layers, preventing the logit scale issues described in §2.4.

### 3.4 Use Gated Architectures (LSTM / GRU)

Gated recurrent units introduce **forget, input, and output gates** that regulate information flow through hidden states:

| Architecture | Hidden State Update | Stability |
|-------------|-------------------|-----------|
| Vanilla RNN | $h_t = \tanh(W \cdot [h_{t-1}, x_t])$ | Poor — prone to vanishing/exploding gradients |
| LSTM | Gated via $f_t, i_t, o_t$ and cell state $c_t$ | Good — gates learn to preserve or discard information |
| GRU | Gated via $z_t, r_t$ (fewer parameters than LSTM) | Good — simpler alternative to LSTM |

Gates provide a "highway" for gradients, making early training more stable and reducing the initial perplexity spike.

---

## 4. A Practical Training Loop with All Mitigations

```python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# Model: LSTM instead of vanilla RNN
model = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Xavier init
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param)

# Warmup scheduler
warmup_steps = 200
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output, _ = model(batch)
        loss = criterion(output, targets)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        scheduler.step()
```

---

## 5. When to Worry

| Situation | Action |
|-----------|--------|
| Perplexity rises for 1–3 epochs then drops | **Normal** — no action needed |
| Perplexity rises for 5+ epochs | Check learning rate, try reducing by 10× |
| Perplexity rises and never decreases | Investigate data preprocessing, model architecture, or label errors |
| Perplexity oscillates wildly | Apply gradient clipping; reduce learning rate |

---

## 6. Summary

A temporary rise in perplexity at the start of RNN training is typically caused by the transition from a uniform random output distribution to an initially miscalibrated one. The model must first "unlearn" bad confidence before it can learn good predictions. This is a well-understood phenomenon, and the standard toolkit — **warmup, gradient clipping, proper initialization, and gated architectures** — effectively addresses it.
