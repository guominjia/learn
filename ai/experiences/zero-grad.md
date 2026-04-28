---
title: "Where Should optimizer.zero_grad() Go in Your PyTorch Training Loop?"
date: 2026-04-28
tags: [pytorch, deep-learning, training, gradient]
---

# Where Should `optimizer.zero_grad()` Go in Your PyTorch Training Loop?

If you've written a PyTorch training loop, you've probably seen `optimizer.zero_grad()` placed in different positions. Some codebases put it **before** the forward pass, others put it **after** computing the loss but **before** `backward()`. Does the placement actually matter?

## The Two Common Patterns

### Pattern A — Zero grad after forward, before backward

```python
pred = model(data_batch)
loss = criterion(pred, labels)
optimizer.zero_grad()   # <-- here
loss.backward()
optimizer.step()
```

### Pattern B — Zero grad at the top of the loop (recommended)

```python
optimizer.zero_grad()   # <-- here
pred = model(data_batch)
loss = criterion(pred, labels)
loss.backward()
optimizer.step()
```

**Short answer:** In most standard supervised-training scenarios, both patterns produce equivalent results. But **Pattern B is the recommended practice**.

## Why Does It Work Either Way?

PyTorch accumulates gradients by default — each call to `loss.backward()` **adds** the newly computed gradients to the existing `.grad` tensors on every parameter. So the only hard constraint is:

$$
\text{zero\_grad()} \;\rightarrow\; \text{backward()} \;\rightarrow\; \text{step()}
$$

The **forward pass** (`model(data_batch)`) merely builds the computational graph. It does **not** write anything to `.grad`. Gradients are only materialized during `backward()`. That means calling `zero_grad()` before or after the forward pass has no effect on which gradients get accumulated — as long as it happens before `backward()`.

## Why Pattern B Is Still Preferred

### 1. Clearer semantics

Placing `zero_grad()` at the top of the iteration communicates intent: *"this training step starts from a clean slate."* Anyone reading the code immediately understands that no stale gradients carry over.

### 2. Safer in complex pipelines

In practice, training loops grow beyond the simple five-line template. You might:

- Compute auxiliary losses
- Run a discriminator forward pass in a GAN
- Access `.grad` for logging or clipping before `backward()`

If `zero_grad()` sits right before `backward()`, it can accidentally erase gradients that were intentionally accumulated earlier in the same step. Placing it at the very beginning avoids this class of bugs.

### 3. Aligns with community convention

PyTorch's official tutorials, Hugging Face Trainer internals, and the vast majority of open-source projects all use Pattern B. Following the convention reduces cognitive load for collaborators and reviewers.

## When Placement Actually Matters

While the two patterns are interchangeable for vanilla training, several advanced scenarios make the distinction meaningful:

| Scenario | Why placement matters |
|---|---|
| **Gradient accumulation** | You intentionally skip `zero_grad()` for *N* steps, then clear. Placing it at the top with a conditional guard (`if step % N == 0`) is the cleanest approach. |
| **Multiple backward passes** | When composing several losses that each call `backward()`, you need precise control over when gradients reset. |
| **Mixed-precision training** | AMP scalers interact with the gradient lifecycle; a predictable zero-then-forward-then-backward order prevents subtle scaling bugs. |
| **Custom training logic** | Any loop that inspects or modifies `.grad` between forward and backward benefits from knowing exactly when gradients were last cleared. |
| **Fault tolerance** | If an iteration is interrupted (e.g., by a data-loading error), starting with `zero_grad()` guarantees the next iteration cannot inherit corrupted gradients. |

## The Recommended Training Loop

```python
for data_batch, labels in dataloader:
    optimizer.zero_grad()

    pred = model(data_batch)
    loss = criterion(pred, labels)

    loss.backward()
    optimizer.step()
```

This is the canonical five-line loop. Memorize it, use it as your starting point, and deviate only when you have an explicit reason.

## A Note on `model.zero_grad()` vs `optimizer.zero_grad()`

You may also encounter `model.zero_grad()`. Both do the same thing — zero out `.grad` for all parameters the optimizer manages — **if** the optimizer's parameter groups cover the entire model. When in doubt, prefer `optimizer.zero_grad()` because it is scoped to exactly the parameters being optimized, which is safer when multiple optimizers are involved (e.g., in GANs).

Since PyTorch 1.7, you can also pass `set_to_none=True`:

```python
optimizer.zero_grad(set_to_none=True)
```

This sets `.grad` to `None` instead of filling with zeros, which can be slightly more memory-efficient and marginally faster. It is the **default behavior since PyTorch 2.0**.

## Key Takeaways

- `zero_grad()` does **not** need to come before `model()` — it just needs to precede `backward()`.
- Placing it at the **top of the loop** is the clearest, safest, and most conventional choice.
- The critical invariant: **always clear stale gradients before the next `backward()` call**.
