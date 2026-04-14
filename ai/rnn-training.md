# Why Is My RNN Generating "wan wan wan..."? A One-Line Bug That Breaks Backpropagation

When training a character-level RNN from scratch, you might encounter training output like this and wonder if something is wrong:

```
Epoch 1/1000   - Batch 7/7, Loss: 6.2105  → "a a a a a a a a a a a ..."
Epoch 101/1000 - Batch 7/7, Loss: 0.0600  → "rithan wwan ows pact — Dhan wwwwww..."
Epoch 201/1000 - Batch 7/7, Loss: 0.0215  → "rithan wwwan wwan wan wan wan wan ..."
Epoch 301/1000 - Batch 7/7, Loss: 0.0172  → "rithan wwwan wwan wwan wwan wwan ..."
Epoch 401/1000 - Batch 7/7, Loss: 0.0142  → "rithan wwwan wwan wwan wwan wwan ..."
Epoch 501/1000 - Batch 7/7, Loss: 0.0108  → "re onf nal tonf wan wan wan wan ..."
Epoch 601/1000 - Batch 7/7, Loss: 0.0099  → "re onf nampact — Dd) — Dd) — Dd) ..."
Epoch 701/1000 - Batch 7/7, Loss: 0.0093  → "ref, angle.READ.Dd) — Dd) — Dd) ..."
Epoch 801/1000 - Batch 7/7, Loss: 0.0089  → "ref, angle.READ.Dd) — Dhan wwwan ..."
Epoch 901/1000 - Batch 7/7, Loss: 0.0086  → "ref, angle.READ.Dd) — Dhan wwwan ..."
```

**This is not normal.** The loss looks like it converges nicely, but the generated text is stuck in repetitive loops — "wan wan wan...", "Dd) — Dd) — Dd)...". The model is not actually learning the sequence.

## The Root Cause: A Silent No-Op Loop

The culprit is a deceptively innocent-looking line:

```python
# BUGGY CODE
for loss in losses: pass   # This loop does nothing
loss.backward()            # Only backpropagates through the last timestep
```

`for loss in losses: pass` iterates through the losses list but takes no action. After the loop, `loss` simply holds the **last element** of the list. The subsequent `loss.backward()` therefore only computes gradients for that single final timestep.

## Why This Breaks Learning

An RNN processes a sequence step by step, producing a loss at each timestep. The correct training signal requires **all those losses** to flow backward through the entire unrolled network.

When only the last loss is backpropagated:

1. **Gradient coverage is minimal.** Only the final timestep contributes to weight updates. Earlier timesteps receive zero gradient signal.
2. **The reported loss is artificially low.** `total_loss` accumulates only the last step's loss instead of the full sequence loss, making convergence look better than it really is.
3. **The model learns short-range shortcuts.** Without full-sequence gradient flow, the model discovers it can minimize loss by repeating local patterns — hence "wan wan wan..." and similar loops.

## The Fix

Sum all per-timestep losses into a single scalar before calling `.backward()`:

```python
# CORRECT CODE — option 1: sum then backward
total_loss = sum(losses)
total_loss.backward()
```

Or equivalently, accumulate in-place during the forward pass:

```python
# CORRECT CODE — option 2: accumulate during forward pass
total_loss = torch.tensor(0.0)
for t in range(seq_len):
    logits = model(x[:, t])
    total_loss = total_loss + criterion(logits, y[:, t])

total_loss.backward()
```

Both approaches ensure that gradients propagate through every timestep of the sequence.

## What Correct Training Looks Like

With the fix applied, the generated text should improve coherence as training progresses instead of collapsing into repetition. Loss values will also be higher initially (reflecting the true full-sequence error), and the model will generalize across long-range dependencies.

## Takeaway

Python's `for x in items: pass` is a legal no-op and produces no error or warning. When it appears just before a `.backward()` call in an RNN training loop, it silently breaks BPTT (Backpropagation Through Time) — one of the hardest bugs to spot because the training *appears* to be working. Always verify that the loss you call `.backward()` on is a **sum or mean over all timesteps**, not just the last one.
