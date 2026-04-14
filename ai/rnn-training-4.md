# Why `Cannot backpropagate more than once` Happens in RNN/LSTM Training

When training an RNN/LSTM with a custom autograd engine, you may see this error:

```text
Exception: Cannot backpropagate more than once
```

This usually appears when `hidden` state is reused across batches without detaching it from the previous computation graph.

---

## The Symptom

The following pattern works:

```python
for batch_i in range(len(input_batches)):
  hidden = (
    Tensor(hidden[0].data, autograd=True),
    Tensor(hidden[1].data, autograd=True),
  )

  for t in range(len(input_batches[batch_i])):
    # forward through time
    ...

  loss.backward()
```

But if you only re-wrap `hidden` once before `for i in range(iterations):`, training can fail with:

```text
Cannot backpropagate more than once
```

---

## Root Cause

`hidden` is not just numeric data. It also carries autograd history (the graph).

If you do this at the start of every batch:

```python
hidden = (
  Tensor(hidden[0].data, autograd=True),
  Tensor(hidden[1].data, autograd=True),
)
```

you effectively **detach** from the old graph:

- keep tensor values
- drop graph history

So each `loss.backward()` only walks the current batch graph.

If you do it only once before the outer iteration loop:

1. Batch 1 finishes, and new `hidden` is now connected to Batch 1 graph.
2. Batch 2 reuses that `hidden`, chaining Batch 2 graph onto Batch 1 graph.
3. Calling `backward()` again tries to traverse nodes/edges already backpropagated.
4. Your autograd engine detects repeated backward on the same path and raises the exception.

---

## Correct Mental Model

The key is not whether detach is inside `for i` or outside it.
The key is:

> Before each new sequence segment that will call `backward()`, detach `hidden` from the old graph.

This is exactly **truncated BPTT** behavior.

---

## Practical Rule

- Across time steps (`t`) in the same segment: keep `hidden` connected.
- Across batches (or before the next `backward()`): detach `hidden`.

Example helper style:

```python
h = Tensor(h.data, autograd=True)
c = Tensor(c.data, autograd=True)
```

or provide a dedicated `detach()` method for clarity.

---

## Takeaway

If you see `Cannot backpropagate more than once` in recurrent training, first inspect hidden-state lifecycle. In most cases, detaching hidden state at the right boundary (between backward passes) resolves the issue cleanly.