---
title: Run wandb in python
categories: [visualization, wandb]
tags: [python, wandb]
---

```python
import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="test",
    # Set the wandb project where this run will be logged.
    project="test",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()
```

## How Hugging Face Trainer Logs to W&B

The example above uses W&B manually. It calls:

```python
run.log({"acc": acc, "loss": loss})
```

Hugging Face `Trainer` uses a different path. After W&B reporting is enabled, training metrics are produced by the `Trainer` logging mechanism and forwarded by `WandbCallback`:

```text
Trainer training loop
    -> Trainer.log(...)
        -> TrainerState.log_history.append(...)
        -> CallbackHandler.on_log(...)
            -> WandbCallback.on_log(...)
                -> wandb.log(...)
```

### Which function creates the training logs?

The main method is:

```python
Trainer.log()
```

It usually records values such as:

```python
{
    "loss": 0.42,
    "learning_rate": 1e-5,
    "epoch": 1.5,
}
```

The current optimizer step is available as:

```python
self.state.global_step
```

After training, the accumulated log entries can be inspected with:

```python
trainer.train()
print(trainer.state.log_history)
```

The result may look like this:

```python
[
    {
        "loss": 0.72,
        "learning_rate": 5e-5,
        "epoch": 0.36,
        "step": 500,
    },
    {
        "loss": 0.51,
        "learning_rate": 4e-5,
        "epoch": 0.72,
        "step": 1000,
    },
]
```

The exact contents of each entry can vary between Transformers versions. In particular, the step is primarily managed by `state.global_step` and may not be included in the dictionary in every version.

### Which function sends the logs to W&B?

The Hugging Face W&B integration is implemented by:

```python
transformers.integrations.WandbCallback
```

Its key method is:

```python
WandbCallback.on_log(...)
```

Conceptually, it receives the logs produced by `Trainer.log()` and calls W&B like this:

```python
def on_log(self, args, state, control, logs=None, **kwargs):
    if logs is not None:
        wandb.log(logs, step=state.global_step)
```

The responsibilities are therefore:

- `Trainer.log()`: creates and stores training logs.
- `TrainerState.global_step`: represents the current optimizer step.
- `WandbCallback.on_log()`: receives Trainer logs and forwards them.
- `wandb.log()`: writes the metrics to W&B.

### Why are logs not shown for every step?

By default, `Trainer` does not necessarily call `Trainer.log()` after every training step. The logging schedule is controlled by arguments such as:

```python
TrainingArguments(
    logging_strategy="steps",
    logging_steps=500,
    logging_first_step=True,
)
```

This records logs at steps 500, 1000, 1500, and so on. The training loop checks `control.should_log` and only then calculates the metrics and calls `self.log(logs)`.

Therefore, values such as `500` and `1000` usually come from `logging_steps=500`; W&B is not randomly dropping the intermediate data.

Logging can also be configured per epoch:

```python
TrainingArguments(
    logging_strategy="epoch",
)
```

### Logging, evaluation, and saving use different schedules

These mechanisms are independent:

```python
TrainingArguments(
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    report_to="wandb",
)
```

```text
logging_steps  -> records training metrics such as loss and learning_rate
eval_steps     -> runs evaluation and records metrics such as eval_loss
save_steps     -> saves a checkpoint
```

If a complete group of values appears at step 1000, it may be because logging, evaluation, and saving happen to use compatible intervals. They are still controlled by separate mechanisms.

### What do `step` and `epoch` mean?

`step` normally refers to `state.global_step`, which is the optimizer step rather than every dataloader batch. With gradient accumulation enabled:

```python
gradient_accumulation_steps=4
```

`global_step` increases approximately once for every four processed batches.

`epoch` is estimated by `Trainer` from the amount of data processed and added to the logs. W&B receives these fields as metrics and can plot them as curves. The W&B callback also associates the Trainer's `global_step` with the W&B logging step.

The most useful settings to check when logs appear only at intervals are:

1. `logging_steps`, for training metrics such as `loss` and `learning_rate`.
2. `eval_steps`, for evaluation metrics such as `eval_loss`.
3. `save_steps`, for checkpoint creation.

## Why Does `step=2` Produce Only One Point?

`TrainerState.global_step` increases automatically in `Trainer`, but it does not increase after every batch. It increases after each optimizer step.

For example, if a manual script repeatedly does this:

```python
run.log(
    {"acc": acc, "loss": loss},
    step=2,
)
```

every log entry uses the same W&B step. W&B steps are expected to increase monotonically. Multiple logs with the same step do not create separate x-axis points; their values may be merged or overwritten, so the chart can appear to contain only one point.

This code creates several points:

```python
step = epoch * 2
```

```text
epoch=0 -> step=0
epoch=1 -> step=2
epoch=2 -> step=4
...
```

However, the multiplier is an arbitrary step invented by the script. It does not represent a real optimizer update.

For a hand-written training loop, the simplest option is to omit `step` and let W&B assign its internal monotonically increasing step:

```python
for epoch in range(epochs):
    # Training ...

    run.log({
        "train/acc": acc,
        "train/loss": loss,
        "epoch": epoch,
    })
```

You can also maintain an explicit step yourself:

```python
global_step = 0

for epoch in range(epochs):
    # Training ...

    global_step += 1
    run.log(
        {
            "train/acc": acc,
            "train/loss": loss,
            "epoch": epoch,
        },
        step=global_step,
    )
```

## `global_step` in Trainer

Conceptually, the relevant part of the `Trainer` loop looks like this:

```python
for batch in train_dataloader:
    loss = model(batch)
    loss.backward()

    if should_update_optimizer:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        state.global_step += 1
        self._maybe_log_save_evaluate()
```

Thus:

```python
state.global_step
```

usually represents the number of completed optimizer updates.

With:

```python
gradient_accumulation_steps=4
```

the trainer normally processes four batches before performing one optimizer update, so `global_step` increases by one.

`WandbCallback` passes this value to W&B, conceptually as:

```python
wandb.log(logs, step=state.global_step)
```

Consequently, you normally do not need to write this yourself in a `Trainer`-based script:

```python
step=state.global_step
```

## Trainer Logging Intervals

For example:

```python
TrainingArguments(
    report_to="wandb",
    logging_strategy="steps",
    logging_steps=500,
)
```

The trainer will approximately call:

```python
self.log(logs)
```

at steps 500, 1000, 1500, and so on. With:

```python
logging_strategy="epoch"
```

it normally records training logs once at the end of each epoch.

## `train`, `eval`, `System`, and `Charts`

These are usually areas that W&B creates in the Workspace to organize metrics. They are not three directories or separate log files written by `Trainer`.

Trainer may send metrics such as:

```python
{
    "loss": 0.42,
    "learning_rate": 1e-5,
    "epoch": 1.2,
    "eval_loss": 0.35,
    "eval_accuracy": 0.88,
}
```

W&B uses metric names and their sources when displaying them:

- `train`: training metrics such as `loss` and `learning_rate`.
- `eval`: evaluation metrics such as `eval_loss` and `eval_accuracy`.
- `System`: system monitoring metrics such as CPU, GPU, memory, and process information.
- `Charts`: the chart panels used to visualize metrics.

The manual script in this post only logs:

```python
run.log({"acc": acc, "loss": loss})
```

It therefore creates ordinary chart metrics. It does not automatically create Trainer-style evaluation metrics or a Trainer log structure.

Metric namespaces make the distinction explicit:

```python
run.log({
    "train/acc": acc,
    "train/loss": loss,
    "epoch": epoch,
})
```

During evaluation, use a separate namespace:

```python
run.log({
    "eval/loss": eval_loss,
    "eval/acc": eval_acc,
    "epoch": epoch,
})
```

## Trainer Logs and W&B Logs Are Different

There are two different categories of local logs to distinguish.

Trainer logs are usually stored in:

```text
output_dir/trainer_state.json
```

The `trainer.state.log_history` entries in that file come from `Trainer.log()`.

W&B logs are usually stored below:

```text
wandb/run-*/files/
```

W&B writes its local history asynchronously. The file format and location can change between W&B versions, so `trainer_state.json` and W&B history should not be treated as the same file.

A typical `Trainer` configuration is:

```python
training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
)
```

Also make sure that the environment has not disabled W&B:

```bash
WANDB_DISABLED=true
```

For a manual script, the pattern used in this post is sufficient:

```python
run = wandb.init(...)
run.log(...)
run.finish()
```

For this example, a clean manual-loop version is:

```python
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
    loss = 2**-epoch + random.random() / (epoch + 1) + offset

    run.log({
        "train/acc": acc,
        "train/loss": loss,
        "epoch": epoch,
    })
```

The important part is to avoid manually forcing every log entry to use the same `step`. W&B can then generate a continuous internal x-axis such as `0, 1, 2, ...`.

## Why Can a W&B `logs` Directory Be Empty?

W&B has several different kinds of history that are easy to confuse:

- W&B metric history, created by `run.log()`.
- Hugging Face Trainer history, kept in `trainer.state.log_history`.
- W&B text and diagnostic logs, such as captured command-line output.

These are related, but they are not the same data or necessarily the same file.

### W&B's internal step

Each `run.log()` call creates a new W&B step by default. W&B recommends logging consecutive steps such as `0, 1, 2, ...`; an arbitrary old history step cannot be rewritten. W&B writes the logged data to a local `wandb` directory and later synchronizes it with W&B Cloud or a private server.

For example, repeatedly doing this is incorrect:

```python
run.log({"loss": loss}, step=2)
```

All entries are assigned to the same history step, so they cannot form a normal sequence of x-axis values. The manual loop should usually omit `step`:

```python
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
    loss = 2**-epoch + random.random() / (epoch + 1) + offset

    run.log({
        "train/acc": acc,
        "train/loss": loss,
        "epoch": epoch,
    })
```

### Why `test/test/runs/7zwgcdtp/logs` may contain nothing

The manual script in this post does not print anything. It calls `run.log()` to record metrics, but it does not produce ordinary terminal text:

```python
run = wandb.init(...)
run.log(...)
run.finish()
```

W&B can capture stdout and stderr as command-line logs. Since this script has little or no console output, its local `logs` directory may be empty or contain only minimal diagnostic data. That does not mean that `acc` and `loss` were not recorded.

Metric history is stored as part of the run's local data. Depending on the W&B SDK version, it may be held in the run's binary `.wandb` file or managed by the SDK's background process rather than appearing as readable text under `logs`.

A run directory may look roughly like this, although the exact layout is version-dependent:

```text
wandb/
  run-.../
    files/
      wandb-summary.json
      wandb-metadata.json
    logs/
      debug.log
      debug-internal.log
    run-....wandb
```

### Why a Trainer run has more log files

Hugging Face `Trainer` normally produces more console output than the small manual script, including training information, progress bars, evaluation results, and checkpoint messages. W&B can capture that stdout/stderr output, so a Trainer run may have more content in its local `logs` directory or in the Logs view of the W&B run page.

The difference can therefore be summarized as:

```text
Trainer run:
    more console output
        -> W&B captures stdout/stderr
        -> text or diagnostic logs are more likely to contain content

Manual run:
    run.log() records metrics but does not print them
        -> metric history still exists
        -> the text logs may be empty
```

This is independent of whether the metric charts are working.

## Why Trainer Runs Have `train` and `eval` Metrics

Trainer may send training metrics such as:

```python
{
    "loss": 0.42,
    "learning_rate": 1e-5,
    "epoch": 1.5,
}
```

During evaluation, it may send metrics such as:

```python
{
    "eval_loss": 0.35,
    "eval_accuracy": 0.88,
    "epoch": 1.5,
}
```

These metrics pass through the callback system:

```text
Trainer.log()
    -> on_log()
        -> WandbCallback.on_log()
            -> wandb.log()
```

Trainer also keeps its own history:

```python
trainer.state.log_history
```

When the Trainer state is saved, it is commonly written below the configured output directory as:

```text
output_dir/trainer_state.json
```

This file is Hugging Face's training state, not W&B's `logs` directory.

The relevant Trainer schedules are separate:

```python
TrainingArguments(
    logging_strategy="steps",
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
)
```

- `logging_steps`: how often to record training metrics.
- `eval_steps`: how often to run evaluation.
- `save_steps`: how often to save a checkpoint.

These values are based on update steps, meaning optimizer updates, rather than necessarily every batch.

## Does `state.global_step` Increase Automatically?

Yes, when using Hugging Face `Trainer`. It is maintained by the Trainer and represents completed optimizer update steps:

```python
state.global_step
```

With:

```python
gradient_accumulation_steps=4
```

the relationship is approximately:

```text
batch 1 -> global_step unchanged
batch 2 -> global_step unchanged
batch 3 -> global_step unchanged
batch 4 -> optimizer.step()
         -> global_step += 1
```

A hand-written loop has no `state.global_step`. To track an equivalent value, maintain it explicitly:

```python
global_step = 0

for epoch in range(epochs):
    # Forward, backward, and optimizer.step() ...

    global_step += 1
    run.log({
        "train/loss": loss,
        "train/acc": acc,
        "epoch": epoch,
        "global_step": global_step,
    })
```

Here, `global_step` is an ordinary metric. It is usually better not to pass it simultaneously as `step=global_step`, because W&B already assigns a consecutive internal history step to each `run.log()` call.

If the charts should use a custom training step as their x-axis, define that axis explicitly:

```python
run.define_metric("train/*", step_metric="train/step")
run.define_metric("eval/*", step_metric="eval/step")

global_step = 0

for epoch in range(epochs):
    global_step += 1
    run.log({
        "train/step": global_step,
        "train/loss": loss,
        "train/acc": acc,
        "epoch": epoch,
    })

run.log({
    "eval/step": global_step,
    "eval/loss": eval_loss,
    "eval/acc": eval_acc,
})
```

In this example, `train/step` and `eval/step` are custom chart axes. They are not the W&B internal history step.

## Keep the Log Types Separate

```text
run.log()
    records W&B metrics

Trainer.log()
    creates Trainer log entries and triggers callbacks

WandbCallback.on_log()
    forwards Trainer logs to W&B

trainer.state.log_history
    Hugging Face's in-memory log list

trainer_state.json
    Hugging Face's local Trainer state file

wandb/.../logs/
    W&B SDK diagnostic and captured console logs

wandb/.../*.wandb or SDK-managed history
    local W&B metric data
```

Therefore, an empty `test/test/runs/7zwgcdtp/logs` directory does not by itself show that W&B failed to record metrics. The reliable checks are the `acc` and `loss` charts on the W&B run page, the run's summary/history data, and, for a Trainer run, `trainer_state.json`.

## References

- [W&B Logging](https://docs.wandb.ai/models/track/log): W&B metric logging, consecutive internal steps, stdout/stderr capture, and local synchronization.
- [W&B Custom Logging Axes](https://docs.wandb.ai/models/track/log/customize-logging-axes/): `define_metric()` and custom `step_metric` axes.
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer): `Trainer.log()`, `TrainingArguments` logging/evaluation/saving intervals, and update-step behavior.
- [Hugging Face Trainer Callbacks](https://huggingface.co/docs/transformers/main_classes/callback): `WandbCallback`, `on_log`, `TrainerState`, `log_history`, and callback control flow.