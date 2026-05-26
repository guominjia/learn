# Rich Progress: Beautiful Progress Bars in Python

Rich is a Python library for rendering rich text and beautiful formatting in the terminal. One of its most useful features is `rich.progress`, which provides highly customizable progress bars for long-running operations.

## Installation

```bash
pip install rich
```

## 1. Basic Progress Bar with Custom Columns

The `Progress` context manager is the core API. You compose the bar's appearance by passing **column** objects that control what information is displayed.

```python
from time import sleep
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

columns = [
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
]

with Progress(*columns) as progress:
    task = progress.add_task("Downloading", total=100)
    for i in range(100):
        sleep(0.03)
        progress.update(task, advance=1)
```

Key points:
- `add_task()` registers a new task and returns a task ID.
- `update()` with `advance=1` increments the completed count by one.
- The `with` block automatically starts and stops the live display.

## 2. Quick Iteration with `track()`

For simple loops where you just need to wrap an iterable, `track()` is the most concise option — no context manager required.

```python
from time import sleep
from rich.progress import track

for item in track(range(100), description="Processing..."):
    sleep(0.03)
    # process item
```

`track()` is essentially syntactic sugar over `Progress`. It infers the total from the iterable length and handles everything automatically.

## 3. Indeterminate (Unknown Total) Tasks

When you don't know the total number of steps in advance — for example, reading from a stream — set `total=None`. Rich renders a spinner instead of a percentage bar.

```python
from time import sleep
from rich.progress import Progress, SpinnerColumn, TextColumn

with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
    task = progress.add_task("Working...", total=None)
    for i in range(50):
        sleep(0.05)
        progress.update(task, advance=1)
```

With `total=None`, the progress bar cannot show a percentage or time remaining, but it still tracks the number of completed steps.

## 4. Updating Descriptions and Transient Display

You can update the task description mid-flight to reflect different phases. Setting `transient=True` removes the progress bar from the terminal once it finishes, keeping the output clean.

```python
from time import sleep
from rich.progress import Progress, BarColumn

with Progress("[progress.description]{task.description}", BarColumn(), transient=True) as progress:
    task = progress.add_task("Phase 1", total=50)
    for i in range(50):
        sleep(0.02)
        progress.update(task, advance=1)
    progress.update(task, description="Finalizing")
```

This is useful in CLI tools where you want to show progress during execution but leave a clean terminal when done.

## 5. Async Support with `asyncio`

`Progress` works seamlessly with Python's `asyncio`. Use `async with` and share the progress instance across concurrent coroutines.

```python
import asyncio
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn

async def worker(progress, task_id):
    for _ in range(20):
        await asyncio.sleep(0.05)
        progress.update(task_id, advance=1)

async def main():
    columns = [SpinnerColumn(), BarColumn(), TimeElapsedColumn()]
    with Progress(*columns) as progress:
        t = progress.add_task("Async tasks", total=40)
        await asyncio.gather(worker(progress, t), worker(progress, t))

asyncio.run(main())
```

Both workers share a single task, so their combined `advance` calls fill the bar to 40. You can also create separate tasks per coroutine for independent tracking.

## Tips and Best Practices

| Tip | Details |
|-----|---------|
| **Transient mode** | Use `transient=True` to remove the progress bar after completion. |
| **Unknown total** | Set `total=None` for indeterminate tasks; only `advance` and `description` updates work. |
| **Custom columns** | Mix and match `SpinnerColumn`, `BarColumn`, `TextColumn`, `TimeElapsedColumn`, `TimeRemainingColumn`, etc. |
| **Simple loops** | Prefer `track()` for straightforward iteration over a known-length sequence. |
| **Complex scenarios** | Use the `Progress` context manager directly to manage multiple tasks, dynamic descriptions, or async workloads. |
| **Thread safety** | `Progress` is thread-safe — you can call `update()` from multiple threads without additional locking. |

## References

- [Rich Documentation — Progress Bars](https://rich.readthedocs.io/en/latest/progress.html)
- [Rich GitHub Repository](https://github.com/Textualize/rich)
- [PyPI — rich](https://pypi.org/project/rich/)