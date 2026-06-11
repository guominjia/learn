---
layout: post
title: "Debugging RAGAS Evaluations: Exceptions, Tracking, and Concurrency Control"
date: 2026-06-11
tags: [ragas, python, asyncio, debugging, llm, evaluation]
---

## Overview

When running RAGAS evaluations in production or during development, several non-obvious behaviors can make debugging difficult: swallowed exceptions, analytics noise in logs, cryptic error formats from `instructor`, and silent data overwrites. This post consolidates practical techniques for diagnosing and fixing each of these issues.

---

## 1. Forcing Exceptions to Surface

By default, RAGAS's `Executor` sets `raise_exceptions=False`, which means per-sample failures are caught and logged rather than propagated. This is convenient for bulk evaluation but makes individual failures invisible.

**Option A: Environment variable** (if the version supports it):

```python
import os
os.environ["RAGAS_RAISE_EXCEPTIONS"] = "true"
```

**Option B: Monkey-patch the dataclass default** (more reliable):

```python
from ragas import executor as _ragas_executor
_ragas_executor.Executor.__dataclass_fields__["raise_exceptions"].default = True
```

Place this before any evaluation call. With exceptions raised, failures will propagate immediately rather than being silently swallowed.

---

## 2. Understanding `track()` — Telemetry, Not Error Logging

If you see `track(...)` calls inside `ragas/llm.py` and wonder why they provide no useful error context, the answer is by design.

- `track` originates from `ragas/_analytics.py` and is a **usage telemetry** function, not a business-level error logger.
- It is wrapped in a `@silent` decorator, so any exception inside it is suppressed and only emitted at `DEBUG` or `INFO` level.
- Seeing "an error occurred with no explanation" in this code path is expected — this channel is for analytics, not diagnostics.

To see the telemetry payload, enable one of these environment variables:

```bash
RAGAS_DEBUG=true
__RAGAS_DEBUG_TRACKING=true
```

Actual evaluation errors live in `ragas/executor.py` and your project's `evaluator.py` exception handling chain — not in the tracking path.

---

## 3. Decoding the `<failed_attempts>` Error Format

When a sample fails, you may see output like:

```
Warning: Task failed with error: ...
<failed_attempts>
  <generation ...>
    <exception></html></exception>
  </generation>
</failed_attempts>
```

Here is what each layer means:

| Layer | Source | Meaning |
|---|---|---|
| `Warning: Task failed with error:` | `experiment.py` | RAGAS caught a single-sample exception in `asyncio.as_completed()` and continued. |
| `<failed_attempts>...</generation>` | `instructor/exceptions.py` | The `__str__` of an `InstructorRetryException`, rendered as XML-style text when all retries are exhausted. |
| `<exception></html></exception>` | Upstream gateway | The LLM endpoint returned an HTML error page (4xx/5xx proxy error), not a valid JSON response. |

The presence of `</html>` inside `<exception>` means the underlying HTTP response was an HTML error page from a proxy or gateway — not a RAGAS-generated artifact. The actual error detail is in the raw HTTP response, not the truncated exception string.

**Call chain summary:**

```
RAGAS task failure
  → exception caught in experiment.py
    → print(e) calls instructor's __str__
      → renders as <failed_attempts> XML
        → <exception> contains raw HTTP body (HTML page)
```

**Relevant source locations:**

| Symbol | File | Role |
|---|---|---|
| [`ExperimentWrapper.arun`](https://github.com/vibrantlabsai/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/src/ragas/experiment.py#L141) | `experiment.py:141` | Drives async task loop with `asyncio.as_completed` |
| [`experiment` (warning site)](https://github.com/vibrantlabsai/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/src/ragas/experiment.py#L201) | `experiment.py:201` | Catches per-sample exception, prints `Warning: Task failed` |
| [`InstructorLLM.agenerate`](https://github.com/vibrantlabsai/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/src/ragas/llms/base.py#L1117) | `llms/base.py:1117` | Calls instructor client; raises `InstructorRetryException` on retry exhaustion |
| [`_get_instructor_client`](https://github.com/vibrantlabsai/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/src/ragas/llms/base.py#L566) | `llms/base.py:566` | Constructs the instructor client used by `agenerate` |
| [`llm_factory` / `get_adapter`](https://github.com/vibrantlabsai/ragas/blob/298b68274234c060deacab3cf5fb52aa3a20e885/src/ragas/llms/base.py#L606) | `llms/base.py:606` | Entry point for LLM construction; selects adapter |
| [`InstructorAdapter.create_llm`](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/llms/adapters/instructor.py#L14) | `llms/adapters/instructor.py:14` | Builds the `InstructorLLM` wrapping the raw client |

---

## 4. Concurrent Requests and 504 / Rate-Limit Errors

RAGAS evaluation runs samples concurrently by design:

1. All sample tasks are created upfront.
2. `asyncio.as_completed(tasks)` processes them as they finish.
3. Each task calls `ascore(...)`, which itself may fire additional LLM requests.

This means multiple requests hit the same endpoint simultaneously. If your backend is behind nginx, an internal gateway, or a shared inference server, this quickly causes 504 timeouts, connection queuing, or 429 rate-limit errors.

**Recommended diagnosis steps:**

1. Set concurrency to 1 as a baseline — confirm the endpoint works at all.
2. Gradually increase to 2, then 4, and identify at which point errors begin.
3. Add a semaphore in your evaluator to cap in-flight requests (see Section 5).
4. Reduce `max_tokens` if it is set very high (e.g. 256000) — large token budgets increase per-request latency and worsen queuing.

---

## 5. Capping Concurrency with `asyncio.Semaphore`

An `asyncio.Semaphore` acts as a token bucket for coroutines. With `max_concurrency=K`, it ensures at most `K` LLM requests are in-flight at any moment:

```python
import asyncio

semaphore = asyncio.Semaphore(K)

async def evaluate_sample(sample):
    async with semaphore:          # acquire a token; suspend if none available
        return await ascore(sample)
```

**How it works:**

- `K` tokens are available at initialization.
- A coroutine reaching `async with semaphore` either acquires a token immediately (if available) or suspends (without blocking a thread) until another coroutine releases one.
- On exiting the `async with` block, the token is released automatically.
- The result: even if `arun()` creates hundreds of tasks, only `K` are ever executing their critical section simultaneously — flattening the burst load that causes 504s and 429s.

> **Important:** the semaphore only governs the code wrapped inside the `async with` block. Place the `ascore(...)` call inside the block to limit LLM concurrency specifically.

---

## 6. Silent Overwrite with the CSV / Local Backend

When using RAGAS's local CSV backend and calling `run_evaluation(experiment_name="same_name")` multiple times in a loop, each call's `arun()` triggers a full file write (`save()`), not an append. The last run silently overwrites all previous results.

To avoid data loss:
- Use a unique `experiment_name` per run (e.g., append a timestamp).
- Or collect all results in memory and write once after the loop completes.

```python
from datetime import datetime

name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_evaluation(..., experiment_name=name)
```

---

## Summary

| Problem | Root Cause | Fix |
|---|---|---|
| Errors silently swallowed | `raise_exceptions=False` in `Executor` | Set env var or monkey-patch the default |
| `track()` gives no error detail | Telemetry path, `@silent` decorator | Use `RAGAS_DEBUG=true`; look in executor logs instead |
| `<failed_attempts>` with `</html>` | Proxy returned HTML error page | Debug the HTTP gateway, not RAGAS |
| 504 / 429 under evaluation load | Default unbounded concurrency | Add `asyncio.Semaphore`, reduce `max_tokens` |
| Results overwritten silently | CSV backend does full-file writes | Use unique experiment names per run |