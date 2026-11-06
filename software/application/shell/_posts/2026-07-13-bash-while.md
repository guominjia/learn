---
layout: post
title: "Is `while true; do ... done &` an Infinite Loop?"
date: 2026-07-13
categories: [linux]
tags: [linux, bash, shell, process, docker, watchdog]
---

## Short answer

Yes, `while true; do ... done` is an **infinite loop**. In this case, however, the loop is intentional. It implements a simple process supervisor, or watchdog, that restarts a child process whenever it exits.

A typical script looks like this:

```bash
while true; do
	python task_executor.py &
	wait
	sleep 1
done &

# Continue with other initialization work here.

wait
```

This is not necessarily an accidental infinite-loop bug. It is a basic **crash-and-restart** pattern.

## How the loop works

The loop condition is the `true` command:

```bash
while true; do
	# Commands
done
```

In a shell, `true` always exits with status `0`, which means success. Because the `while` condition always succeeds, Bash repeatedly executes the loop body.

The loop stops only if something explicitly interrupts it, for example:

- A `break` statement is executed.
- The script calls `exit`.
- The shell receives a terminating signal.
- The process is killed.
- An unhandled shell error causes the script to exit under options such as `set -e`.

## What happens inside the loop?

### 1. Start the worker in the background

```bash
python task_executor.py &
```

The trailing `&` starts `task_executor.py` as a background job. The shell does not wait for it automatically and immediately moves to the next command.

### 2. Wait for the worker to exit

```bash
wait
```

`wait` blocks until the shell's active background jobs have exited. In this simple loop there is only one worker, so it effectively waits for `task_executor.py`.

The worker may exit normally, fail with an error, or crash. In each case, control eventually returns to the loop after `wait` finishes.

### 3. Avoid a rapid restart loop

```bash
sleep 1
```

The script pauses for one second before restarting the worker. This delay prevents a broken worker from being restarted thousands of times per second, which could consume excessive CPU and flood the logs.

### 4. Restart the worker

After `sleep` completes, execution returns to the top of the loop. Since `true` still succeeds, the script starts a new `task_executor.py` process.

The resulting lifecycle is:

```text
start worker
    -> wait for worker to exit
    -> sleep for one second
    -> start worker again
    -> repeat forever
```

## Why is there another `&` after `done`?

The final ampersand applies to the entire loop:

```bash
while true; do
	...
done &
```

It makes the complete supervisor loop a background job. Without this `&`, the shell would remain inside the loop and would never execute the initialization commands that follow it.

The two ampersands therefore have different scopes:

| Syntax | Effect |
|---|---|
| `python task_executor.py &` | Runs the worker in the background relative to the supervisor loop |
| `done &` | Runs the complete supervisor loop in the background relative to the main script |

This makes it possible to start several supervised workers and then continue with the rest of the entrypoint script.

## Why is there a final `wait`?

Container entrypoint scripts often end with an unqualified `wait`:

```bash
wait
```

This final `wait` is executed by the main shell. It waits for all background jobs started by that shell, including the supervisor loop.

Because the supervisor loop is intended to run forever, the final `wait` normally never finishes. As a result, the main script remains alive.

This matters in Docker because the container's lifecycle is tied to its PID 1 process. If the entrypoint script is PID 1 and exits, the container stops, even if some unexpected detached processes remain. Keeping the entrypoint alive with `wait` therefore keeps the container running.

The process relationship is approximately:

```text
entrypoint shell (PID 1)
    -> supervisor loop
        -> task_executor.py
```

## A clearer and safer version

When a shell manages more than one background job, a bare `wait` may wait for jobs other than the intended worker. Capturing `$!`, the PID of the most recently started background process, makes the relationship explicit:

```bash
supervise_task_executor() {
	while true; do
		python task_executor.py &
		worker_pid=$!

		wait "$worker_pid"
		exit_code=$?

		echo "task_executor.py exited with status $exit_code; restarting in 1 second" >&2
		sleep 1
	done
}

supervise_task_executor &
supervisor_pid=$!

# Perform other initialization work here.

wait "$supervisor_pid"
```

This version records the worker PID, waits for that specific process, and logs its exit status before restarting it.

If the script uses `set -e`, a nonzero result from `wait` may terminate the shell before the restart logic runs. Handle that result explicitly:

```bash
if wait "$worker_pid"; then
	exit_code=0
else
	exit_code=$?
fi
```

## Important limitations

Although this pattern is useful for small scripts, it is not a complete process-management system.

### Signal forwarding

Docker sends stop signals to PID 1. A shell waiting on background jobs may not automatically forward every signal to its child processes in the desired way. The script may need `trap` handlers that terminate the worker and wait for it to shut down.

### Restart policy

The simple loop restarts the worker after every exit, including a successful exit. Real supervisors can distinguish between success and failure, limit restart attempts, and use exponential backoff.

### Logging and health checks

The loop does not provide log rotation, readiness checks, dependency management, or health monitoring.

For production systems, consider a dedicated supervisor or an orchestration-level restart policy, such as Docker's `--restart` option, Docker Compose restart policies, Kubernetes controllers, `systemd`, or another process supervisor. When possible, running one main foreground process per container is usually simpler than implementing supervision in an entrypoint shell.

## Summary

`while true; do ... done &` is an infinite loop, but it can be an intentional one:

1. `while true` repeats forever.
2. The worker's `&` starts the worker as a background process.
3. `wait` pauses until the worker exits.
4. `sleep 1` prevents an aggressive crash-restart cycle.
5. `done &` runs the entire supervisor loop in the background.
6. A final `wait` keeps the main entrypoint process—and therefore the container—alive.

The pattern is a lightweight watchdog, not automatically a bug. Its purpose is to restart a failed process and keep the container's main script running, although production deployments should also account for signal handling, restart limits, and graceful shutdown.
