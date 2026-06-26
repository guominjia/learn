---
layout: post
title: "Understanding procps, top, and ps on Linux"
date: 2026-06-26
categories: [linux]
tags: [linux, process, procps, top, ps, procfs]
---

## Overview

On Linux, process management usually starts with three names:

- `procps`
- `top`
- `ps`

They are closely related, but they are not the same thing.

In simple terms:

- **`procps`** is a package/project that provides common process and system monitoring tools.
- **`ps`** prints a snapshot of current processes.
- **`top`** shows an interactive, continuously refreshed view of processes and system load.

Most Linux users learn `ps` and `top` first, but understanding `procps` explains where these tools come from and why their behavior is closely tied to the Linux `/proc` filesystem.

---

## What is `procps`?

`procps` is a collection of command-line utilities for inspecting and managing Linux processes and system state.

On many modern Linux distributions, the package is named **`procps`** or **`procps-ng`**. The `procps-ng` project is the actively maintained continuation used by many distributions.

Common tools provided by `procps` / `procps-ng` include:

| Tool | Purpose |
|---|---|
| `ps` | Show process status snapshot |
| `top` | Interactive real-time process viewer |
| `free` | Show memory usage |
| `vmstat` | Show virtual memory, CPU, and I/O statistics |
| `uptime` | Show system uptime and load average |
| `pgrep` | Find processes by name or attributes |
| `pkill` | Send signals to processes matched by name |
| `pidof` | Find process IDs by program name |
| `watch` | Repeatedly run a command and display output |
| `sysctl` | Read and write kernel parameters |

Depending on the distribution, some commands may come from related packages, but `procps` remains one of the core user-space packages for process inspection.

### Package names on common distributions

On Debian and Ubuntu:

```bash
dpkg -S /bin/ps
dpkg -S /usr/bin/top
apt show procps
```

Typical package name:

```text
procps
```

On Fedora, CentOS Stream, and RHEL-like systems:

```bash
rpm -qf /usr/bin/ps
rpm -qf /usr/bin/top
dnf info procps-ng
```

Typical package name:

```text
procps-ng
```

---

## Why `/proc` matters

The name `procps` comes from the Linux **process filesystem**, usually mounted at:

```text
/proc
```

`/proc` is not a normal disk directory. It is a virtual filesystem exposed by the Linux kernel. It provides live information about processes, memory, CPU, kernel parameters, mounts, and many other runtime details.

For example:

```bash
ls /proc
```

You will see many numeric directories:

```text
1
2
1234
5678
...
```

Each numeric directory is a process ID, or PID.

For a process with PID `1234`, useful files include:

| Path | Meaning |
|---|---|
| `/proc/1234/cmdline` | Command-line arguments |
| `/proc/1234/environ` | Environment variables |
| `/proc/1234/exe` | Symbolic link to executable |
| `/proc/1234/fd/` | Open file descriptors |
| `/proc/1234/status` | Human-readable process status |
| `/proc/1234/stat` | Compact process statistics |
| `/proc/1234/maps` | Memory mappings |

Tools such as `ps` and `top` read data from `/proc`, format it, sort it, and present it in a human-friendly way.

So the relationship is:

```text
Linux kernel -> /proc virtual filesystem -> procps tools -> user output
```

---

## What is `ps`?

`ps` means **process status**.

It prints a snapshot of processes at one moment in time. Unlike `top`, it does not continuously refresh by default.

The simplest command is:

```bash
ps
```

Typical output:

```text
		PID TTY          TIME CMD
	12345 pts/0    00:00:00 bash
	12380 pts/0    00:00:00 ps
```

This only shows processes associated with the current terminal session.

### Common `ps` commands

Show all processes in BSD style:

```bash
ps aux
```

The three BSD-style flags mean:

| Flag | Meaning |
|---|---|
| `a` | Show processes from all users that are attached to terminals |
| `u` | Use a user-oriented format with columns such as `USER`, `%CPU`, and `%MEM` |
| `x` | Also include processes without a controlling terminal, such as daemons |

Together, `ps aux` is a practical way to list almost everything running on the system.

If the command line is truncated, use a wider output:

```bash
ps auxww
```

Show all processes in Unix/POSIX style:

```bash
ps -ef
```

Show a process tree-like relationship:

```bash
ps -ejH
```

Another common tree view is:

```bash
ps auxf
```

The `f` flag adds ASCII tree markers so parent-child relationships are easier to see.

Show selected columns:

```bash
ps -eo pid,ppid,user,stat,pcpu,pmem,comm,args
```

Sort by CPU usage:

```bash
ps -eo pid,user,pcpu,pmem,comm,args --sort=-pcpu
```

Sort by memory usage:

```bash
ps -eo pid,user,pcpu,pmem,rss,vsz,comm,args --sort=-pmem
```

Find a process by command name:

```bash
ps -C nginx -o pid,ppid,user,stat,cmd
```

Or combine it with `grep`:

```bash
ps aux | grep nginx
```

However, for name-based matching, `pgrep` is usually cleaner:

```bash
pgrep -a nginx
```

---

## Why does `ps` have different option styles?

One confusing thing about `ps` is that it supports several option styles:

```bash
ps aux
ps -ef
ps -eo pid,cmd
```

This is historical.

`ps` carries compatibility with different Unix traditions:

- **Unix/POSIX style**: options use a leading dash, such as `ps -ef`.
- **BSD style**: options often do not use a leading dash, such as `ps aux`.
- **GNU long options**: options use `--`, such as `--sort`.

That is why these commands look inconsistent but are all valid on Linux.

Common examples:

| Command | Meaning |
|---|---|
| `ps` | Processes in current shell session |
| `ps aux` | All processes with user-oriented columns |
| `ps -ef` | All processes with full-format listing |
| `ps -eo ...` | All processes with custom output format |

For scripts, `ps -eo ...` is often better because the output columns are explicit.

---

## Important `ps` columns

`ps` can display many fields. These are some of the most useful ones:

| Column | Meaning |
|---|---|
| `PID` | Process ID |
| `PPID` | Parent process ID |
| `USER` | Effective user owning the process |
| `%CPU` / `PCPU` | CPU usage percentage |
| `%MEM` / `PMEM` | Memory usage percentage |
| `RSS` | Resident Set Size: physical memory currently used |
| `VSZ` | Virtual memory size |
| `STAT` | Process state flags |
| `TTY` | Controlling terminal |
| `START` | Process start time or start date |
| `TIME` | Accumulated CPU time |
| `COMMAND` / `CMD` / `ARGS` | Command name or full command line |

Example:

```bash
ps -eo pid,ppid,user,stat,pcpu,pmem,rss,vsz,args --sort=-rss | head
```

This is useful when looking for memory-heavy processes.

---

## Understanding process states

The `STAT` column in `ps` is important for troubleshooting.

Common process states include:

| State | Meaning |
|---|---|
| `R` | Running or runnable |
| `S` | Sleeping, interruptible |
| `D` | Uninterruptible sleep, often waiting for I/O |
| `T` | Stopped by job control or debugger |
| `Z` | Zombie process |
| `I` | Idle kernel thread |

Additional flags may appear after the main state:

| Flag | Meaning |
|---|---|
| `<` | High-priority process |
| `N` | Low-priority process |
| `L` | Pages locked into memory |
| `s` | Session leader |
| `l` | Multi-threaded process |
| `+` | Foreground process group |

For example:

```text
Ss
Sl
R+
Z
```

Meanings:

- `Ss`: sleeping session leader.
- `Sl`: sleeping multi-threaded process.
- `R+`: running in the foreground process group.
- `Z`: zombie process.

---

## What is `top`?

`top` is an interactive process monitor.

It repeatedly reads system and process information, then refreshes the screen. It is useful when you want to watch the machine while it is running.

Start it with:

```bash
top
```

Typical sections include:

```text
top - 10:30:00 up 5 days,  2:10,  2 users,  load average: 0.20, 0.35, 0.40
Tasks: 200 total,   1 running, 198 sleeping,   0 stopped,   1 zombie
%Cpu(s):  2.0 us,  1.0 sy,  0.0 ni, 97.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :  16000.0 total,   4000.0 free,   8000.0 used,   4000.0 buff/cache
MiB Swap:   2048.0 total,   2048.0 free,      0.0 used.   7000.0 avail Mem
```

Below that, `top` lists processes.

---

## Reading the `top` header

The first lines of `top` summarize the whole system.

### Uptime and load average

Example:

```text
load average: 0.20, 0.35, 0.40
```

These three numbers are the average runnable or uninterruptible workload over:

- 1 minute
- 5 minutes
- 15 minutes

Load average is not exactly CPU usage. It includes tasks waiting for CPU and, on Linux, tasks in uninterruptible sleep.

As a rough rule:

- On a 1-core machine, load `1.00` means roughly fully loaded.
- On a 4-core machine, load `4.00` means roughly fully loaded.
- On a 16-core machine, load `16.00` means roughly fully loaded.

So always compare load average with CPU count.

### Task summary

Example:

```text
Tasks: 200 total, 1 running, 198 sleeping, 0 stopped, 1 zombie
```

This line summarizes process states.

If the number of zombies is increasing, check parent processes that are not reaping child exit status.

### CPU line

Example:

```text
%Cpu(s):  2.0 us,  1.0 sy,  0.0 ni, 97.0 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
```

Common fields:

| Field | Meaning |
|---|---|
| `us` | User-space CPU time |
| `sy` | Kernel-space CPU time |
| `ni` | CPU time for niced processes |
| `id` | Idle CPU time |
| `wa` | Waiting for I/O |
| `hi` | Hardware interrupt time |
| `si` | Software interrupt time |
| `st` | Stolen time, common in virtual machines |

High `us` usually means application CPU work.

High `sy` may indicate heavy kernel activity, system calls, networking, storage, or container overhead.

High `wa` often points to storage or filesystem latency.

High `st` means the virtual machine wants CPU time but the hypervisor is not giving it enough.

### Memory lines

`top` shows memory and swap usage.

Linux memory output can be confusing because `buff/cache` is not simply wasted memory. The kernel uses available memory for page cache and buffers to improve performance. This memory can usually be reclaimed when applications need it.

The most useful field is often:

```text
avail Mem
```

It estimates how much memory can be used by new applications without heavy swapping.

---

## Useful interactive keys in `top`

After starting `top`, you can press keys to change the view.

| Key | Action |
|---|---|
| `q` | Quit |
| `h` | Help |
| `P` | Sort by CPU usage |
| `M` | Sort by memory usage |
| `T` | Sort by accumulated CPU time |
| `N` | Sort by PID |
| `k` | Kill a process by PID |
| `r` | Renice a process |
| `u` | Show processes for one user |
| `c` | Toggle command name / full command line |
| `1` | Toggle per-CPU display |
| `H` | Toggle thread display |
| `V` | Toggle forest view, if supported |
| `W` | Write configuration file |

Common workflow:

1. Run `top`.
2. Press `P` to sort by CPU.
3. Press `M` to sort by memory.
4. Press `c` to see full command lines.
5. Press `1` to inspect individual CPU cores.
6. Press `q` to exit.

---

## Useful `top` command-line options

Run `top` in batch mode:

```bash
top -b -n 1
```

This prints one snapshot and exits. It is useful for scripts and logs.

Show only processes for a user:

```bash
top -u alice
```

Monitor specific PIDs:

```bash
top -p 1234,5678
```

Set refresh delay to 1 second:

```bash
top -d 1
```

Show threads:

```bash
top -H
```

Batch mode with a few iterations:

```bash
top -b -d 2 -n 5
```

This runs five refreshes with a two-second delay.

---

## `ps` vs `top`

Both `ps` and `top` show processes, but they are optimized for different tasks.

| Feature | `ps` | `top` |
|---|---|---|
| View type | Snapshot | Live refreshing view |
| Interaction | Non-interactive | Interactive |
| Scripting | Very suitable | Use batch mode if needed |
| Sorting | Via options such as `--sort` | Via keys or config |
| Best use | Precise command output | Real-time observation |
| Typical command | `ps -ef` | `top` |

Use `ps` when you want a stable one-time answer:

```bash
ps -eo pid,ppid,user,stat,cmd
```

Use `top` when you want to observe change over time:

```bash
top
```

---

## Practical troubleshooting examples

### 1) Find the highest CPU consumers

With `ps`:

```bash
ps -eo pid,user,pcpu,pmem,stat,comm,args --sort=-pcpu | head -20
```

With `top`:

```bash
top
```

Then press `P`.

### 2) Find the highest memory consumers

```bash
ps -eo pid,user,pmem,rss,vsz,stat,comm,args --sort=-rss | head -20
```

Or in `top`, press `M`.

### 3) Inspect one process

```bash
ps -p 1234 -o pid,ppid,user,stat,pcpu,pmem,rss,vsz,start,time,args
```

Useful `/proc` paths:

```bash
cat /proc/1234/status
readlink /proc/1234/exe
ls -l /proc/1234/fd
```

### 4) Find child processes

```bash
ps --ppid 1234 -o pid,ppid,user,stat,args
```

Or show a process tree:

```bash
ps -ejH
```

Another common tool is:

```bash
pstree -p
```

Depending on the distribution, `pstree` may come from a different package such as `psmisc`.

Useful `pstree` options:

| Option | Meaning |
|---|---|
| `-p` | Show PIDs |
| `-s` | Show parent chain, or ancestors, of a process |
| `-a` | Show command-line arguments |

For example:

```bash
pstree -sp 1234
pstree -ap 1234
```

These commands help trace a chain such as:

```text
systemd -> sshd -> bash -> python3
```

You can also inspect the parent PID directly from `/proc`:

```bash
grep '^PPid:' /proc/1234/status
```

### 5) Check for zombie processes

```bash
ps -eo pid,ppid,stat,cmd | awk '$3 ~ /Z/ { print }'
```

A zombie process has exited, but its parent has not collected its exit status. Zombies do not consume normal CPU or memory like running processes, but many zombies may indicate a bug in the parent process.

### 6) Check whether a process is stuck in I/O wait

Processes in `D` state are in uninterruptible sleep:

```bash
ps -eo pid,ppid,stat,wchan,cmd | awk '$3 ~ /^D/ { print }'
```

This often points to disk, network filesystem, driver, or kernel-level waiting.

---

## Signals, `kill`, and process control

Although `ps` and `top` are mainly inspection tools, they are often used before sending signals.

Find a process:

```bash
ps -ef | grep nginx
```

Send a graceful termination signal:

```bash
kill 1234
```

This sends `SIGTERM` by default.

Force kill only when necessary:

```bash
kill -9 1234
```

This sends `SIGKILL`, which cannot be caught or ignored by the process. It should not be the first choice because the process cannot clean up resources gracefully.

With `top`, press `k`, enter the PID, and choose the signal.

For name-based process control, `procps` also provides:

```bash
pgrep -a nginx
pkill nginx
```

Be careful with `pkill`, especially on shared systems.

---

## Common mistakes

### Mistake 1: Treating load average as CPU percentage

Load average is not CPU percentage. It is a queue-like measure of runnable and uninterruptible tasks. Compare it with CPU core count.

### Mistake 2: Thinking cached memory is wasted memory

Linux uses free memory for cache. Look at available memory, swap activity, and actual application pressure before concluding that memory is exhausted.

### Mistake 3: Using `grep` carelessly

This command often matches itself:

```bash
ps aux | grep nginx
```

Better alternatives:

```bash
pgrep -a nginx
ps -C nginx -o pid,user,stat,args
```

### Mistake 4: Always using `kill -9`

Use normal `kill` first. Reserve `kill -9` for processes that do not respond to graceful termination.

### Mistake 5: Comparing `%CPU` without understanding threads

On multi-core systems, CPU percentage may exceed 100% for multi-threaded processes depending on the tool and display mode. Use `top -H` or press `H` in `top` to inspect threads.

---

## Recommended daily commands

For quick process overview:

```bash
ps aux | head
```

For full command lines without truncation:

```bash
ps auxww
```

For all processes with explicit columns:

```bash
ps -eo pid,ppid,user,stat,pcpu,pmem,args
```

For top CPU usage:

```bash
ps -eo pid,user,pcpu,pmem,args --sort=-pcpu | head
```

For top memory usage:

```bash
ps -eo pid,user,pmem,rss,args --sort=-rss | head
```

For live monitoring:

```bash
top
```

For one-time `top` output:

```bash
top -b -n 1
```

For finding processes by name:

```bash
pgrep -a sshd
```

For tracing the parents of a process:

```bash
pstree -sp 1234
```

For checking the parent PID through procfs:

```bash
grep '^PPid:' /proc/1234/status
```

---

## Summary

`procps` is a foundational Linux user-space package for process and system inspection. Its tools read information exposed by the kernel through `/proc` and present it in practical command-line formats.

The two most important tools are:

- `ps`: best for one-time, scriptable process snapshots.
- `top`: best for interactive, real-time process monitoring.

When troubleshooting Linux systems, a good mental model is:

```text
/proc contains live kernel data
ps formats a snapshot
top refreshes the view continuously
```

Mastering `ps`, `top`, and the basic structure of `/proc` gives you a strong foundation for debugging CPU load, memory pressure, stuck processes, zombie processes, and general system behavior.
