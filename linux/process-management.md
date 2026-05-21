# Linux Process Management: A Practical Guide to `ps` and Process Inspection

Understanding how to inspect and manage processes is a fundamental skill for any Linux user or system administrator. Whether you are debugging a runaway service, hunting down a memory hog, or simply curious about what is running on your machine, the `ps` command family has you covered.

This post walks through the `ps aux` output format, explains every column, and then dives into techniques for tracing parent-child process relationships.

---

## Understanding `ps aux` Output

`ps aux` is the go-to command for listing all running processes with detailed resource information. The three flags work together:

| Flag | Meaning |
|------|---------|
| `a`  | Show processes from all users |
| `u`  | Display user-oriented (readable) format |
| `x`  | Include processes not attached to a terminal |

### Column Breakdown

A typical `ps aux` header looks like this:

```
USER       PID  %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
```

| Column | Description |
|--------|-------------|
| **USER** | The owner of the process (the user account running it). |
| **PID** | **Process ID** — a unique integer identifying the process. |
| **%CPU** | CPU usage as a percentage (averaged over recent time). |
| **%MEM** | Memory usage as a percentage of total physical memory. |
| **VSZ** | **Virtual Memory Size** (KB) — includes all memory the process can access: physical RAM, swap, shared libraries, and mapped files. |
| **RSS** | **Resident Set Size** (KB) — actual physical memory currently in use. Does *not* include swap or shared libraries counted elsewhere. |
| **TTY** | The terminal the process is attached to. `?` means no terminal (typical for daemons and background services). |
| **STAT** | Process state code (see details below). |
| **START** | Time the process was started. Shows `HH:MM` if today, or a date if older than 24 hours. |
| **TIME** | Cumulative CPU time consumed (format: `MM:SS`). |
| **COMMAND** | The full command line that launched the process. If truncated, use `ps auxww` to see the complete string. |

### Decoding the `STAT` Column

The `STAT` column uses single-character codes, often combined:

| Code | Meaning |
|------|---------|
| `R` | **Running** — actively executing or in the run queue. |
| `S` | **Sleeping** (interruptible) — waiting for an event such as I/O completion. |
| `D` | **Uninterruptible sleep** — usually waiting on disk I/O. Cannot be killed until the I/O completes. |
| `Z` | **Zombie** — process has terminated but its parent has not yet read its exit status. |
| `T` | **Stopped** — suspended, e.g. via `Ctrl+Z` or a `SIGSTOP` signal. |
| `<` | High-priority (not nice to other processes). |
| `N` | Low-priority (nice to other processes). |
| `s` | Session leader. |
| `+` | In the foreground process group. |

### Example Output

```
root          1  0.0  0.0   2892   988 ?        Ss   Jan01   0:00 /sbin/init
alice      1234  1.5  2.3 1023456 23800 tty2     Sl   10:00   3:25 /usr/bin/python3 app.py
```

Reading the second line: user `alice` owns PID `1234`, which is consuming 1.5% CPU and 2.3% memory. It is in state `Sl` (sleeping + multi-threaded), attached to `tty2`, and the command is a Python application.

### Common Use Cases

| Scenario | What to Look For |
|----------|-----------------|
| **High resource usage** | Sort by `%CPU` or `%MEM` to find greedy processes. |
| **Zombie processes** | Filter for `Z` in the `STAT` column: `ps aux \| grep ' Z'`. |
| **Service health check** | Confirm a daemon appears in the list and is not in `T` or `Z` state. |

> **Tip:** For a real-time, interactive view, consider `top`, `htop`, or `btop`. For systemd-managed services, use `systemctl status <service>`.

---

## Tracing Parent Processes

Every process (except PID 1) has a **parent process** — the process that spawned it. Knowing the parent is essential when you need to trace how a process was launched or debug unexpected process trees.

### Method 1: `ps -f` — Full-Format Listing

```bash
ps -f -p 1234
```

Output:

```
UID        PID  PPID  C STIME TTY      TIME CMD
user      1234   567  0 10:00 ?        00:01:00 /usr/bin/python3 app.py
```

The **PPID** column (567 here) is the parent process ID. You can also list custom columns for all processes:

```bash
ps -eo pid,ppid,user,command
```

- `-e` selects every process.
- `-o` lets you pick exactly which columns to display.

### Method 2: `ps auxf` — ASCII Process Tree

```bash
ps auxf
```

The `f` flag renders an ASCII art tree showing parent-child relationships:

```
USER       PID  PPID %CPU %MEM    VSZ   RSS COMMAND
root         2     0  0.0  0.0      0     0 [kthreadd]
root       567     1  0.0  0.2 123456 8900  \_ /usr/bin/some_daemon
user      1234   567  5.0  3.1 234567 20000    \_ /usr/bin/python3 app.py
```

The indentation and `\_` markers make it easy to see that `python3` (1234) was spawned by `some_daemon` (567), which itself was spawned by `init`/`systemd` (1).

### Method 3: `pstree` — Dedicated Tree Viewer

```bash
# Show tree rooted at a specific PID
pstree -p 1234

# Show the entire system tree
pstree -p
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `-p` | Show PIDs next to process names. |
| `-s` | Show ancestors (parents) of the target process. |
| `-a` | Show command-line arguments. |

Example output:

```
systemd(1)───kthreadd(2)
            ├─sshd(567)───bash(1234)───python3(12345)
            └─...
```

Reading right to left: `python3(12345)` was launched from `bash(1234)`, which was launched from `sshd(567)`, ultimately rooted at `systemd(1)`.

### Method 4: Reading `/proc` Directly

The `/proc` virtual filesystem exposes kernel data for every running process:

```bash
cat /proc/1234/status | grep PPid
```

Output:

```
PPid:   567
```

This is the most direct way to query a single process's parent and works even when other utilities are unavailable.

---

## Quick Reference

| Task | Command |
|------|---------|
| List all processes with details | `ps aux` |
| Show full command (no truncation) | `ps auxww` |
| Show parent PID for one process | `ps -f -p <PID>` |
| Custom columns for all processes | `ps -eo pid,ppid,user,%cpu,%mem,command` |
| ASCII process tree | `ps auxf` |
| Dedicated tree view | `pstree -p` |
| Ancestors of a specific process | `pstree -sp <PID>` |
| Parent PID via procfs | `cat /proc/<PID>/status \| grep PPid` |

By combining these tools, you can quickly trace the full call chain of any process — for example, `systemd` → `sshd` → `bash` → `python3` — and gain a clear picture of what is happening on your system.
