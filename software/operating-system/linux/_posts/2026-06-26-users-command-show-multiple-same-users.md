---
layout: post
title: "Why the Linux users Command Shows the Same User Multiple Times"
date: 2026-06-26
categories: [linux]
tags: [linux, users, who, w, ssh, session, utmp]
---

## Overview

The Linux `users` command can sometimes print the same username more than once:

```bash
users
```

Example output:

```text
user_1 user_1 user_2 user_2 user_3 user_4 user_4 user_4 user_4
```

At first glance, this looks like duplicate user accounts. It is not.

The `users` command does **not** show the unique list of user accounts on the system. It shows the usernames of users currently recorded as logged in. If the same account has multiple login sessions, the same username appears multiple times.

In short:

```text
one printed name = one login session record
```

So if `user_1` appears twice, it usually means `user_1` has two login session records.

---

## What `users` really shows

The `users` command reads login session information from the system login database, commonly associated with files such as:

```text
/var/run/utmp
/run/utmp
```

These records are maintained by login-related programs such as `login`, `sshd`, terminal emulators, display managers, and related session managers.

This means `users` is about **sessions**, not about account definitions.

It is different from checking local accounts in:

```text
/etc/passwd
```

For example:

```bash
cut -d: -f1 /etc/passwd
```

prints account names known to the local system, while:

```bash
users
```

prints users who currently have login session records.

---

## Why duplicate names appear

Duplicate names usually appear for one of these reasons.

### 1. The same user opened multiple SSH connections

If `user_1` connects to the same server from two different terminals, the output may look like this:

```text
user_1 user_1
```

Each SSH login gets its own pseudo-terminal, such as `pts/0`, `pts/1`, or `pts/2`.

Use `who` to see this more clearly:

```bash
who
```

Example:

```text
user_1   pts/0        2026-06-26 09:30 (192.0.2.10)
user_1   pts/1        2026-06-26 09:35 (192.0.2.10)
```

This means there are two login sessions for the same account.

### 2. A graphical desktop, VNC, or remote desktop session exists

If the system runs a graphical desktop, VNC server, or remote desktop service, that environment may create additional login records.

For example:

```text
user_2   :1           2026-06-26 08:10
user_2   pts/3        2026-06-26 08:12 (:1)
```

Here `user_2` may have a graphical session and one terminal inside that graphical session.

### 3. VS Code Remote SSH or automation opened background sessions

Even if only one visible SSH shell is open, tools can create additional SSH sessions in the background.

Common examples include:

- VS Code Remote SSH
- SSH port forwarding
- file synchronization tools
- monitoring agents
- automation scripts
- Jupyter or development services started through SSH

From the user's perspective, there may be only one visible shell. From the system's perspective, there may be more than one login-related session.

### 4. A previous session was not cleaned up correctly

If an SSH connection was interrupted by a network problem, laptop sleep, terminal crash, or abrupt disconnection, the process may exit but the login record may temporarily remain.

This can make `users` show an old session that is no longer meaningful.

This is less common on healthy systems, but it can happen.

---

## Why there may be two entries when only one SSH shell is visible

A common question is:

> I only opened one SSH shell. Why does `users` show my username twice?

There are several likely explanations.

### Case 1: There is another hidden SSH session

Some clients open more than one SSH connection. For example, an editor or remote development tool may open one session for the visible terminal and another for background services.

Check with:

```bash
who -u
```

Example:

```text
user_1   pts/0        2026-06-26 10:00   .          12345 (192.0.2.10)
user_1   pts/1        2026-06-26 10:01   .          12388 (192.0.2.10)
```

The two `pts` entries indicate two pseudo-terminal sessions.

### Case 2: One entry is from a previous disconnected session

If a previous SSH connection did not close cleanly, it may leave a stale login record.

Use `who -u` and check the PID column:

```bash
who -u
```

Then verify whether the process still exists:

```bash
ps -fp 12345
```

If the PID does not exist, the login record may be stale.

### Case 3: A terminal multiplexer or remote tool is involved

Tools such as `tmux`, `screen`, VNC terminals, and remote IDE terminals can make the session structure less obvious.

One visible working environment may still correspond to multiple terminal records.

---

## Useful commands for investigation

### Show login sessions

```bash
who
```

This is usually the first command to run after seeing repeated names from `users`.

### Show login sessions with idle time and PID

```bash
who -u
```

The PID is useful because it can be checked with `ps`.

### Show active users and what they are running

```bash
w
```

Example output:

```text
USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT
user_1   pts/0    192.0.2.10       10:00    1:20   0.03s  0.03s -bash
user_1   pts/1    192.0.2.10       10:01    0.00s  0.05s  0.01s w
```

This shows both the session and the command currently associated with it.

### Show SSH daemon processes

```bash
ps -ef | grep '[s]shd'
```

This helps correlate SSH sessions with server-side `sshd` processes.

### Count sessions per user

```bash
users | tr ' ' '\n' | sort | uniq -c
```

Example:

```text
	2 user_1
	2 user_2
	1 user_3
	4 user_4
```

### Show unique logged-in users only

```bash
users | tr ' ' '\n' | sort -u
```

Example:

```text
user_1
user_2
user_3
user_4
```

---

## How to read the output correctly

Suppose `users` prints:

```text
user_1 user_1 user_2 user_2 user_3 user_4 user_4 user_4 user_4
```

This should be interpreted as:

| User | Meaning |
|---|---|
| `user_1` | 2 login session records |
| `user_2` | 2 login session records |
| `user_3` | 1 login session record |
| `user_4` | 4 login session records |

It does **not** mean there are duplicate accounts named `user_1` or `user_4`.

---

## Practical troubleshooting flow

When `users` shows repeated names, use this sequence:

### Step 1: Check detailed sessions

```bash
who -u
```

Look at:

- username
- terminal name, such as `pts/0`
- login time
- idle time
- PID
- remote host

### Step 2: Check what each session is doing

```bash
w
```

This helps determine whether the session is an active shell, an editor, a background remote tool, or an idle session.

### Step 3: Check related processes

For one PID shown by `who -u`:

```bash
ps -fp PID
```

Replace `PID` with the actual process ID.

To see a broader SSH process list:

```bash
ps -ef | grep '[s]shd'
```

### Step 4: Decide whether it is normal

It is usually normal if:

- the entries map to active SSH, VNC, desktop, or remote development sessions
- the remote host and login time make sense
- the processes still exist

It may require cleanup or investigation if:

- the PID no longer exists
- the session has been idle for a very long time
- the remote host is unexpected
- the user should not be logged in

---

## Can duplicate session records be removed?

Usually, you should not manually edit login databases such as `utmp`.

Instead, close the real session cleanly:

```bash
exit
```

or terminate a confirmed stale or unwanted session process carefully:

```bash
kill PID
```

If a session does not exit, a stronger signal can be used, but only after confirming the target:

```bash
kill -9 PID
```

Be careful when killing sessions on a shared machine. Terminating the wrong PID may disconnect another user's work.

---

## Key takeaway

Repeated names in `users` are normal when a user has multiple login session records.

The command answers this question:

```text
Which usernames are currently recorded as logged in?
```

It does not answer this question:

```text
Which unique user accounts exist on this system?
```

If you see the same user twice while only one SSH shell is visible, check `who -u` and `w`. The second entry is usually another SSH-related session, a remote development background connection, a graphical/VNC session, or a stale login record.
