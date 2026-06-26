---
layout: post
title: "Understanding GNU Coreutils User and Group Commands"
date: 2026-06-26
categories: [linux]
tags: [linux, coreutils, users, groups, id, whoami, logname]
---

## Overview

GNU Coreutils is one of the most basic user-space packages on a Linux system. It provides many commands people use every day, including `ls`, `cp`, `mv`, `cat`, `chmod`, and `date`.

It also provides a small but useful set of commands for checking user and group identity:

- `whoami`
- `id`
- `groups`
- `users`
- `logname`
- `who`

These commands look simple, but they answer different questions. Confusing them can lead to wrong assumptions when debugging login sessions, permissions, SSH behavior, containers, or automation scripts.

In short:

| Command | Main question |
|---|---|
| `whoami` | Which effective user am I running as? |
| `id` | What are my UID, GID, and group memberships? |
| `groups` | Which groups does a user belong to? |
| `users` | Which user names are currently logged in? |
| `logname` | What was the original login name? |
| `who` | Which login sessions are recorded on the system? |

---

## What is GNU Coreutils?

GNU Coreutils is a collection of fundamental command-line utilities used by GNU/Linux systems and many Unix-like environments.

On Debian and Ubuntu, the package is usually named:

```text
coreutils
```

You can check which package owns one of these commands with `dpkg`:

```bash
dpkg -S /usr/bin/id
dpkg -S /usr/bin/whoami
dpkg -S /usr/bin/groups
```

Typical output on Ubuntu or Debian:

```text
coreutils: /usr/bin/id
coreutils: /usr/bin/whoami
coreutils: /usr/bin/groups
```

On Fedora, RHEL, CentOS Stream, and similar distributions, use `rpm`:

```bash
rpm -qf /usr/bin/id
rpm -qf /usr/bin/whoami
rpm -qf /usr/bin/groups
```

Typical output:

```text
coreutils-9.x-...
```

The exact path can vary between distributions, especially on systems using a merged `/usr` layout where `/bin` may be a symbolic link to `/usr/bin`.

---

## Identity is not one single thing

Before comparing the commands, it is important to understand that Linux has several related but different identity concepts.

### User account

A user account is usually defined in:

```text
/etc/passwd
```

Example entry:

```text
alice:x:1000:1000:Alice:/home/alice:/bin/bash
```

Important fields include:

| Field | Meaning |
|---|---|
| `alice` | user name |
| `1000` | user ID, or UID |
| `1000` | primary group ID, or GID |
| `/home/alice` | home directory |
| `/bin/bash` | login shell |

### Group account

Group information is usually defined in:

```text
/etc/group
```

Example entries:

```text
alice:x:1000:
sudo:x:27:alice
docker:x:999:alice
```

A user has one primary group and may have many supplementary groups.

### Real user ID and effective user ID

Processes have identity too. A process can have a real user ID and an effective user ID.

The effective user ID is especially important for permission checks. For example, when using `sudo`, a command may run with effective UID `0`, which is the `root` user.

That is why this command:

```bash
whoami
```

may print:

```text
root
```

inside a `sudo` command, even if the original login user was `alice`.

### Login session

A login session is a record that says a user is logged in through a terminal, SSH connection, graphical session, or similar login path.

Commands such as `users` and `who` are about login sessions. They are not simply reading all accounts from `/etc/passwd`.

This distinction explains many confusing cases.

---

## `whoami`: show the effective user name

The simplest identity command is:

```bash
whoami
```

Example output:

```text
alice
```

`whoami` prints the user name associated with the current process's effective user ID.

It is roughly similar to:

```bash
id -un
```

### Example with `sudo`

Run:

```bash
whoami
sudo whoami
```

Example output:

```text
alice
root
```

This does not mean the login session changed from `alice` to `root`. It means the second command ran with the effective identity of `root`.

Use `whoami` when you want to answer:

> Which user is this process effectively running as?

---

## `id`: show UID, GID, and groups

The `id` command gives a more complete view of user and group identity:

```bash
id
```

Example output:

```text
uid=1000(alice) gid=1000(alice) groups=1000(alice),27(sudo),999(docker)
```

This tells you:

- the current user ID
- the primary group ID
- supplementary group IDs
- symbolic names for those IDs, if available

Useful forms include:

```bash
id -u
id -un
id -g
id -gn
id -G
id -Gn
```

| Command | Meaning |
|---|---|
| `id -u` | numeric user ID |
| `id -un` | user name |
| `id -g` | numeric primary group ID |
| `id -gn` | primary group name |
| `id -G` | all group IDs |
| `id -Gn` | all group names |

Example:

```bash
id -Gn
```

Output:

```text
alice sudo docker
```

Use `id` when debugging permissions. It is usually more useful than `whoami` because file access often depends on both user ID and group membership.

---

## `groups`: show group names

The `groups` command prints group memberships.

For the current user:

```bash
groups
```

Example output:

```text
alice sudo docker
```

For another user:

```bash
groups alice
```

Example output:

```text
alice : alice sudo docker
```

In many day-to-day cases, `groups` and `id -Gn` appear to give similar information. However, `id` is more flexible and can show numeric IDs too.

### Common use cases

Check whether the current user is in the `docker` group:

```bash
groups
```

or:

```bash
id -nG
```

Check a specific account:

```bash
groups bob
```

If a newly added group does not appear in the current shell, the user may need to log out and log back in. Group membership is normally loaded when the login session starts.

---

## `users`: show logged-in user names

The `users` command prints user names of currently logged-in users:

```bash
users
```

Example output:

```text
alice alice bob
```

This output does not mean there are two `alice` accounts. It usually means there are two login session records for `alice`.

For example, `alice` may have:

- one SSH terminal on `pts/0`
- another SSH terminal on `pts/1`
- a graphical desktop session
- a stale or disconnected session record
- an editor or remote development tool that opened another session

`users` is intentionally compact. It does not show terminals, login times, source addresses, or process IDs.

When `users` output is confusing, use `who` instead.

---

## `who`: show login session details

The `who` command shows more detailed login session records:

```bash
who
```

Example output:

```text
alice    pts/0        2026-06-26 09:30 (192.0.2.10)
alice    pts/1        2026-06-26 09:35 (192.0.2.10)
bob      pts/2        2026-06-26 10:00 (192.0.2.20)
```

This explains why `users` printed:

```text
alice alice bob
```

Useful options include:

```bash
who -u
who -a
who am i
```

`who -u` can show idle time and the process ID associated with a session:

```text
alice    pts/0        2026-06-26 09:30   .          12345 (192.0.2.10)
```

That PID can then be checked with:

```bash
ps -fp 12345
```

Use `who` when you want to answer:

> Which login sessions exist on this machine?

---

## `logname`: show the login name

The `logname` command prints the name of the user who logged in:

```bash
logname
```

Example output:

```text
alice
```

This is different from `whoami`.

For example:

```bash
whoami
logname
sudo whoami
sudo logname
```

Possible output:

```text
alice
alice
root
alice
```

Here, `sudo whoami` prints `root` because the command is effectively running as root. But `sudo logname` may still print `alice` because the original login session belongs to `alice`.

`logname` depends on login session information. It can fail in environments without a normal login record, such as some containers, cron jobs, CI jobs, or minimal service environments.

---

## Comparing common commands

The following examples show why these commands should not be treated as interchangeable.

### Normal shell

```bash
whoami
id -un
logname
groups
users
```

Possible output:

```text
alice
alice
alice
alice sudo docker
alice bob
```

Interpretation:

- the shell is running as `alice`
- the login user is `alice`
- `alice` belongs to `alice`, `sudo`, and `docker`
- the system currently has login sessions for `alice` and `bob`

### Inside `sudo`

```bash
sudo sh -c 'whoami; id; logname'
```

Possible output:

```text
root
uid=0(root) gid=0(root) groups=0(root)
alice
```

Interpretation:

- the command is running as `root`
- the effective UID is `0`
- the original login name may still be `alice`

### Inside a container

In a container, output may look different:

```bash
whoami
id
logname
users
```

Possible output:

```text
root
uid=0(root) gid=0(root) groups=0(root)
logname: no login name
```

Containers often start processes directly instead of creating traditional login sessions. Therefore, `who`, `users`, and `logname` may have little or no useful data.

---

## How these commands relate to system files

These commands use several sources of information.

| Source | Used for |
|---|---|
| `/etc/passwd` | user name, UID, home directory, shell |
| `/etc/group` | group names and supplementary group membership |
| `/etc/shadow` | password hashes and password aging, usually not read by these commands directly |
| `/run/utmp` or `/var/run/utmp` | active login session records |
| NSS | user and group lookup from files, LDAP, SSSD, systemd, or other sources |

NSS means Name Service Switch. It allows user and group information to come from more than local files.

For example, on an enterprise system, a user may not appear in `/etc/passwd`, but this command can still work:

```bash
id alice
```

because the account may come from LDAP, Active Directory, SSSD, or another identity provider.

---

## Practical debugging examples

### Check who the current process is

```bash
whoami
id
```

Use this when a script behaves differently from an interactive shell.

### Check whether a user can access a group-owned file

```bash
id alice
ls -l /path/to/file
```

If the file is owned by group `docker`, `www-data`, or another service group, verify that the user is actually a member of that group.

### Check whether a group change is active

After adding a user to a group:

```bash
sudo usermod -aG docker alice
```

the existing shell may still show old group membership:

```bash
id
```

The user usually needs a new login session. After logging out and back in:

```bash
id -nG
```

should include the new group.

### Check duplicate names from `users`

If this appears:

```bash
users
```

```text
alice alice
```

run:

```bash
who -u
```

This usually reveals multiple terminals, SSH sessions, or graphical sessions.

---

## Common misunderstandings


### `users` does not list all system accounts

To list local account names, use:

```bash
cut -d: -f1 /etc/passwd
```

To query a specific account through NSS, use:

```bash
getent passwd alice
```

`users` only prints names from current login session records.

### `groups` does not prove the current shell has refreshed membership

If you run:

```bash
groups alice
```

you are asking about the account database view for `alice`.

If you run:

```bash
id
```

you are checking the current process identity.

After group changes, these may temporarily differ until a new login session starts.

### `whoami` is not always the original human user

Under `sudo`, `su`, setuid programs, or service managers, `whoami` reports the effective process user, not necessarily the human who first logged in.

Use `logname`, `who am i`, or environment variables such as `SUDO_USER` only when they match the trust model of your script.

---

## Script recommendations

For scripts, prefer explicit checks.

To require root privileges:

```bash
if [ "$(id -u)" -ne 0 ]; then
	echo "This script must run as root" >&2
	exit 1
fi
```

To get the current effective user name:

```bash
id -un
```

To check whether the current process has a group:

```bash
id -nG | grep -qw docker
```

To inspect login sessions, do not parse `users` if you need details. Use:

```bash
who
```

or:

```bash
who -u
```

---

## Summary

GNU Coreutils provides several commands related to users and groups, but each command answers a different question.

| Command | Best used for |
|---|---|
| `whoami` | current effective user name |
| `id` | UID, GID, and group membership |
| `groups` | group names for a user |
| `users` | compact list of logged-in user names |
| `who` | detailed login session records |
| `logname` | original login name |

The most important rule is:

```text
identity, group membership, and login sessions are related, but they are not the same thing
```

When debugging Linux permissions, start with `id`. When debugging logged-in sessions, start with `who`. When checking the effective process user, use `whoami` or `id -un`.
