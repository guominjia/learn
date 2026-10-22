# Why `sh -c "cd dir && pwd"` Prints the Wrong Directory

## The Problem

While working in a Linux environment, I ran into a puzzling behavior. My home directory is `/home/user`, and there is a `backend/` subdirectory under it. I expected the following command to print `/home/user/backend`:

```bash
sh -c "cd backend/ && pwd"
```

But it printed `/home/user` instead — as if `cd` never happened.

## Ruling Out Common Mistakes

### 1. Is `&&` Being Parsed by the Parent Shell?

A classic mistake is accidentally closing the quotes before `&&`:

```bash
# WRONG — pwd runs in the parent shell
sh -c "cd backend/" && pwd       # prints /home/user

# CORRECT — both commands run inside the child shell
sh -c "cd backend/ && pwd"       # should print /home/user/backend
```

I double-checked: my quotes were correct. Both `cd` and `pwd` were inside the quoted string passed to `sh -c`.

### 2. Does the Directory Exist?

Yes. `ls backend/` worked fine, and `cd backend/` alone produced no errors.

## The Clue

After some experimentation, I discovered that adding **any external command** before `cd && pwd` fixed the output:

```bash
sh -c 'ls backend && cd backend/ && pwd'
# Correctly prints /home/user/backend
```

Simply prepending `ls` (an external binary at `/bin/ls`) made `pwd` report the right directory. This pointed to something deeper than a typo.

## Root Cause: `dash` Builtin Optimization

On many Linux distributions (Debian, Ubuntu, etc.), `/bin/sh` is symlinked to **`dash`**, a lightweight POSIX shell. You can verify this:

```bash
ls -la /bin/sh
# lrwxrwxrwx 1 root root 4 ... /bin/sh -> dash
```

`dash` has an optimization for commands passed via `sh -c`: when the **entire command chain consists of only shell builtins** (`cd` and `pwd` are both builtins), `dash` takes a fast path that avoids forking a full subshell process. Unfortunately, this optimization path has a known issue where `cd` does not properly update the internal state that `pwd` reads from.

When an **external command** like `ls` is present in the chain, `dash` is forced to take the normal execution path, which correctly updates the working directory state after `cd`.

## Workarounds

Any of the following will produce the correct output:

```bash
# 1. Use bash instead of sh
bash -c 'cd backend/ && pwd'

# 2. Use the external pwd binary instead of the builtin
sh -c 'cd backend/ && /bin/pwd'

# 3. Insert any external command to break the all-builtin optimization
sh -c 'true && cd backend/ && pwd'   # if true is external
sh -c 'ls backend && cd backend/ && pwd'
```

## Takeaway

- On systems where `/bin/sh` is `dash`, be aware that `sh -c` with **pure-builtin command chains** may behave unexpectedly.
- When in doubt, use `bash -c` or call `/bin/pwd` explicitly.
- This is a subtle difference between `bash` and `dash` that can silently produce wrong results in scripts and CI pipelines.
