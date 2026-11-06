---
layout: post
title: "Understanding ~/.bashrc, ~/.profile, /etc/bash.bashrc, and /etc/environment"
date: 2026-07-02
categories: [linux]
tags: [linux, bash, shell, environment, profile, ubuntu, debian]
---

## Overview

Linux has several files that look similar but are used for different purposes:

```text
~/.bashrc
~/.profile
/etc/bash.bashrc
/etc/environment
```

They are all related to login sessions, shells, and environment variables, but they are not interchangeable.

The most important difference is this:

- `~/.bashrc` is mainly for one user's interactive Bash shell.
- `~/.profile` is mainly for one user's login session.
- `/etc/bash.bashrc` is mainly for all users' interactive Bash shells.
- `/etc/environment` is mainly for system-wide environment variables, and it is not a shell script.

---

## Shell sessions: login, non-login, interactive, non-interactive

Before comparing the files, it helps to understand four common shell concepts.

### Login shell

A login shell is started when a user logs in to the system.

Examples:

- Logging in from a Linux text console.
- Logging in through SSH.
- Starting a terminal emulator configured to run a login shell.

The word "login" here is a session concept. It does not always mean the shell was started by the `/bin/login` program.

`/bin/login` is the traditional login program used by text-console login flows. A simplified console login flow looks like this:

```text
getty -> /bin/login -> PAM authentication -> user's shell
```

SSH login is different. It is usually handled by `sshd`, often together with PAM:

```text
sshd -> PAM authentication/session setup -> user's shell
```

Even if SSH does not directly use `/bin/login`, it can still start the user's shell as a login shell.

For Bash, the key point is not whether `/bin/login` was involved. The key point is whether Bash was started in login-shell mode. One common sign is that the shell name starts with `-`, such as:

```text
-bash
```

You can check it inside Bash with:

```bash
shopt -q login_shell && echo "login shell" || echo "non-login shell"
```

For Bash, a login shell reads files such as:

```text
/etc/profile
~/.bash_profile
~/.bash_login
~/.profile
```

Bash reads the first existing user profile file from this list:

```text
~/.bash_profile
~/.bash_login
~/.profile
```

On many Ubuntu and Debian systems, users usually have `~/.profile`, not `~/.bash_profile`.

### Non-login shell

A non-login shell is started after the user has already logged in.

Examples:

- Opening a normal graphical terminal window.
- Running `bash` from an existing shell.
- Running a shell inside a terminal multiplexer such as `tmux` or `screen`.

For interactive non-login Bash shells, Bash usually reads:

```text
~/.bashrc
```

### Interactive shell

An interactive shell accepts commands from a user.

Examples:

```bash
echo hello
cd /tmp
history
```

Interactive shells are where aliases, prompts, command completion, and history settings matter.

### Non-interactive shell

A non-interactive shell runs a script or command without direct user interaction.

Examples:

```bash
bash script.sh
bash -c 'echo hello'
```

Non-interactive shells usually should not load interactive customizations such as aliases, colorful prompts, or commands that print text automatically.

---

## `~/.bashrc`

`~/.bashrc` is a per-user Bash startup file.

It is normally read by interactive non-login Bash shells.

For example, when you open a new terminal window on Ubuntu, that shell commonly reads `~/.bashrc`.

Typical content includes:

```bash
# Aliases
alias ll='ls -alF'
alias grep='grep --color=auto'

# Prompt
PS1='\u@\h:\w\$ '

# Bash history
HISTSIZE=10000
HISTFILESIZE=20000

# Shell options
shopt -s histappend
```

### What belongs in `~/.bashrc`

Good candidates:

- Aliases.
- Bash functions.
- Prompt configuration such as `PS1`.
- Command completion setup.
- Shell history settings.
- Interactive-only tools such as `fzf`, `direnv`, or shell prompt frameworks.

Example:

```bash
alias gs='git status'
alias gd='git diff'

mkcd() {
	mkdir -p "$1" && cd "$1"
}
```

### What should usually not go in `~/.bashrc`

Avoid putting noisy or long-running commands in `~/.bashrc`.

Bad examples:

```bash
echo "Welcome"
sudo service something start
python long_startup_script.py
```

These commands may run every time a new interactive Bash shell starts, which can make terminals slow or break automation.

### Interactive guard

Many default `~/.bashrc` files contain this pattern near the top:

```bash
case $- in
	*i*) ;;
	*) return;;
esac
```

`$-` contains current shell options. If it contains `i`, the shell is interactive.

This guard prevents interactive-only configuration from running in non-interactive contexts.

---

## `~/.profile`

`~/.profile` is a per-user login startup file.

It is read when the user starts a login shell, as long as Bash does not find `~/.bash_profile` or `~/.bash_login` first.

It is more general than `~/.bashrc`. Historically, `~/.profile` is used by POSIX-compatible shells, not only Bash.

Typical content includes:

```sh
# Add user-local executables to PATH
if [ -d "$HOME/bin" ]; then
	PATH="$HOME/bin:$PATH"
fi

if [ -d "$HOME/.local/bin" ]; then
	PATH="$HOME/.local/bin:$PATH"
fi

export PATH
```

### What belongs in `~/.profile`

Good candidates:

- User-level environment variables needed by the whole login session.
- User-level `PATH` changes.
- Commands that should run once when the user logs in.
- POSIX-compatible shell code.

Example:

```sh
export EDITOR=vim
export LANG=en_US.UTF-8
export PATH="$HOME/.local/bin:$PATH"
```

### Why `~/.profile` often loads `~/.bashrc`

On Ubuntu, the default `~/.profile` often contains logic like this:

```sh
if [ -n "$BASH_VERSION" ]; then
	if [ -f "$HOME/.bashrc" ]; then
		. "$HOME/.bashrc"
	fi
fi
```

This means:

- If the login shell is Bash,
- and `~/.bashrc` exists,
- load `~/.bashrc` too.

This is why SSH login sessions often get the same aliases and prompt settings as normal terminal windows.

However, this behavior comes from distribution defaults. It is not because Bash always reads `~/.bashrc` for login shells by itself.

---

## `/etc/bash.bashrc`

`/etc/bash.bashrc` is a system-wide Bash startup file.

It is commonly used on Debian and Ubuntu systems for interactive Bash shells of all users.

In simple terms:

```text
~/.bashrc        -> one user's interactive Bash configuration
/etc/bash.bashrc -> all users' interactive Bash configuration
```

Typical system-wide settings include:

- Default prompt behavior.
- System-wide command-not-found handlers.
- Global Bash completion hooks.
- Common aliases or functions for all users.

Example:

```bash
# System-wide alias for all interactive Bash users
alias grep='grep --color=auto'
```

### Be careful with `/etc/bash.bashrc`

Because `/etc/bash.bashrc` affects all users, mistakes can have a wide impact.

Avoid:

- User-specific paths.
- Commands that assume a particular home directory.
- Commands that print unnecessary output.
- Slow network or package-manager operations.
- Changes that may break root or service accounts.

For personal customization, prefer `~/.bashrc`.

For organization-wide shell policy, `/etc/bash.bashrc` may be appropriate.

---

## `/etc/environment`

`/etc/environment` is a system-wide environment variable file.

It is different from the other files because it is not a shell script.

It is usually read by PAM modules such as `pam_env` during login session setup.

The syntax is simple key-value assignments:

```text
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LANG="en_US.UTF-8"
JAVA_HOME="/usr/lib/jvm/default-java"
```

### Important limitation

Do not write shell syntax in `/etc/environment`.

This is wrong:

```bash
export PATH="$HOME/bin:$PATH"
if [ -d "$HOME/bin" ]; then
	PATH="$HOME/bin:$PATH"
fi
alias ll='ls -alF'
```

`/etc/environment` does not understand `export`, `if`, shell variables, command substitution, aliases, or functions.

This is usually correct:

```text
FOO="bar"
MY_APP_HOME="/opt/my-app"
```

### What belongs in `/etc/environment`

Good candidates:

- Simple system-wide environment variables.
- Locale settings.
- Tool installation paths that should apply to all login sessions.

Use it only when the variable is truly system-wide.

For one user, prefer `~/.profile`.

For interactive Bash behavior, prefer `~/.bashrc` or `/etc/bash.bashrc`.

---

## Quick comparison

| File | Scope | Read by | Shell script? | Common use |
| --- | --- | --- | --- | --- |
| `~/.bashrc` | Current user | Interactive non-login Bash shells | Yes, Bash syntax | Aliases, functions, prompt, completion |
| `~/.profile` | Current user | Login shells, if no Bash-specific profile file is used | Yes, usually POSIX shell syntax | User environment variables and `PATH` |
| `/etc/bash.bashrc` | All users | Interactive Bash shells on Debian/Ubuntu | Yes, Bash syntax | Global Bash interactive settings |
| `/etc/environment` | All users | Login session environment via PAM on many systems | No | Simple global environment variables |

---

## Common startup flows

### Opening a normal graphical terminal

On Ubuntu, a normal terminal window usually starts an interactive non-login Bash shell.

Typical flow:

```text
Terminal starts Bash
  -> Bash reads /etc/bash.bashrc
  -> Bash reads ~/.bashrc
```

This is why aliases and prompt settings usually belong in `~/.bashrc`.

### Logging in through SSH

SSH commonly starts a login shell.

Typical Bash flow:

```text
SSH login
  -> login environment is prepared
  -> /etc/environment may be read by PAM
  -> Bash reads /etc/profile
  -> Bash reads the first existing file among:
       ~/.bash_profile
       ~/.bash_login
       ~/.profile
```

If `~/.profile` sources `~/.bashrc`, then Bash interactive settings are loaded too.

### Running a shell script

When running a script:

```bash
bash script.sh
```

Bash usually does not read `~/.bashrc`, `~/.profile`, or `/etc/bash.bashrc`.

This is intentional. Scripts should define what they need explicitly.

---

## Where should I put my setting?

### Add an alias

Use `~/.bashrc`:

```bash
alias ll='ls -alF'
```

### Change the prompt

Use `~/.bashrc`:

```bash
PS1='\u@\h:\w\$ '
```

### Add `$HOME/.local/bin` to `PATH` for one user

Use `~/.profile`:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

### Set `JAVA_HOME` for one user

Use `~/.profile`:

```sh
export JAVA_HOME="/usr/lib/jvm/default-java"
```

### Set `JAVA_HOME` for all users

Use `/etc/environment` if a simple static value is enough:

```text
JAVA_HOME="/usr/lib/jvm/default-java"
```

If you need shell logic, use a shell startup file such as `/etc/profile.d/java.sh` instead.

### Configure Bash completion for all users

Use `/etc/bash.bashrc` or a distribution-supported completion file.

---

## Common mistakes

### Mistake 1: Putting aliases in `~/.profile`

Aliases are interactive shell features. Put them in `~/.bashrc` instead.

### Mistake 2: Using `export` in `/etc/environment`

`/etc/environment` is not a shell script. Use plain key-value assignments only.

Wrong:

```bash
export FOO=bar
```

Right:

```text
FOO="bar"
```

### Mistake 3: Expecting scripts to read `~/.bashrc`

Scripts should not depend on interactive shell startup files.

If a script needs a variable, set it inside the script, pass it from the caller, or load a dedicated configuration file explicitly.

### Mistake 4: Modifying `/etc/bash.bashrc` for personal preferences

This affects all users. For personal preferences, use `~/.bashrc`.

### Mistake 5: Assuming all distributions behave exactly the same

The general ideas are portable, but exact startup behavior can differ between distributions, shells, terminal emulators, display managers, and PAM configuration.

Always test on the target system.

---

## Practical debugging commands

Check the current shell:

```bash
echo "$SHELL"
echo "$0"
```

Check whether the current shell is interactive:

```bash
case $- in
	*i*) echo "interactive" ;;
	*) echo "non-interactive" ;;
esac
```

Check whether Bash is a login shell:

```bash
shopt -q login_shell && echo "login shell" || echo "non-login shell"
```

Check where a variable came from by starting clean shells and comparing values:

```bash
bash --noprofile --norc
bash --login
bash -i
```

Reload `~/.bashrc` manually after editing it:

```bash
source ~/.bashrc
```

Reload `~/.profile` manually after editing it:

```bash
source ~/.profile
```

For `/etc/environment`, log out and log back in, because it is usually applied during login session setup.

---

## Takeaway

Use the files according to their purpose:

- Put personal interactive Bash settings in `~/.bashrc`.
- Put personal login environment settings in `~/.profile`.
- Put system-wide interactive Bash settings in `/etc/bash.bashrc`.
- Put simple system-wide environment variables in `/etc/environment`.

The clean mental model is:

```text
Interactive behavior -> bashrc files
Login environment    -> profile files
Global static env    -> /etc/environment
```

Keeping these responsibilities separate makes shell configuration easier to debug and safer to maintain.
