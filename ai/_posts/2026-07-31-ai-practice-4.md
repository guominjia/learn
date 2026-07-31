---
layout: post
title: "Optimizing a Bash .env Loader Without eval"
date: 2026-07-31
categories: [bash, performance]
tags: [bash, env, dotenv, performance, git-bash, shell, security]
---

Loading a `.env` file should be a small startup cost. In one Git Bash environment, however, sourcing an `env.sh` loader took about 34 seconds for a file with 91 entries. The loader was intentionally avoiding `eval`, but its parsing hot path still created a large number of external processes.

This post describes how I changed the loader to use Bash builtins only, preserved its parsing semantics, and reduced the measured median load time from **34,036.8 ms** to **322.3 ms**.

The measurements and conclusions below apply to the tested Git Bash environment on Windows. They are not a claim that every `.env` loader, shell, or operating system will show the same numbers.

## The Problem

The `.env` file contained 91 environment-variable entries:

- 86 entries used double-quoted values.
- Some values contained variable references such as `$VAR` and `${VAR}`.
- The loader needed to preserve the current parsing behavior without executing arbitrary input through `eval`.

The original implementation performed work such as path calculation, whitespace trimming, quote processing, comment handling, and variable expansion by invoking commands including `dirname`, `sed`, and `echo`. It also used command substitution around helper functions.

Each external command has a startup cost. On Git Bash for Windows, repeatedly starting those processes was the dominant cost. Quoted values were especially expensive because they followed paths that started additional processes.

## A Falsifiable Hypothesis

The working hypothesis was straightforward:

> The 34-second load time is primarily caused by external processes and command-substitution subshells in the per-line parsing path, rather than by reading or exporting 91 variables.

This hypothesis could be disproved by replacing the per-line operations with Bash builtins and measuring little or no improvement. Before changing the code, I confirmed that the available Bash version supported the required built-in string operations.

## Keeping the Security Boundary

A common shortcut for `.env` files is to source them directly or reconstruct shell code and pass it to `eval`. That may be acceptable only when the file is fully trusted and is explicitly intended to be a shell program. It was not the desired contract here.

The loader continues to parse a limited `.env` format instead:

- assignment names use shell-style variable names;
- leading and trailing whitespace is handled by the parser;
- quoted values are processed as before;
- `$VAR` and `${VAR}` references are expanded;
- values already parsed from the `.env` file take precedence over variables inherited from the process environment.

The implementation does not turn a value back into shell source code, so it needs no `eval`.

## Moving Work Into Bash Builtins

The optimized script removes external commands from the hot path. Bash parameter expansion handles path and string operations, including trimming and the portions of comment and quote processing that previously used `sed`.

For decoded quoted values, the script uses `printf -v` to assign directly to a variable. This avoids capturing `printf` output through command substitution.

Variable expansion is also done in Bash. Rather than using a helper like this:

```bash
value="$(expand_variables "$value")"
```

the helper scans from left to right and writes its result into a global result variable:

```bash
expand_variables "$value"
value=$EXPANDED_VALUE
```

The scanner recognizes both forms used by the `.env` file:

```text
$VAR
${VAR}
```

It replaces each reference using the values parsed so far, falling back to the existing environment when necessary. Returning through `EXPANDED_VALUE` removes the command-substitution subshell from each expansion.

## Validation Before Benchmarking

Before measuring performance, I ran narrow checks against the rewritten script:

```powershell
$bash = 'C:\Program Files\Git\usr\bin\bash.exe'
$shellPath = (Resolve-Path env\env.sh).Path -replace '\','/'

& $bash -n $shellPath
& $bash --noprofile --norc -c "export PATH=/usr/bin:/bin; unset OPENAI_API_KEY; source '$shellPath'; [[ -v OPENAI_API_KEY ]]"
```

The syntax check passed, and sourcing the real `.env` file exported the expected `OPENAI_API_KEY` variable.

I also tested mixed variable syntax with known values:

```bash
FIRST=one
SECOND=two

expand_variables 'a-$FIRST-b-${SECOND}-c'
[[ "$EXPANDED_VALUE" == 'a-one-b-two-c' ]]
```

This check passed, confirming that both `$VAR` and `${VAR}` were expanded correctly.

## Benchmark Result

The benchmark started a clean, non-interactive Git Bash process and sourced the loader:

```powershell
$command = "export PATH=/usr/bin:/bin; source '$shellPath'"
$times = 1..7 | ForEach-Object {
	(Measure-Command {
		& $bash --noprofile --norc -c $command
	}).TotalMilliseconds
}
```

The original median was **34,036.8 ms**. After the built-in-only rewrite, the seven-run benchmark reported a median of **322.3 ms**.

$$
\frac{34{,}036.8}{322.3} \approx 105.6
$$

That is approximately a **105.6x** reduction in the measured median startup time.

## Takeaway

For a shell parser, the execution model matters as much as the algorithm. A small external command is not small when it is executed once or more for every line of a configuration file, particularly under Git Bash on Windows.

Keeping parsing, string handling, assignment, and variable expansion inside Bash removed the repeated process startup cost while retaining the no-`eval` security boundary. The follow-up checks were intentionally small: syntax, loading a real file, and exact variable-expansion behavior. Once those passed, the benchmark showed that the original hypothesis matched the observed bottleneck.
