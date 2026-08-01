---
layout: post
title: "Bash Script Paths, Regex Captures, and .env Variable Expansion"
date: 2026-07-31
categories: [linux]
tags: [linux, bash, shell, environment, regex, dotenv]
---

Bash scripts often need to find their own directory, parse configuration, and expand values that refer to earlier entries. These tasks look compact in Bash, but the syntax combines arrays, pattern removal, and regular expressions. This post breaks down the pieces used by a small `.env` loader.

## `BASH_SOURCE` Is More Reliable Than `$0` for Sourced Files

`BASH_SOURCE` is a Bash built-in array variable. `${BASH_SOURCE[0]}` identifies the Bash source file currently being executed. That makes it useful when a script may be loaded with `source` (or `.`):

```bash
# env.sh
printf 'loaded from: %s\n' "${BASH_SOURCE[0]}"
```

```bash
source ./env.sh
```

In this situation, `$0` usually still names the shell that was started, such as `bash`. It does not identify `env.sh`. `${BASH_SOURCE[0]}` does.

The value is the path spelling used to load the file. It may be relative rather than an absolute, canonical path, so a production script that needs a physical directory should resolve it separately.

## Extracting the Directory with `%`

Given a script path:

```bash
SCRIPT_PATH=/home/user/env/env.sh
SCRIPT_DIR="${SCRIPT_PATH%/*}"
```

`SCRIPT_DIR` becomes:

```text
/home/user/env
```

The `%` operator removes a suffix matching a shell pattern. A single `%` removes the **shortest** matching suffix; `%%` removes the **longest** matching suffix.

For the pattern `/*`, the shortest suffix that matches starts at the final slash and continues to the end of the value:

```text
/home/user/env/env.sh
              ^------ removed by %/*
```

This concise form assumes that `SCRIPT_PATH` contains a slash. If it contains only a filename such as `env.sh`, the pattern does not match and the value is unchanged. Use `dirname` or normalize the input when that case matters.

## `BASH_REMATCH` Stores Regex Capture Groups

`BASH_REMATCH` is another Bash built-in array. Bash populates it after a successful regular-expression match in `[[ ... =~ ... ]]`.

```bash
line='PORT=8080'

if [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
	key="${BASH_REMATCH[1]}"
	value="${BASH_REMATCH[2]}"
fi
```

After this match, the array contains:

| Element | Value | Meaning |
|---|---|---|
| `${BASH_REMATCH[0]}` | `PORT=8080` | The complete matching text |
| `${BASH_REMATCH[1]}` | `PORT` | The first capture group |
| `${BASH_REMATCH[2]}` | `8080` | The second capture group |

Capture groups are the parenthesized expressions in the regular expression. `BASH_REMATCH` is updated by each successful `=~` match, so copy the values you need before performing another regex match.

## Removing Leading and Trailing Whitespace

The following two expansions trim spaces, tabs, and other characters in the POSIX `[:space:]` character class from a value:

```bash
TRIMMED_VALUE="${TRIMMED_VALUE#"${TRIMMED_VALUE%%[![:space:]]*}"}"
TRIMMED_VALUE="${TRIMMED_VALUE%"${TRIMMED_VALUE##*[![:space:]]}"}"
```

They are easier to understand by reading the inner expansion first.

### First line: remove leading whitespace

```bash
${TRIMMED_VALUE%%[![:space:]]*}
```

The inner `%%` expansion removes the longest suffix that begins with the first non-whitespace character. What remains is the continuous whitespace at the beginning. The outer `#` removes that prefix.

```text
"   hello  " -> inner result: "   " -> outer removal: "hello  "
```

### Second line: remove trailing whitespace

```bash
${TRIMMED_VALUE##*[![:space:]]}
```

The inner `##` expansion removes the longest prefix ending at the final non-whitespace character. What remains is the continuous whitespace at the end. The outer `%` removes that suffix.

```text
"hello  " -> inner result: "  " -> outer removal: "hello"
```

Together, the first line removes leading whitespace and the second removes trailing whitespace.

## `PARSED_VARS` Is Not a Copy of the Environment

An `.env` loader may declare an associative array like this:

```bash
declare -A PARSED_VARS
```

This does not copy every existing shell environment variable. It creates an initially empty Bash associative array. The loader stores only entries it has already parsed successfully:

```bash
PARSED_VARS["$key"]="$expanded_value"
```

That storage makes earlier entries available to later entries in the same `.env` file:

```dotenv
HOST=example.com
URL=https://${HOST}/api
```

When the loader expands `URL`, it can look in `PARSED_VARS` first and find `HOST=example.com`. The resulting value is:

```text
https://example.com/api
```

This precedence is important. A pre-existing shell variable named `HOST` should not unexpectedly override the `HOST` value that was already declared in the `.env` file being parsed.

## How `cmd.exe` Differs

Windows `cmd.exe` also supports environment variables, but its syntax is different:

```bat
set "NAME=value"
echo %NAME%
```

Microsoft documents `set` as the command that displays, creates, changes, and removes `cmd.exe` environment variables. Batch files can expand a variable with `%NAME%`.

Batch files also support special parameter modifiers. For example, `%~dp0` combines the drive (`d`) and path (`p`) of the batch file's `%0` parameter. The related `%~dp1` syntax applies those modifiers to the first argument.

```bat
@echo off
echo This batch file is in %~dp0
```

This is useful for locating a `.bat` or `.cmd` file, but it is a different model from Bash arrays and parameter expansion. `cmd.exe` does not offer a Bash-style associative array or the `BASH_REMATCH` regex-capture mechanism, so substantial parsing logic is usually more awkward in batch files.

## Summary

- Use `${BASH_SOURCE[0]}` when a Bash file may be sourced and must identify that file rather than the invoking shell.
- `${value%pattern}` and `${value%%pattern}` remove the shortest and longest matching suffixes; `#` and `##` do the analogous work on prefixes.
- `BASH_REMATCH` exposes a successful `[[ string =~ regex ]]` match and its capture groups.
- The two nested parameter expansions trim leading and trailing whitespace without calling an external command.
- A `PARSED_VARS` associative array contains parsed `.env` values, not a snapshot of the process environment, allowing earlier file entries to take precedence during later expansion.

## References

- [bash(1) manual](https://man7.org/linux/man-pages/man1/bash.1.html) documents `BASH_SOURCE`, `BASH_REMATCH`, associative arrays, regular-expression matching, and Bash parameter-expansion operators.
- [Microsoft Learn: `set`](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/set_1) documents `cmd.exe` environment-variable assignment and `%NAME%` expansion.
- [Microsoft Learn: `call`](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/call) documents batch parameters and `%~dp1` drive-and-path modifiers, which are the basis for `%~dp0` in a batch file.
