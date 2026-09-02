
---
title: "How to Print Only Part of a File in the Terminal"
date: 2026-09-01
tags: [terminal, command-line, linux, windows]
---

# How to Print Only Part of a File in the Terminal

When a file is large, printing everything to the terminal is rarely useful. A common task is to inspect a specific range of lines, such as lines 10 through 20, without opening an editor or reading the whole file.

The right command depends on the shell and operating system. The important distinction is that commands such as `cat` on Unix-like systems and `type` in Windows Command Prompt are designed to output a complete stream. They do not provide a built-in line-range selector, so another tool must do that work.

## Unix-Like Systems

### `head` and `tail`

To print lines 10 through 20, start at line 10 and keep the next 11 lines:

```bash
tail -n +10 file.txt | head -n 11
```

`tail -n +10` skips the first nine lines and begins output at line 10. `head -n 11` limits the result to lines 10 through 20.

This approach is short and uses tools that are available on most Unix-like systems.

### `sed`

`sed` can express the range directly:

```bash
sed -n '10,20p' file.txt
```

The `-n` option disables automatic printing, and the `p` command prints only the selected range. For one-off inspections, this is often the clearest command.

### `awk`

`awk` is useful when the selection needs additional conditions:

```bash
awk 'NR >= 10 && NR <= 20' file.txt
```

`NR` is the current input line number. Because `awk` is a text-processing language, the same command can be extended to filter by content, fields, or other rules.

## Windows

### PowerShell

PowerShell treats file content as a sequence of lines. To select lines 10 through 20:

```powershell
Get-Content file.txt | Select-Object -Skip 9 -First 11
```

The arguments use zero-based skipping: `-Skip 9` skips the first nine lines, and `-First 11` returns the following eleven lines.

For a reusable script, the equivalent form is also readable:

```powershell
$lines = Get-Content file.txt
$lines[9..19]
```

The pipeline form is preferable when the file may be large because it expresses the operation as a bounded selection rather than requiring the entire result to be handled as an indexed collection.

### Command Prompt

The traditional `type` command is convenient for displaying a file, but it does not have a simple option for selecting a line range. A batch-file solution using `for /f` is possible, but it is more cumbersome and has additional parsing rules.

If line-range inspection is a regular task on Windows, PowerShell is usually a better default. Unix-style tools are another option when a compatible environment such as WSL is already part of the workflow.

## A Cross-Platform Python Option

For automation that must behave consistently across operating systems, a small Python script can make the indexing explicit:

```python
from pathlib import Path

start = 10
end = 20

for line in Path("file.txt").read_text().splitlines(keepends=True)[start - 1:end]:
    print(line, end="")
```

This example uses one-based line numbers in the variables and converts the start position to Python's zero-based slice notation.

For very large files, reading the whole file into memory may be unnecessary. A streaming implementation is more appropriate when the script is part of a long-running tool or processes files that do not fit comfortably in memory.

## Choosing a Tool

| Environment | Recommended command | Best suited for |
| --- | --- | --- |
| Unix-like shell | `sed -n '10,20p' file.txt` | A clear, direct range selection |
| Unix-like shell | `tail -n +10 file.txt \| head -n 11` | A compact pipeline using common tools |
| Unix-like shell | `awk 'NR >= 10 && NR <= 20' file.txt` | Ranges combined with text-processing logic |
| PowerShell | `Get-Content file.txt \| Select-Object -Skip 9 -First 11` | Native Windows shell usage |
| Python | A line-slicing script | Cross-platform automation |

The design of `cat` and `type` is intentionally simple: they emit file content and leave specialized transformations to other commands. This separation keeps the basic tools lightweight and makes more complex behavior composable through pipelines.