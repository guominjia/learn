---
title: Create Nested Directories in Python (os vs pathlib)
tags: [os, pathlib, mkdir, makedirs, windows]
---

When creating directories in Python, `os.makedirs` and `pathlib.Path.mkdir` are both valid options.

The most common mistake is calling `Path.mkdir()` for a deep path **without** `parents=True`.

## Quick Commands

```bash
py -c "import os; os.makedirs('a/b/c/d', exist_ok=True)"
py -c "import pathlib; pathlib.Path('a/b/c/d').mkdir(parents=True, exist_ok=True)"
```

These two commands create the full nested path safely, even if part of it already exists.

## Common Error

```bash
py -c "import pathlib; pathlib.Path('a/b/c/d').mkdir(exist_ok=True)"
```

This raises:

```text
FileNotFoundError: [WinError 3] The system cannot find the path specified: 'a\\b\\c\\d'
```

## Why It Fails

`Path.mkdir(exist_ok=True)` only handles the final directory if its parent already exists.

For nested paths, Python needs permission to create missing parent directories, which is what `parents=True` does.

## Recommendation

- Use `os.makedirs(path, exist_ok=True)` for a simple procedural style.
- Use `Path(path).mkdir(parents=True, exist_ok=True)` if you prefer `pathlib`.

Both are correct. The key is: for nested paths, always include parent creation.