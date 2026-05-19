# Formula

In LaTeX, commands like `\mathbf{...}` and `\mathcal{...}` are **font-style commands**. They usually do not change mathematical meaning by themselves, but they help communicate different kinds of mathematical objects.

## Why this matters

Good notation improves readability. In technical writing, readers often infer object type from visual style before reading the full sentence.

## `\mathbf{...}`

`\mathbf{...}` turns Latin letters and digits into **bold upright** symbols.

Common use cases:
- vectors (depending on style guide)
- matrices
- parameter tensors in ML papers

Examples:

$$
\mathbf{x},\ \mathbf{W},\ \mathbf{X}
$$

## `\mathcal{...}`

`\mathcal{...}` turns uppercase letters into **calligraphic** style.

Common use cases:
- sets
- operators
- losses/objectives
- datasets or distributions (style-dependent)

Examples:

$$
\mathcal{N},\ \mathcal{L},\ \mathcal{D}
$$

## Practical notes

1. `\mathcal` is mainly effective and visually clear for uppercase letters. For lowercase letters (for example, `\mathcal{x}`), output is often unattractive or unsupported depending on fonts/packages.
2. For bold Greek symbols (for example, $\beta$), use `\boldsymbol{\beta}` (or `\bm{\beta}` with the `bm` package), **not** `\mathbf{\beta}`.

## Quick style guideline

- Use `\mathbf` for vectors/matrices if your document style prefers bold symbols.
- Use `\mathcal` for higher-level objects like sets, objectives, and named operators.
- Keep notation consistent across the whole document.

## References

- <https://www.cnblogs.com/syqwq/p/15190115.html>
- <https://zhuanlan.zhihu.com/p/441454622>