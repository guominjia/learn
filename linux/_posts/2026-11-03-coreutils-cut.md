---
layout: post
title: "cut usage"
date: 2026-11-03
categories: [linux]
tags: [linux, coreutils, cut]
---

- **`cut -d: -f1`**：按字段切。`-d:` 把冒号 `:` 设为分隔符，`-f1` 取第 1 个字段。
  ```bash
  echo "root:x:0:0:/root:/bin/bash" | cut -d: -f1   # root
  ```
  如果不写 `-d`，默认按 Tab 分隔。也可以写 `-f1,3`（取第 1 和第 3 个字段）或 `-f2-`（取第 2 个字段及之后所有字段）。

- **`cut -c1-200`**：按字符切。只保留每行第 1 到第 200 个字符，超出的部分截掉。常用来防止日志里的超长行刷屏。
  ```bash
  echo "abcdefg" | cut -c1-3   # abc
  ```
  也可以写 `-c5-`（从第 5 个字符到行尾）或 `-c-10`（前 10 个字符）。

注意：`-c` 在 GNU coreutils 里其实按**字节**切，所以碰到中文这类多字节字符可能从中间截断，导致乱码。