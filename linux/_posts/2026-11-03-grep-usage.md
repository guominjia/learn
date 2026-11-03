---
layout: post
title: "grep usage"
date: 2026-11-03
categories: [linux]
tags: [linux, grep]
---

install via `apt isntall grep`

## Parameter

- `-a`（`--text`）：把二进制文件当文本处理。没有它的话，如果文件里有不可打印字符，grep 只会输出 `Binary file ... matches`，不显示匹配的行。
- `-n`（`--line-number`）：在每行匹配结果前加上它在文件中的行号，例如 `123:匹配内容`。
- `-o`（`--only-matching`）只输出匹配到的那部分文本，不输出整行。一行里有多处匹配时，每处单独占一行。

常见用法：

```bash
grep -ao 'reset times: [0-9]*' run.out              # 从日志里提取关键字段
grep -o 'ERROR' run.out | wc -l                     # 统计出现次数（一行多次也会都算上）
grep -oE '\[[A-Z]+\]' run.out | sort | uniq -c      # 统计各类日志标签
grep -n '' f.txt | sed -n '200,300p'                # 空模式能匹配每一行，-n 会给每行加行号
```

它可以和 `-n` 一起用，输出变成 `行号:匹配片段`，例如 `grep -on 'ERROR' f`。