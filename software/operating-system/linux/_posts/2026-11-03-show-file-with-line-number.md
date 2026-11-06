---
layout: post
title: "Show file with line number"
date: 2026-11-03
categories: [linux]
---

| 命令 | `-n` 含义 | 示例 |
|---|---|---|
| `cat -n` | 给每行加行号，和 `grep -n` 类似 | `cat -n f.txt` → `1  第一行` |
| `head -n N` | 只输出前 N 行，**不加行号** | `head -n 20 f.txt` |
| `tail -n N` | 只输出最后 N 行，**不加行号** | `tail -n 40 f.txt` |

`head` 和 `tail` 的 `-n` 后面要跟数字，表示行数。下面两种写法也常用：
- `tail -n +100 f.txt`：从第 100 行一直输出到文件末尾。
- `head -n -5 f.txt`（GNU）：输出除最后 5 行之外的全部内容。

`head -20` 是 `head -n 20` 的旧式简写，两者等价。

如果想同时显示行号并截取一段，可以把命令组合起来，例如 `cat -n f.txt | tail -n 40`，或者 `sed -n '200,300p'`（取第 200～300 行，但不加行号）。

`sed` 可以用 `=` 命令输出行号，但行号会单独占一行：

```bash
sed -n '200,300{=;p}' f.txt
# 200
# 第200行内容
# 201
# ...
```

要让行号和内容在同一行，可以用下面几种写法：

```bash
cat -n f.txt | sed -n '200,300p'                    # 先编号再截取，最常用
cat -n f.txt | head -n 20                           # 前 20 行带行号
cat -n f.txt | tail -n 40                           # 后 40 行带行号，行号是原文件的真实行号
awk 'NR>=200 && NR<=300 {print NR": "$0}' f.txt     # awk 的 NR 就是当前行号
grep -n '' f.txt | sed -n '200,300p'                # 空模式能匹配每一行，-n 会给每行加行号
```

写法的关键是先用 `cat -n`、`grep -n` 或 `awk` 编号，再交给 `head`/`tail`/`sed` 截取，这样显示的行号才是原文件中的行号。