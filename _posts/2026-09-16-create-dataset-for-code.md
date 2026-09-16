---
title: "Prepare dataset and Do statistic"
date: 2026-09-16
tags: [datasets, pandas, ai, llm]
---

## Create dataset from current directory

```python
from pathlib import Path
from datasets import Dataset

root = Path(".")
files = sorted(p for p in root.rglob("*") if p.suffix in {".c", ".h"})

records = [
    {
        "path": str(p.as_posix()),
        "name": p.name,
        "ext": p.suffix,
        "size": p.stat().st_size,
        "text": p.read_text(encoding="utf-8", errors="replace"),
    }
    for p in files
]

ds = Dataset.from_list(records)
print(ds)
ds.save_to_disk("c_sources_ds")
```

要点：

| 需求 | 改法 |
|---|---|
| 只扫当前目录、不递归 | `root.glob("*")` 代替 `rglob("*")` |
| 文件很多/很大，避免一次性进内存 | 用生成器 + `Dataset.from_generator(gen)` |
| 想要 train/test 划分 | `ds.train_test_split(test_size=0.1)` |
| 导出成 parquet/jsonl | `ds.to_parquet("x.parquet")` / `ds.to_json("x.jsonl")` |

大数据量版本：

```python
def gen():
    for p in root.rglob("*"):
        if p.suffix in {".c", ".h"}:
            yield {"path": p.as_posix(), "text": p.read_text(encoding="utf-8", errors="replace")}

ds = Dataset.from_generator(gen)
```

`errors="replace"` 是必要的，源码里常有非 UTF-8 的 latin-1 注释，直接 `read_text()` 会抛 `UnicodeDecodeError`。

## Do statistic for current directory

```python
from pathlib import Path
import pandas as pd

root = Path(".")
df = pd.DataFrame(
    [{"ext": p.suffix.lower() or "<none>", "size": p.stat().st_size} for p in root.rglob("*") if p.is_file()]
)

stat = (
    df.groupby("ext")
    .agg(count=("ext", "size"), total_size=("size", "sum"), avg_size=("size", "mean"))
    .sort_values("count", ascending=False)
)
print(stat)
```

只要个数的话一行就够：

```python
df["ext"].value_counts()
```

补充：

| 需求 | 改法 |
|---|---|
| 只统计当前目录 | `root.glob("*")` |
| 加占比列 | `stat["pct"] = stat["count"] / stat["count"].sum()` |
| 排除 `.git` 等目录 | `if p.is_file() and ".git" not in p.parts` |
| 大小显示成 MB | `stat["total_size"] / 1024**2` |

`p.suffix` 对 `Makefile`、`.gitignore` 这类文件分别返回 `""` 和 `""`（`.gitignore` 整体被当成 stem），所以用 `or "<none>"` 兜底，否则空字符串一组会很难读。

pandas 默认只显示前后各 5 行/列，中间用 `...` 折叠。

```python
pd.set_option("display.max_rows", None)      # 不折叠行
pd.set_option("display.max_columns", None)   # 不折叠列
pd.set_option("display.width", 200)          # 不换行折行
pd.set_option("display.max_colwidth", None)  # 单元格内容不截断
print(stat)
```

只想临时生效，用上下文管理器更干净：

```python
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
    print(stat)
```

其它几种绕开显示限制的办法：

| 场景 | 写法 |
|---|---|
| 只看前 N 行 | `stat.head(50)` |
| 转成字符串一次性打印 | `print(stat.to_string())` |
| 导出看全量 | `stat.to_csv("ext_stat.csv")` |
| Markdown 表格 | `print(stat.to_markdown())` |

`to_string()` 是最省事的——它忽略 `max_rows`/`max_columns` 设置，直接输出完整内容，不用改全局选项。

## Filter directory

```python
rel = p.relative_to(root).as_posix()
if rel.startswith(".") or "/." in rel:
    continue
```

```python
rel = p.relative_to(root)
if any(part.startswith(".") for part in rel.parts[:-1]):   # 只看目录层
    continue
```