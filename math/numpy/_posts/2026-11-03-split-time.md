---
layout: post
title: "Numpy Reset Time"
date: 2026-06-26
categories: [linux]
tags: [linux, shell, bash, dash, sh, ubuntu, debian]
---

`np.r_[0, np.where(np.diff(t) < 0)[0] + 1, len(t)]` 解释

这行代码把数组 `t` 中**数值下降的位置**当作分界点，算出每一段的起止下标。常见场景是时间戳在中途被重置（例如设备重启后从 0 重新计数）。

逐步拆解：

| 部分 | 含义 |
|---|---|
| `np.diff(t)` | 相邻元素的差 `t[i+1] - t[i]`，长度比 `t` 少 1 |
| `np.diff(t) < 0` | 差为负，说明 `t` 在这一步变小了 |
| `np.where(...)[0]` | 找出这些位置的下标 `i`，表示 `t[i+1] < t[i]` |
| `+ 1` | 换成新一段开头的下标 `i+1` |
| `np.r_[0, ..., len(t)]` | 在前面拼上 `0`，后面拼上 `len(t)`，组成完整的边界数组 |

例子：

```python
t = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3])
np.diff(t)                          # [ 1,  1, -2,  1, -1,  1,  1,  1]
np.where(np.diff(t) < 0)[0]         # [2, 4]
np.where(np.diff(t) < 0)[0] + 1     # [3, 5]
b = np.r_[0, [3, 5], len(t)]        # [0, 3, 5, 9]
```

通常接下来会这样用：

```python
segments = [t[b[i]:b[i+1]] for i in range(len(b) - 1)]
# [array([0, 1, 2]), array([0, 1]), array([0, 1, 2, 3])]
```

`np.r_` 是 NumPy 用来拼接数组的简写，这里相当于 `np.concatenate([[0], idx, [len(t)]])`。