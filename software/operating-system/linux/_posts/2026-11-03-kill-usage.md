---
layout: post
title: "kill usage"
date: 2026-11-03
categories: [linux]
---

`kill -0 $PID` 用来检查进程是否还**存在**（也就是是否还在运行）。

- 信号 0 不会真的发给进程，内核只检查目标 PID 是否存在、当前用户有没有权限给它发信号。
- 返回码为 0 表示进程还在，非 0 表示进程已经退出，或者没有权限。
- `2>/dev/null` 用来屏蔽进程不存在时的 `No such process` 报错。

```bash
while kill -0 $PID 2>/dev/null; do sleep 60; ...; done   # 进程活着就继续轮询
kill -0 $PID 2>/dev/null && echo RUNNING || echo EXITED
```

注意以下几点：
- 它**只能判断是否存在，不能判断是否卡住**。进程挂起（hang）、死循环或等待超时的时候，`kill -0` 仍然返回成功。所
- 退出码要用 `wait $PID` 获取，而且只适用于当前 shell 的子进程。
- 如果子进程已经退出、但父进程还没回收（僵尸进程），`kill -0` 仍然返回成功。在同一个 shell 里用 `nohup ... &` 启动后，再用 `wait` 回收即可。
- PID 理论上可能被复用，短时间内轮询一般不会遇到这个问题。