---
layout: post
title: "Shell behavior of interactive and login"
date: 2026-10-22
categories: [shell]
---

四种组合，以 bash 为例（基于 [GNU Bash Manual 6.2](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html)）：

| | **登录 (login)** | **非登录 (non-login)** |
|---|---|---|
| **交互式 (interactive)** | 读 `/etc/profile` → `~/.bash_profile`（或 `~/.bash_login`、`~/.profile`，取第一个存在的） | 读 `~/.bashrc` |
| **非交互式 (non-interactive)** | 读 `/etc/profile` → `~/.bash_profile`（等），**因为 `--login` 强制按登录 shell 处理** | **什么都不读**（除非设了 `BASH_ENV`） |

## 逐个说明

### 1. 交互式 + 登录 (`interactive + login`)

**触发方式：** `ssh user@host`、文本控制台登录、`bash --login -i`

**读取顺序：**
```
/etc/profile → ~/.bash_profile (或 ~/.bash_login, ~/.profile，取第一个)
```

通常 `~/.bash_profile` 里会 `source ~/.bashrc`，所以实际上也能拿到交互配置。

### 2. 交互式 + 非登录 (`interactive + non-login`)

**触发方式：** 在已登录的桌面环境中打开终端窗口、运行 `bash`

**读取：**
```
~/.bashrc
```

这是最常见的日常场景。`~/.bashrc` 里的别名、提示符、补全等都在这里生效。

### 3. 非交互式 + 登录 (`non-interactive + login`)

**触发方式：** `bash --login script.sh`、`ssh user@host 'command'`（某些 sshd 配置下）

**读取：**
```
/etc/profile → ~/.bash_profile (或 ~/.bash_login, ~/.profile)
```

文档原文：

> If Bash is invoked as an interactive login shell, **or as a non-interactive shell with the `--login` option**, it first reads and executes commands from the file `/etc/profile`...

`--login` 标志强制登录行为，不管是否交互式。但注意：`ssh user@host 'command'` 是否启动登录 shell 取决于 sshd 配置和用户 shell 类型，不是必然的。

### 4. 非交互式 + 非登录 (`non-interactive + non-login`)

**触发方式：** `bash script.sh`、`bash -c 'echo hello'`

**读取：** **什么都不读**（默认情况下）

文档原文：

> When Bash is started non-interactively, to run a shell script, for example, it looks for the variable `BASH_ENV` in the environment, expands its value if it appears there, and uses the expanded value as the name of a file to read and execute.

行为等价于：
```bash
if [ -n "$BASH_ENV" ]; then . "$BASH_ENV"; fi
```

如果 `BASH_ENV` 未设置，就**什么都不读**。

## 特殊情况：sshd 远程调用

文档里还有一个特殊段落 **"Invoked by remote shell daemon"**：

> If Bash determines it is being run non-interactively in this fashion [with stdin connected to a network connection, as when executed by sshd], it reads and executes commands from `~/.bashrc`, if that file exists and is readable.

所以 `ssh user@host 'command'` 实际上可能**会读 `.bashrc`**——因为 bash 检测到 stdin 是网络连接时会特殊处理。这是 bash 的一个特例行为，不适用于 `bash script.sh` 这种本地非交互式调用。

[dash_man]: https://man7.org/linux/man-pages/man1/dash.1.html