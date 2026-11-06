---
layout: post
title: "SSH issue in windows"
date: 2026-11-06
categories: [ssh]
---

## Windows 遇到下面问题

运行下面命令，预期结果与实际结果不符

| 命令 | 预期 | 实际 |
|---|---|---|
| `ssh remote_host "cd Projects/ && pwd"` | `/home/remote_user/Projects` | `/home/remote_user/Projects` |
| `ssh remote_host sh -c "cd Projects/ && pwd"` | `/home/remote_user/Projects` | `/home/remote_user` |
| `ssh remote_host bash -c "cd Projects/ && pwd"` | `/home/remote_user/Projects` | `/home/remote_user` |
| `ssh remote_host bash -c "echo $SHELL"` | `/bin/bash` | 无任何输出 |
| `ssh remote_host sh -c "echo $SHELL"` | `/bin/bash` | 无任何输出 |
| `ssh remote_host "echo $SHELL"` | `/bin/bash` | `/bin/bash` |

也就是加了 `bash -c` 和 `sh -c` 都没达到预期效果

## 解决过程

问了 AI，获得了正确答案

原因：**ssh 会把命令参数拼成一个字符串，再交给远端的登录 shell 重新解析一遍**。本地加的双引号**只起分组作用**，到远端前已经被去掉了。

### 举实际讲解

`ssh remote_host "cd Projects/ && pwd"`

远端直接执行 `bash -c "cd Projects/ && pwd"`，因此输出预期结果

`ssh remote_host sh -c "cd Projects/ && pwd"`

1. cmd.exe 和 C 运行时去掉引号后，ssh 收到 3 个参数：`sh`、`-c`、`cd Projects/ && pwd`。
2. ssh 用空格把它们拼成一个字符串：`sh -c cd Projects/ && pwd`。
3. 远端执行的相当于 `bash -c 'sh -c cd Projects/ && pwd'`，bash 会这样拆：
   - `sh -c cd Projects/`：子 shell 只执行 `cd`，`Projects/` 被当成 `$0`。不带参数的 `cd` 回到家目录，而且它是在子进程里执行的，不影响外面。
   - `&& pwd`：由远端 bash 在初始目录（家目录）里执行，所以输出 `/home/remote_user`。

其它错误是类似原因。

### 解决方案

三种：
- `ssh remote_host "<待运行命令>"`
- `ssh remote_host "sh -c '<待运行命令>'"`
- `ssh remote_host "bash -c '<待运行命令>'"`

工作原理：在 cmd 里单引号是普通字符，会原样传给远端，由远端 bash 当作引号处理。

注意：PowerShell 的引号规则不同，命令复杂时用 base64 传过去更稳妥。

### 关于 `sh` 与 `$0` 的解释

`sh -c <命令字符串> [arg0 arg1 ...]` 的规则是：

- `-c` 后面**只有第一个参数**是要执行的脚本。
- 后面的参数依次赋给 `$0`、`$1`、`$2`……，脚本可以读取，但不会被执行。

**举实例讲解**：`ssh remote_host sh -c "'echo $0' Projects/"` 会显示 `Projects/`

- `'echo $0'` 会被作为整体让 `sh` 执行，`Projects/` 作为参数 `$0`，于是显示 `Projects/`
- 当没 `'` 时，只有 `echo` 让 `sh` 执行，其后的内容都作为 `sh` 命令参数，但整个执行只有 `echo`，参数其实没起到任何作用。

## 更多有趣例子

### 1. `sh -c ""echo $0;cd $0;pwd" Projects/ && pwd"`

`""` 是一对空引号，后面的 `"..."` 只起分组作用，本地都会被去掉。ssh 拼接后，远端 bash 收到的是：

```bash
sh -c echo $0;cd $0;pwd Projects/ && pwd
```

远端 bash 按 `;` 拆成几条命令执行。注意 `$0` 现在由**远端 bash 自己**展开，它是 ssh 启动的 `bash -c`，所以 `$0` 的值是 `bash`：

| 远端执行 | 输出 |
|---|---|
| `sh -c echo bash` | 空行（脚本只是 `echo`，`bash` 成为 `$0`） |
| `cd bash` | `bash: line 0: cd: bash: No such file or directory` |
| `pwd Projects/` | `/home/remote_user`（`pwd` 忽略参数） |
| `&& pwd` | `/home/remote_user` |

最终输出是：

```

/home/remote_user
/home/remote_user
bash: line 0: cd: bash: No such file or directory
```

但这个结果和预期也不同，预期是：

```

bash: line 0: cd: bash: No such file or directory
/home/remote_user
/home/remote_user
```

解释是这样：

执行顺序没有变，变的只是**输出到屏幕上的顺序**。stdout 和 stderr 是两条独立的通道，ssh 不保证它们之间的先后顺序。

远端实际按这个顺序执行：

| 远端执行 | 写到 |
|---|---|
| `sh -c echo`（`$1` 被远端 bash 展开成空） | stdout：空行 |
| `cd bash` | **stderr**：`cd: bash: No such file or directory` |
| `pwd Projects/` | stdout：`/home/remote_user` |
| `&& pwd` | stdout：`/home/remote_user` |

不分配终端时，远端 sshd 用两根独立的管道接收 stdout 和 stderr，再通过 SSH 通道分别转发。本地 ssh.exe 把它们分别写到控制台，所以 stderr 的那一行可能先到，也可能后到。

运行下列命令能看到顺序是期望顺序：

```
ssh -t remote_host sh -c ""echo $0;cd $0;pwd" Projects/ && pwd"
ssh remote_host sh -c "echo $1;cd $0;pwd Projects/ && pwd 2>&1"
```

### 2. `ssh remote_host sh -c '"echo $0;cd $0;pwd" Projects/ && pwd'`

这个命令的问题是：cmd 只认双引号，单引号对它来说是普通字符；

#### 1. cmd 把 `&&` 当成了本地命令分隔符

cmd 只把 `"echo $0;cd $0;pwd"` 当作引号区间，`&&` 落在引号**外面**。于是本地执行的是两条命令：

```cmd
ssh remote_host sh -c '"echo $0;cd $0;pwd" Projects/
pwd'
```

要让 cmd 把其当作整体，要把 `&&` 当成普通字符，用 `^` 转义：

```
ssh remote_host sh -c '"echo $0;cd $0;pwd" Projects/ ^&^& pwd'
```

#### 2. ssh 收到的参数里只剩一个单引号

参数解析时双引号被去掉，单引号保留。ssh 收到的参数是：

```text
sh
-c
'echo $0;cd $0;pwd
Projects/
```

拼接后，远端 bash 收到的是：

```bash
sh -c 'echo $0;cd $0;pwd Projects/
```

这里只有开头一个单引号，没有闭合，所以报了这两条错误：

```text
unexpected EOF while looking for matching `''
syntax error: unexpected end of file
```

#### 3. `pwd'` 错误因为 ssh 错误而没执行

`pwd'` 本来会报错，但因为 ssh 出错了，所以这个指令本身也没执行

### 3. `ssh remote_host sh -c "\"echo $0;cd $0;pwd Projects/ && pwd\""`

类似上面，要让命令正确执行，需要转义 `&&`

## Windows C 运行时

疑问：C运行时没看到关闭引号，为何不报错？

答案: 因为 Windows C 运行时的命令行解析规则里根本没有“报错”这一项。引号没闭合时，它会把引号区间一直延续到命令行末尾，然后正常生成 `argv`。

这是有意设计的：

- **C 运行时只负责拆分参数。** 任何字符串都必须能拆成 `argv`，因为这一步发生在 `main` 之前，程序还没机会处理错误。

规则可以查 Microsoft 文档 *Parsing C command-line arguments*：`"` 用来切换“在引号内/不在引号内”的状态，命令行结束时也就结束了引号区间，文档没有把它定义为错误。

## 总结

- **ssh 把参数拼成一个字符串，再交给远端 shell 解析**：这是 ssh 自身的行为，因为 SSH 协议只能发送一个命令字符串。
- **本地去掉引号**：任何本地 shell 都会先处理一遍引号，但不同 shell 处理机制不同。Windows 认双引号但**不认单引号**。
- **变量展开**：Windows 会展开 `%VAR%`，Linux 对单引号内的 `$VAR` 不展开，但**会展开双引号内**的 `$VAR`。
- **ssh.exe 的 C 运行时认 `\"`，把它当成一个普通的 `"` 字符。**
- **cmd 不认 `\` 转义。**
- cmd 判断引号“开/关”只是为了决定 `&`、`|`、`>` 这类字符要不要当成特殊符号，它**不会删除任何字符**。
- **Git for Windows 自带的 ssh** 现象可能会和上面不同。它通过 MSYS2 运行时解析命令行，规则和 C 运行时不完全一样，例如会处理单引号等，所以同一条命令可能得到不同的结果。