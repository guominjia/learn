---
layout: post
title: "SSH bastion"
date: 2026-11-16
tags: [ssh]
---

`ssh -N -L` 是 SSH 端口转发（隧道）命令，各部分含义如下：

| 参数 | 含义 |
|------|------|
| `-L` | **本地端口转发**（Local Forwarding）。将本地某个端口上的流量，通过 SSH 加密隧道转发到远程目标地址的指定端口。 |
| `-N` | **不执行远程命令**（No command）。只建立隧道连接，不打开远程 shell。通常与 `-L` 或 `-R` 配合使用，让 SSH 纯做隧道。 |

**典型用法：**

```bash
ssh -N -L 本地端口:目标主机:目标端口 用户@跳板机
```

**示例：**

```bash
ssh -N -L 3306:database.internal.com:3306 user@bastion.example.com
```

这条命令的作用：
1. 在本地监听 `3306` 端口
2. 通过 SSH 连接到跳板机 `bastion.example.com`
3. 所有发到本地 `3306` 端口的数据，都会加密传输到跳板机，再由跳板机转发到 `database.internal.com:3306`
4. `-N` 表示不登录远程 shell，纯做隧道

**常见场景：**
- 访问内网数据库/Web 服务（无需 VPN）
- 加密不安全的协议（如 MySQL、Redis）
- 绕过防火墙限制

加上 `-f`（后台运行）可以让它驻留后台：`ssh -f -N -L ...`。