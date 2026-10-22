---
layout: post
title: "SSH issue when connect host with multiple IPs"
date: 2026-10-22
categories: [shell]
tags: [ssh]
---

假设有一个登录服务名 `login.example.com`，DNS 为它配置了两个 IP：`IP1` 和 `IP2`，分别对应两个登录节点。这里的 `example.com` 是示例域名；服务名代表的是一组可提供同一服务的节点。

连接 `login.example.com` 时，SSH 会通过系统解析器获取地址并尝试建立连接，它可能连到任一节点。

如果两个节点使用不同的 SSH 主机密钥，客户端连接时看到的服务名虽然相同，实际连接到的节点及其主机密钥却可能不同。这时 SSH 可能报 `HOST IDENTIFICATION HAS CHANGED` 并拒绝连接。

**临时处理**：删除 `known_hosts` 中的记录来暂时绕过校验。

## 根源

同一个服务名对应两个使用不同主机密钥的节点，而 SSH 会校验连接到的节点是否持有客户端信任的密钥。若 `known_hosts` 里只记录了其中一个节点的密钥，连接到另一个节点时就会校验失败。

在默认配置下，`UpdateHostKeys` 通常处于启用状态。连接经过主机密钥验证并完成用户认证后，如果服务器支持相应扩展，SSH 可以接收服务器通知的其他主机密钥并更新 `known_hosts`。这可能让不同节点各自维护的密钥记录相互影响。

**解决方法**：把两个节点的主机密钥都可信地添加到服务名对应的 `known_hosts` 记录中，并设置 `UpdateHostKeys no`，避免节点各自发送的密钥通知影响这些记录。在 `~/.ssh/config` 中添加：
```
Host login.example.com
  User your_user_name
  UpdateHostKeys no
```

## `ssh-keygen` 和 `ssh-keyscan`

| 工具 | 作用 | 用法 |
|---|---|---|
| `ssh-keygen` | 管理 SSH 密钥：生成密钥对、算指纹（`-l`）、从 `known_hosts` 删除某台主机的记录（`-R`） | `-lf` 算指纹，`-R` 删记录 |
| `ssh-keyscan` | 不登录，直接从服务器取回主机公钥 | 在我的 Windows 环境中没有输出，因此用下面的方法获取指纹 |

## Windows获取指纹方法

```powershell
$f = "$env:USERPROFILE\.ssh\known_hosts"
$want = @{ 'IP1' = 'Finger 1'; 'IP2' = 'Finger 2' }
foreach ($ip in $want.Keys) {
  $tmp = "$env:TEMP\kh_$ip"; Remove-Item $tmp -ErrorAction SilentlyContinue
  ssh -o BatchMode=yes -o UserKnownHostsFile=$tmp -o GlobalKnownHostsFile=NUL -o StrictHostKeyChecking=accept-new -o HostKeyAlgorithms=ssh-ed25519 nobody@$ip exit 2>$null
  if ((ssh-keygen -lf $tmp) -match [regex]::Escape($want[$ip])) {
    $line = (Get-Content $tmp) -replace "^$([regex]::Escape($ip))", 'login.example.com'
    if (-not (Select-String -Path $f -SimpleMatch $line.Split(' ')[2] -Quiet)) { Add-Content $f $line -Encoding ascii; "$ip added" } else { "$ip already present" }
  } else { "$ip fingerprint MISMATCH - not added" }
}
```

逐行说明：

1. `ssh ... nobody@$ip exit`：只为拿到服务器的主机密钥，不登录。
   - 用户名 `nobody` 是故意写的，登录本来就会失败。
   - `UserKnownHostsFile=$tmp` 让 ssh 把密钥写进一个临时文件，不碰真正的 `known_hosts`。
   - `accept-new` 表示遇到新主机自动记下密钥。
   - `HostKeyAlgorithms=ssh-ed25519` 只要 ED25519 这一种密钥。
   - 用这种写法，是因为这台 Windows 上的 `ssh-keyscan` 没有任何输出。
2. `ssh-keygen -lf $tmp`：算出临时文件里那把密钥的指纹，跟 `$want` 里的期望值比对。
   - 比对一致才往下走，这一步是为了防止中间人攻击。
   - 不一致就输出 `MISMATCH`，什么都不写。
3. `-replace "^$ip", 'login.example.com'`：临时文件里这条记录是按 IP 存的，改成按服务名存。
4. `Select-String ... -Quiet` 再加 `Add-Content`：`known_hosts` 里还没有这把密钥才追加，避免写重复。

## 关于HostKeyAlias解释

```
ssh -o HostKeyAlias=login.example.com your_user_name@IP1 hostname
```

- **`HostKeyAlias` 的作用**：在 `known_hosts` 里查找和写入密钥时，用这个别名代替真实的主机名或 IP。这里虽然是连 IP，ssh 查的、改的都是 `login.example.com` 名下的记录，没有任何按 IP 存的记录。
- **`UpdateHostKeys` 仍可能影响记录**：设置 `HostKeyAlias` 只改变 SSH 查找和保存主机密钥时使用的名字，不会让不同节点的密钥自动变得相同。若节点密钥不同，应确保别名下已可信地记录所有节点的密钥；也可以设置 `UpdateHostKeys no`，避免服务器发送的密钥通知改动这些记录。

建议：用了 `HostKeyAlias`，最好同时加上 `-o UpdateHostKeys=no`。