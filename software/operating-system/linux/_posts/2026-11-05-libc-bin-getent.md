---
layout: post
title: "getent usage"
date: 2026-11-05
categories: [linux]
---

`getent` 可用来查看用户权限或者用户所属组。

`getent` 来自 glibc（Debian/Ubuntu 上是 `libc-bin` 包）。

`getent` 按 `/etc/nsswitch.conf` 的配置查询 NSS（Name Service Switch）数据库。

下述命令查看数据库输出：

```bash
grep '^group' /etc/nsswitch.conf
```

常见输出：

```text
group:  files systemd sss
```

`getent` 会按这一行的顺序依次查询各个来源：

| 来源 | 实际查询的位置 |
|---|---|
| `files` | `/etc/group` |
| `systemd` | systemd 动态用户（DynamicUser） |
| `sss` | SSSD，背后可能是 LDAP 或 AD |
| `ldap` / `winbind` | LDAP 或 Samba/AD |

`id`、`groups`、`ls -l` 显示的属主名也都通过 NSS 解析。

常见用法：

- `getent group $USER`
- `getent passwd $USER`