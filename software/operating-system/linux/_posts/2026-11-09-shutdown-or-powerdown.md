---
layout: post
title: "Linux shutdown or poweroff"
date: 2026-11-09
categories: [linux]
---

无法直接使用 shutdown 或 poweroff 原因

## 常见原因

1. **普通用户的 `PATH` 不含 `/sbin`、`/usr/sbin`**：Radxa 官方镜像（rsdk / Debian）就是这样。`poweroff`、`shutdown`、`reboot`、`halt` 都在 `/usr/sbin`，并且是指向 `systemctl` 的符号链接。普通用户输入 `poweroff` 时会看到 `command not found`。
2. **用 `su` 切到 root 时没加 `-`**：这样还会沿用原来用户的 `PATH`，同样找不到。
3. **精简或自制镜像**：可能没装 systemd 或 sysvinit，只有 busybox。这种情况相对少见。

先确认是哪一种：

```bash
ls -l /usr/sbin/poweroff /sbin/poweroff   # 文件是否存在
echo $PATH                                # 是否包含 /usr/sbin
ps -p 1 -o comm=                          # 1 号进程是 systemd 还是 init/busybox
```

## 关机方法（按推荐顺序）

```bash
sudo poweroff            # sudo 用 secure_path，里面有 /usr/sbin
sudo /usr/sbin/poweroff  # 直接写完整路径
systemctl poweroff       # systemctl 在 /usr/bin；普通用户可能需要 sudo 或 polkit 授权
sudo shutdown -h now
su -                     # 带 "-" 登录为 root，PATH 正确后再执行 poweroff
```

只有 busybox 的系统：

```bash
busybox poweroff
```

如果系统已经卡住、上面的方法都不行，可以用内核 SysRq 做应急关机。这种方式不会正常停止服务，可能丢数据，应先同步磁盘：

```bash
echo s | sudo tee /proc/sysrq-trigger   # 将数据写回磁盘
echo u | sudo tee /proc/sysrq-trigger   # 把文件系统重新挂载为只读
echo o | sudo tee /proc/sysrq-trigger   # 断电
```

## 让 `poweroff` 以后能直接用

把下面这行加到 `~/.bashrc`：

```bash
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin
```

加上之后，普通用户可以直接输入 `poweroff`。但如果没有 polkit 授权，执行时仍可能提示需要权限，这时用 `sudo poweroff`。

**注意**：Radxa 板子执行 `poweroff` 后，系统会停机，但部分型号的板载电源指示灯会一直亮着，这是正常现象。确认 SSH 断开、活动 LED 熄灭后，再拔掉电源。
