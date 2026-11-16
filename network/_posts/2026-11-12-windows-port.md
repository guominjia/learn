---
layout: post
title: "windows port"
date: 2026-11-12
---

Windows 和 Linux（需安装 `net-tools` 包） 都有 `netstat` 命令。

Windows 可以用 `netstat -ano` 查看当前端口。与下面的 Powershell 结合可以查看端口对应的应用，已经关闭应用：

```powershell
Get-NetTCPConnection -State Listen | Select-Object LocalPort, OwningProcess, @{N="ProcessName";E={(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName}}

Get-Process -Id <pid> | Select-Object Id, ProcessName, StartTime, CommandLine

Get-CimInstance Win32_Process -Filter "ProcessId=<pid>" | Select-Object CommandLine
Get-CimInstance Win32_Service -Filter "Name='<name>'" | Select-Object Name, DisplayName, PathName, StartName, State, StartMode

taskkill /PID <pid> /F
```

## 系统端口列表

| 端口 | 进程 | 说明 |
|------|------|------|
| **135** | `svchost` | Windows RPC 服务 |
| **139** | `System` | NetBIOS |
| **445** | `System` | SMB 文件共享 |
| **623** | `LMS` | Intel AMT/LMS |
| **3389** | `svchost` | 远程桌面 (RDP) |
| **5040** | `svchost` | Windows 服务 |
| **5985** | `System` | WinRM (远程管理) |
| **7680** | `svchost` | Windows Update 优化传输 |
| **42050** | `OneDrive` | OneDrive 同步 |
| **47001** | `System` | WinRM |
| **49664-49715** | `lsass/wininit/svchost/spoolsv/services/jhi_service` | Windows 系统动态端口 |

**困惑与疑问**：
- 上面端口真的对吗？