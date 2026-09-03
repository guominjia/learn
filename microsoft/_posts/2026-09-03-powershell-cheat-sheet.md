---
layout: post
title: "PowerShell Pipeline Cheat Sheet"
date: 2026-09-03
categories: [microsoft, powershell]
tags: [powershell, cheat-sheet, pipeline]
---

These commands are useful for finding items, filtering them, and working with the results in a PowerShell pipeline.

| Command | Use |
|---|---|
| `Set-Location` | Change the current directory. |
| `Get-ChildItem` | List files and folders. |
| `Get-Content` | Read a file. |
| `Where-Object` | Keep items that match a condition. |
| `Select-Object` | Choose properties or limit results. |
| `ForEach-Object` | Run a command for each item. |
| `Select-String` | Search text in files. |

## A common pipeline

```powershell
Get-ChildItem -Filter *.log |
	Where-Object Length -gt 1KB |
	Select-Object Name, Length
```

Read matching lines from several files:

```powershell
Get-ChildItem -Filter *.log |
	ForEach-Object { Select-String -Path $_.FullName -Pattern 'error' }
```