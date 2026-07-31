---
layout: post
title: "PowerShell Command History: HistoryInfo and PSReadLine Persistence"
date: 2026-07-31
categories: [microsoft, powershell]
tags: [powershell, history, psreadline, command-line, shell]
---

PowerShell has two related kinds of command history. `Get-History` exposes commands recorded for the current PowerShell session, while PSReadLine maintains the command lines that can be persisted and recalled in later sessions. They are connected in everyday use, but they are not the same store.

Understanding this distinction helps answer two common questions:

1. Is a history entry a `HistoryInfo` object?
2. Does the command window write its current history to `HistorySavePath` immediately?

## Session History Uses `HistoryInfo`

The objects returned by `Get-History` are normally instances of:

```powershell
[Microsoft.PowerShell.Commands.HistoryInfo]
```

In practice, use the cmdlet or its alias rather than constructing the type yourself:

```powershell
Get-History

# Alias
h
```

For example, filter commands that contain `git`:

```powershell
Get-History | Where-Object CommandLine -match 'git'
```

Inspect the returned object type and properties with:

```powershell
Get-History | Get-Member
```

Useful `HistoryInfo` properties include:

| Property | Meaning |
|---|---|
| `Id` | The session-specific command number. |
| `CommandLine` | The command text. |
| `ExecutionStatus` | Whether the command completed, failed, or is still running. |
| `StartExecutionTime` | When execution started. |
| `EndExecutionTime` | When execution ended. |

This is in-memory session history. Closing the session normally removes it unless it has been exported explicitly, for example with `Export-Clixml`.

## PSReadLine Uses a Separate Persistent History

PSReadLine provides interactive editing, arrow-key history navigation, and persistent command-line history. Its configuration reveals both the history file and its saving behavior:

```powershell
Get-PSReadLineOption |
	Select-Object HistorySavePath, HistorySaveStyle
```

To read the persisted history file directly:

```powershell
Get-Content (Get-PSReadLineOption).HistorySavePath
```

The file stores command input for future interactive sessions. It does not preserve the `HistoryInfo` metadata available from `Get-History`, such as execution status and timing.

Conceptually, the two paths are:

```text
commands executed in this session -> Get-History -> HistoryInfo objects
commands entered at the prompt    -> PSReadLine -> HistorySavePath
```

The overlap is common but not exact. `Get-History` describes commands executed in the current session; PSReadLine records interactive input history for recall.

## Does PowerShell Save the Current Window Immediately?

The answer depends on PSReadLine's `HistorySaveStyle` setting. Check the value first:

```powershell
(Get-PSReadLineOption).HistorySaveStyle
```

The main options are:

| Value | Behavior |
|---|---|
| `SaveAtExit` | Save history when the PowerShell session exits. |
| `SaveIncrementally` | Append accepted command lines to the history file as they are entered. |
| `SaveNothing` | Do not write history to the history file. |

Set incremental saving:

```powershell
Set-PSReadLineOption -HistorySaveStyle SaveIncrementally
```

With `SaveIncrementally`, PSReadLine writes history after each command executes. This is the closest behavior to real-time persistence and reduces the amount of history lost if the terminal or host process closes unexpectedly.

To inspect the most recently persisted commands:

```powershell
Get-Content (Get-PSReadLineOption).HistorySavePath -Tail 20
```

## Practical Checks

Use these commands when diagnosing what PowerShell has stored:

```powershell
# Session execution history and its object type
Get-History | Select-Object -Last 10
Get-History | Get-Member

# PSReadLine persistence configuration
Get-PSReadLineOption |
	Select-Object HistorySavePath, HistorySaveStyle, MaximumHistoryCount

# Inspect the commands that have already been persisted
Get-Content (Get-PSReadLineOption).HistorySavePath -Tail 20
```

## Takeaway

`[Microsoft.PowerShell.Commands.HistoryInfo]` is the type normally returned by `Get-History`, so it represents the current session's execution history. `HistorySavePath` belongs to PSReadLine and holds persisted command-line input for later sessions.

To have commands written after each execution, configure PSReadLine with `SaveIncrementally`. With `SaveAtExit`, history is persisted when the PowerShell session exits. Treat `Get-History` and the PSReadLine history file as complementary sources rather than interchangeable representations of the same data.

## References

- [Get-History documentation](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/get-history)
- [about_History](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_history)
- [Get-PSReadLineOption documentation](https://learn.microsoft.com/powershell/module/psreadline/get-psreadlineoption)
- [Set-PSReadLineOption documentation](https://learn.microsoft.com/powershell/module/psreadline/set-psreadlineoption)
