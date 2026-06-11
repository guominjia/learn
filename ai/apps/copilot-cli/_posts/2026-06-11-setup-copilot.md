---
title: Fixing "copilot is not recognized" on Windows
categories: [ai, copilot-cli]
tags: [github-copilot, copilot-cli, npm, windows, powershell]
---

When I tried to set up `copilot-cli` on Windows, the command failed with:

```text
'copilot' is not recognized as an internal or external command,
operable program or batch file.
```

This post explains why it happens and how to fix it quickly.

## Root Cause

I originally ran:

```powershell
npm run @github/copilot
```

That command is not an install command. `npm run` only executes scripts defined in `package.json`, so it does not install `@github/copilot` globally.

## Correct Installation

Install the CLI globally:

```powershell
npm install -g @github/copilot
```

Verify installation:

```powershell
copilot --version
```

## If `copilot` Is Still Not Found

If the package is installed but the shell still cannot find `copilot`, your npm global bin path is likely missing from `PATH`.

### 1) Check npm global prefix

```powershell
npm config get prefix
```

Common output examples:

- `C:\Users\<you>\AppData\Roaming\npm`
- `C:\Program Files\nodejs`

### 2) Check whether PATH already contains npm-related entries

```powershell
$env:Path -split ';' | Where-Object { $_ -like '*npm*' }
```

### 3) Add npm global bin to user PATH (if needed)

```powershell
$npmPrefix = npm config get prefix
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$npmPrefix", "User")
```

Restart PowerShell and run:

```powershell
copilot --version
```

## Fast Alternative: Use `npx`

If you want to avoid PATH issues, run the CLI with `npx`:

```powershell
npx @github/copilot --version
npx @github/copilot <command>
```

This is often the fastest way to get started.

## Recommended Workflow

For stability and convenience on Windows:

1. Install globally with `npm install -g @github/copilot`
2. Verify with `copilot --version`
3. If unresolved, use `npx @github/copilot ...` while fixing `PATH`

That sequence resolves most setup problems in a few minutes.