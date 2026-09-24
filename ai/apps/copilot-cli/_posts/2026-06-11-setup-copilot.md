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

## Builtin MCP

It is not stored in any configuration file. `github-mcp-server` is a built-in MCP server for Copilot CLI, and its definition is embedded directly in the CLI package (`@github/copilot`).

The official docs ([CLI command reference → Built-in MCP servers](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference)) list the built-in servers as `github-mcp-server`, `playwright`, `fetch`, and `time`. They can be used immediately without extra setup.

### Check it

```powershell
copilot mcp list                    # built-in, user, workspace, and plugin servers are grouped by source
copilot mcp get github-mcp-server   # view the configuration and tool list
```

In an interactive session, you can also use `/mcp show github-mcp-server` or `/env`.

### Adjust it via command-line flags

| Flag | Purpose |
|---|---|
| `--disable-builtin-mcps` | Disable all built-in MCP servers |
| `--disable-mcp-server=github-mcp-server` | Disable only this one |
| `--enable-all-github-mcp-tools` | Enable all of its tools. By default, only a subset of tools is enabled by the CLI |
| `--add-github-mcp-tool=TOOL` / `--add-github-mcp-toolset=TOOLSET` | Enable additional tools or tool sets |

`copilot mcp disable github-mcp-server` also persists the disabled state.

If you want full control over its configuration, add another remote GitHub MCP server under a different name in `mcp-config.json`, then disable the built-in one with `--disable-builtin-mcps`.

### If the new MCP tools do not appear

Try one of the following:

1. In the resumed session, run `/mcp reload`, then use `/mcp show github-mcp-server` to see whether the tool count increases.
2. Skip resuming the old session and start a fresh one: `npx copilot --enable-all-github-mcp-tools`. If the tool list is complete in the new session, it confirms the issue is specific to the resumed session.