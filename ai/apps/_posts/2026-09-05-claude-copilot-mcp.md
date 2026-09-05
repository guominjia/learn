---
title: "MCP Configuration in VS Code and Copilot CLI"
date: 2026-09-05
categories: [ai, agent]
tags: [mcp, copilot, claude, vscode]
---

# MCP Configuration in VS Code and Copilot CLI

MCP configuration looks similar across AI clients, but the clients do not all
discover the same files or expect the same top-level JSON key. VS Code and
Copilot CLI are the important example: both can start MCP servers, but their
native configuration formats are different.

## The Short Version

| | VS Code extension host | Copilot CLI / Agent Host |
| --- | --- | --- |
| Workspace file | `.vscode/mcp.json` | `.mcp.json` at the repository root, or `.github/mcp.json` |
| User file | The profile's `mcp.json`, opened with `MCP: Open User Configuration` | `~/.copilot/mcp-config.json`; `COPILOT_HOME` can change the base directory |
| Top-level key | `servers` | `mcpServers` |
| Local server | `command`, `args`, and `env` | `command`, `args`, and `env` |
| Remote server | `type: "http"` and `url` | `type: "http"` or `type: "sse"` and `url` |

The server entries are conceptually similar. The file location and the
top-level key are the compatibility boundary.

## VS Code Configuration

For a VS Code-only workspace, create `.vscode/mcp.json`:

```json
{
	"servers": {
		"playwright": {
			"command": "npx",
			"args": ["-y", "@microsoft/mcp-server-playwright"]
		},
		"github": {
			"type": "http",
			"url": "https://api.githubcopilot.com/mcp"
		}
	}
}
```

VS Code also supports a user-profile configuration. Run `MCP: Open User
Configuration` from the Command Palette rather than looking for a fixed file
in the home directory. User-profile servers are available across workspaces;
workspace servers can be checked into source control when the configuration is
appropriate for the team.

VS Code provides IntelliSense and management actions for `mcp.json`. You can
also run `MCP: Add Server`, or manage installed servers from the Extensions
view and the Command Palette.

## Copilot CLI and Agent Host Configuration

For a configuration that the Agent Host reads natively, use the CLI format:

```json
{
	"mcpServers": {
		"playwright": {
			"command": "npx",
			"args": ["-y", "@microsoft/mcp-server-playwright"]
		},
		"github": {
			"type": "http",
			"url": "https://api.githubcopilot.com/mcp"
		}
	}
}
```

At workspace scope, put this configuration in `.mcp.json` at the repository
root. Copilot CLI can also discover `.github/mcp.json`, and a project-level
configuration may use the bare format in which each top-level key is a server
name.

At user scope, the default file is:

```text
~/.copilot/mcp-config.json
```

Set `COPILOT_HOME` when the Copilot home directory should be somewhere else.

## The Agent Host Detail in VS Code

A VS Code chat session can run on the Agent Host instead of the traditional
extension host. In that case, the Agent Host does not read `.vscode/mcp.json`
directly. VS Code forwards the configuration to it, except for servers that
need interactive input such as `${input:...}` variables.

This creates two practical choices:

1. Use `.vscode/mcp.json` when the configuration is specifically for VS Code.
2. Use the Agent Host format in a repository-level `.mcp.json` when the same
	 configuration should work natively in VS Code Agent Host and Copilot CLI.

The second option avoids maintaining two files, but it also means that tools
which only understand VS Code's `servers` format will not consume the file.

## Migrating a Configuration

The conversion is structural rather than a redesign of every server entry:

```diff
- .vscode/mcp.json
+ .mcp.json

- "servers": {
+ "mcpServers": {
```

Then review the server-specific transport fields. Local stdio servers usually
keep their `command`, `args`, and `env`; remote servers may differ in whether
the client supports HTTP, SSE, or another transport.

Do not copy secrets directly into either file. Use the client's input-variable
or environment-file mechanism, and review local server configurations before
trusting them: a local MCP server can run arbitrary code on the machine.

## Choosing a Format

Use this rule of thumb:

- **VS Code only:** `.vscode/mcp.json` with `servers`.
- **VS Code Agent Host and Copilot CLI:** repository `.mcp.json` with
	`mcpServers`.
- **Personal Agent Host configuration:** `~/.copilot/mcp-config.json`.
- **Shared team configuration:** commit the workspace file only after checking
	that it contains no credentials or machine-specific paths.

MCP is the shared protocol; configuration discovery is still client-specific.
When a server is missing, check the file location, the top-level key, the
transport fields, and whether the current session is using the extension host
or Agent Host.

## References

- [Add and manage MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/customization/mcp-servers) - documents VS Code workspace and user configuration, the `servers` schema, Agent Host forwarding behavior, the native `.mcp.json` and `~/.copilot/mcp-config.json` locations, server trust, and secret-handling guidance.
