---
layout: post
title: "Skills and MCP Config: Where VS Code and Copilot CLI Actually Look"
date: 2026-08-21
categories: [ai, copilot]
tags: [copilot, vscode, copilot-cli, claude-code, skills, mcp]
---

VS Code's Copilot Chat and GitHub Copilot CLI both implement the [Agent Skills](https://agentskills.io/) open standard and both support MCP servers, but they scan different directories for each. Mixing up the two causes skills or MCP servers that work in one to silently go missing in the other.

## Skill directories

| Scope | VS Code | Copilot CLI |
|---|---|---|
| Project (in repo) | `.github/skills/`, `.claude/skills/`, `.agents/skills/` | `.github/skills/`, `.claude/skills/`, `.agents/skills/` |
| Personal (home dir) | `~/.copilot/skills/`, `~/.claude/skills/`, `~/.agents/skills/` | `~/.copilot/skills/`, `~/.agents/skills/` |

Project-level directories are identical between the two. The gap is personal skills: VS Code additionally scans `~/.claude/skills/` (a compatibility scan for Claude Code's personal skill directory), while Copilot CLI only recognizes `~/.copilot/skills/` and `~/.agents/skills/`. A skill dropped only into `~/.claude/skills/` loads in VS Code but is invisible to the CLI.

Workaround: point the CLI at the directory explicitly, either interactively with `/skills add` or from the shell with `copilot skill add <DIRECTORY>`. Repo-level `.claude/skills/` needs no such workaround since both tools already read it.

VS Code also lets you add extra project skill locations via the `chat.agentSkillsLocations` setting.

## MCP server config files

| Scope | VS Code | Copilot CLI |
|---|---|---|
| Workspace/project | `.vscode/mcp.json` (`servers` key) | `.mcp.json` (walked up from cwd to repo root) or `.github/mcp.json` (`mcpServers` key or bare top-level format) |
| User/personal | User profile `mcp.json`, opened via **MCP: Open User Configuration** | `~/.copilot/mcp-config.json` (`mcpServers` key) |

The two ecosystems use different top-level JSON keys — `servers` for `.vscode/mcp.json` vs. `mcpServers` (or a bare per-server-name format) for Copilot CLI's files — so a file written for one isn't valid for the other without editing the key name.

There's also a runtime wrinkle inside VS Code itself: when a chat session runs on [Agent Host](https://code.visualstudio.com/docs/agents/concepts/agent-host) rather than the classic extension host, VS Code doesn't hand `.vscode/mcp.json` to the Agent Host directly — it forwards the resolved server config instead, and skips any server that needs interactive `${input:...}` variables. For a config that both Agent Host and Copilot CLI can read natively, use a workspace `.mcp.json` or a user `~/.copilot/mcp-config.json` file instead of `.vscode/mcp.json`.

Copilot CLI precedence when multiple files apply:

1. `.mcp.json` beats `.github/mcp.json` if both exist in the same directory.
2. A definition in a file closer to the current working directory beats one further up.
3. Any project-level file beats `~/.copilot/mcp-config.json`.
4. Project-level files load only after the directory has been trusted; in prompt mode (`copilot -p`) they're skipped in untrusted directories unless `GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP=true`.

```jsonc
// .vscode/mcp.json — VS Code only, "servers" key
{
  "servers": {
    "playwright": { "command": "npx", "args": ["-y", "@microsoft/mcp-server-playwright"] }
  }
}

// .mcp.json — Copilot CLI (and portable to VS Code Agent Host), "mcpServers" key
{
  "mcpServers": {
    "playwright": { "type": "local", "command": "npx", "args": ["@playwright/mcp@latest"] }
  }
}
```

## Bottom line

Repo-level directories (`.github/skills`, `.claude/skills`, `.agents/skills` for skills; `.mcp.json`/`.github/mcp.json` for MCP) work across both tools without changes. Personal, home-directory configuration does not automatically carry over: `~/.claude/skills` needs `copilot skill add` to reach the CLI, and `.vscode/mcp.json`'s `servers` key needs converting to `mcpServers` before Copilot CLI (or the Agent Host, for full portability) will read it.

## References

- [Use Agent Skills in VS Code](https://code.visualstudio.com/docs/agent-customization/agent-skills) — lists VS Code's project (`.github/skills/`, `.claude/skills/`, `.agents/skills/`) and personal (`~/.copilot/skills/`, `~/.claude/skills/`, `~/.agents/skills/`) skill directories, and the `chat.agentSkillsLocations` setting.
- [About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) — confirms Copilot's supported skill locations across surfaces.
- [Adding agent skills for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills) — CLI project skills (`.github/skills`, `.claude/skills`, `.agents/skills`) vs. personal skills (`~/.copilot/skills`, `~/.agents/skills`, no `~/.claude/skills`), plus the `/skills add` and `copilot skill add` commands.
- [Add and manage MCP servers in VS Code](https://code.visualstudio.com/docs/copilot/customization/mcp-servers) — `.vscode/mcp.json` (`servers` key), user-profile `mcp.json`, and the Agent Host forwarding behavior with the `.mcp.json`/`~/.copilot/mcp-config.json` portability recommendation.
- [Adding MCP servers for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers) — `~/.copilot/mcp-config.json` (`mcpServers` key), project-level `.mcp.json`/`.github/mcp.json` discovery and precedence, trust rules, and confirmation that `.vscode/mcp.json`'s `servers` key is not read by the CLI.
