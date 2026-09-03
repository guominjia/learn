---
title: "Claude Skills in VS Code Copilot and Copilot CLI"
date: 2026-09-03
categories: [ai, agent]
tags: [copilot, claude, skills]
---

# Claude Skills in VS Code Copilot and Copilot CLI

Agent Skills use a simple distribution model: a skill is a directory containing a `SKILL.md` file and, optionally, scripts, examples, or other resources. The format is portable, but portability does not mean that every client searches the same directories.

This distinction matters when a skill works in VS Code but does not appear in Copilot CLI.

## The Short Version

| Location | GitHub Copilot in VS Code | Copilot CLI |
| --- | --- | --- |
| Project `.github/skills`, `.claude/skills`, or `.agents/skills` | Supported | Supported |
| Personal `~/.copilot/skills` | Supported | Supported |
| Personal `~/.claude/skills` | Supported | Not listed as a default location |
| Personal `~/.agents/skills` | Supported | Supported |

The important boundary is the last two rows. VS Code explicitly scans `~/.claude/skills` as a personal skill location. The Copilot CLI documentation lists `~/.copilot/skills` and `~/.agents/skills` for personal skills, but does not list `~/.claude/skills`.

## Repository Skills Are Portable

For a skill that belongs to a project or team, put the skill under one of the documented project directories:

```text
repository/
└── .claude/
	└── skills/
		└── release-check/
			└── SKILL.md
```

The directory name should normally match the `name` in the `SKILL.md` frontmatter. A minimal skill looks like this:

```markdown
---
name: release-check
description: Check release metadata and required validation steps.
---

Inspect the release metadata, run the required checks, and report failures.
```

Because `.claude/skills` is a documented project location for both VS Code Copilot and Copilot CLI, this is a reasonable place for a repository-owned skill. The skill remains part of the repository and can be reviewed, versioned, and shared with the project.

## Personal Skills Have Different Defaults

Personal skills are discovered from the user's home directory rather than from a repository. VS Code supports these three personal locations:

```text
~/.copilot/skills/
~/.claude/skills/
~/.agents/skills/
```

Copilot CLI documents only these two default personal locations:

```text
~/.copilot/skills/
~/.agents/skills/
```

Therefore, placing a skill in `~/.claude/skills` can make it available to VS Code without making it automatically available to Copilot CLI. The skill format is the same; the default discovery paths are different.

For a personal skill that should work in both clients without extra configuration, use `~/.copilot/skills` or `~/.agents/skills`.

## Adding Another Location in Copilot CLI

Copilot CLI provides a way to add an alternative skills directory. Inside an interactive session, use:

```text
/skills add ~/.claude/skills
```

The equivalent command-line form is:

```text
copilot skill add ~/.claude/skills
```

After adding a directory during a session, reload the skills and inspect the result:

```text
/skills reload
/skills list
/skills info release-check
```

This is a discovery configuration change, not a conversion of the skill. It tells the CLI to consider an additional location. If a skill was added while a session was already running, reloading avoids requiring a new session.

## Choosing the Location

Use the location according to ownership and scope:

1. Put team or repository behavior in `.github/skills`, `.claude/skills`, or `.agents/skills`.
2. Put a personal cross-client skill in `~/.copilot/skills` or `~/.agents/skills`.
3. Keep a VS Code-specific personal skill in `~/.claude/skills` when that compatibility path is useful.
4. Add `~/.claude/skills` to Copilot CLI explicitly when moving the same personal collection to the CLI.

The practical lesson is simple: check both the skill format and the client's discovery paths. A valid `SKILL.md` can still be invisible when it is stored outside the client's default locations.

## References

- [Use Agent Skills in VS Code](https://code.visualstudio.com/docs/agent-customization/agent-skills) - documents the Agent Skills format, project locations, VS Code personal locations including `~/.claude/skills`, and progressive loading.
- [About agent skills](https://docs.github.com/en/copilot/concepts/agents/about-agent-skills) - documents Copilot's supported project locations and personal locations, including the absence of `~/.claude/skills` from the personal list.
- [Adding agent skills for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills) - documents project and personal skill directories, `/skills add`, `copilot skill add`, reload, and inspection commands.
