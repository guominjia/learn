---
layout: post
title: "Claude Code Sessions: --continue, --resume, and Where Transcripts Live"
date: 2026-08-20
categories: [ai, claude-code]
tags: [claude-code, cli, sessions, resume, continue]
---

Claude Code writes every conversation to disk continuously, so a session survives closing the terminal, running `/clear`, or even a crash. Two flags cover almost every case.

## `--continue` vs `--resume`

| Command | Behavior |
|---|---|
| `claude --continue` / `claude -c` | Resumes the most recent session in the current directory (including sessions that added this directory with `/add-dir`) |
| `claude --resume` / `claude -r` | Opens an interactive session picker |
| `claude --resume <name-or-id>` | Resumes that session directly, no picker |
| `claude --from-pr <number>` | Opens the picker filtered to sessions linked to that pull request |
| `/resume` (inside a session) | Switches to a different conversation without leaving Claude Code |

```bash
claude --continue                # pick up where you left off, same directory
claude --resume                  # browse all sessions for this project/worktree
claude --resume auth-refactor    # jump straight to a named session
claude --resume abc123 --fork-session   # resume into a new session ID, leaving the original untouched
```

Passing a session ID works from any directory: Claude Code checks the current project and its git worktrees first, then every other project on the machine, so it can find a session that was started elsewhere or moved with `/cd`. Sessions started non‑interactively with `claude -p` or the Agent SDK don't show up in the picker, but can still be resumed by ID.

## Where transcripts are actually stored

Transcripts are JSONL files at:

```
~/.claude/projects/<project>/<session-id>.jsonl
```

- `<project>` is the working directory's path with every non-alphanumeric character replaced by `-` — not a hash. A hash suffix is only appended when the sanitized name would exceed 200 characters (Claude Code truncates it first).
- Each line is one JSON object: a message, a tool call, or a tool result. The format is internal and can change between releases, so don't parse it directly — use `/export`, or `claude -p --resume <id> --output-format json` for scripted access.
- On Windows, `~/.claude` resolves to `%USERPROFILE%\.claude`. Setting `CLAUDE_CONFIG_DIR` moves the whole tree elsewhere.
- Transcripts (and the rest of `~/.claude/projects`) are deleted after `cleanupPeriodDays` (default 30 days) unless referenced by auto memory, which is excluded from that sweep.
- Files here are plaintext, not encrypted — anything a tool reads or prints (including `.env` contents) ends up in the transcript.

```bash
# read the transcript path from a hook or statusline payload's `transcript_path` field,
# or send a scripted follow-up to an existing session:
claude -p --resume <session-id> --output-format json "summarize what we changed" | jq -r '.result'
```

Run `claude project purge <path>` to delete a project's transcripts, auto memory, tasks, and its entry in `~/.claude.json` — pass `--dry-run` first to preview what would be removed.

## References

- [Manage sessions](https://code.claude.com/docs/en/sessions) — official docs for `--continue`, `--resume`, the session picker, and the transcript storage path/format described above.
- [CLI reference](https://code.claude.com/docs/en/cli-reference) — flag definitions for `--continue`/`-c`, `--resume`/`-r`, `--fork-session`, and `--from-pr`.
- [Explore the .claude directory](https://code.claude.com/docs/en/claude-directory) — application-data table covering retention (`cleanupPeriodDays`), `CLAUDE_CONFIG_DIR`, and `claude project purge`.
- [Shared note](https://share.google/aimode/inzpSXfvJAyZSlaUh) — the source prompt for this post. This is a Google AI Mode share link; its page is client-rendered, so its content could not be fetched or verified for this article.