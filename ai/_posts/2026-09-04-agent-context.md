---
layout: post
title: "How an AI Agent Keeps Context Without Keeping Everything"
date: 2026-09-04
categories: [ai, agents]
tags: [ai-agents, context-window, memory, vscode]
---

An agent does not usually put the entire history of a project or conversation into every request. That would be expensive, slow, and eventually impossible: the model has a finite context window.

A better mental model is a set of layers. The exact implementation differs between products, but the division of responsibilities is useful when an agent appears to “remember” something.

## 1. Recent conversation context

The current conversation is the most immediate source of context. Recent prompts, responses, tool results, and decisions are normally kept in the active session so the agent can continue the task without making the user repeat everything.

Older turns do not have to remain verbatim. When the context window becomes full, an agent can compact earlier conversation into a shorter summary and continue with the resulting context. A good summary preserves decisions, changed files, constraints, and unresolved questions rather than every sentence.

This explains why a long-running session can still feel continuous even though its complete transcript is no longer being sent to the model on every turn.

## 2. Persistent memory

Memory is different from conversation history. It is a small store of reusable facts, such as:

- a repository convention;
- an architectural decision;
- a preferred coding style; or
- a recurring workflow constraint.

The purpose is not to archive the chat. It is to avoid re-explaining stable information in future interactions. Memory should also be scoped: a repository fact belongs to that repository, while a personal preference belongs to the user. Stale memories need validation or expiration; otherwise, a once-correct rule can become a source of mistakes.

For example, GitHub Copilot Memory stores repository-level facts and user-level preferences, associates memories with supporting citations, and validates repository facts against the current branch before using them. This is one concrete implementation of the broader memory layer, not a definition of how every agent works.

## 3. Workspace context on demand

The workspace is an external source of truth. The agent can search and read it when the task requires more detail instead of keeping the whole repository in the conversation.

Depending on the tool, the agent may receive context from:

- the active file or selection;
- explicitly referenced files, folders, or symbols;
- an indexed codebase search;
- terminal output;
- an issue, pull request, or web page; and
- tools that inspect or modify the workspace.

This is why “the agent has access to the repository” does not mean “the entire repository is already in the prompt.” Access is a capability. The relevant files still have to be selected, searched, or read for a particular request.

## A compact model

The process can be pictured like this:

```text
new request
	|
	+--> recent session context
	|
	+--> compacted summary of older turns
	|
	+--> relevant persistent memories
	|
	+--> workspace and tool results retrieved for this task
	|
	+--> model request
```

These layers are not equally authoritative. The current repository contents should win over an old summary or memory. A newly read configuration file should win over a remembered convention that the project has since changed. When the agent makes a decision, that decision may then become part of the recent context, a later summary, or, if it is stable and supported, persistent memory.

## Why the context can feel smaller

It is normal to see only a filtered working context rather than an infinite transcript. Systems have to spend their context budget on the material most likely to affect the next answer.

The practical consequence is simple:

1. Keep important decisions explicit in the current task.
2. Put durable project rules in documentation or configuration where tools can find them.
3. Treat memory as a convenience, not as the only source of truth.
4. When an answer seems to miss an important detail, reference the relevant file, symbol, or previous decision directly.

The agent is not remembering everything. It is combining a recent working set, compressed history, selected durable facts, and material it can retrieve from the workspace when needed.

## References

- [Manage agent sessions in VS Code](https://code.visualstudio.com/docs/agents/run/sessions/manage-sessions) — documents per-session context windows, automatic conversation compaction, and the distinction between continuing a session and starting a new one.
- [Add context to chat](https://code.visualstudio.com/docs/copilot/chat/copilot-chat-context) — documents automatic workspace indexing and explicit file, folder, symbol, codebase, terminal, and web context.
- [About GitHub Copilot Memory](https://docs.github.com/en/copilot/concepts/agents/copilot-memory) — documents repository-level facts, user-level preferences, citations, validation, and retention for one concrete memory implementation.
