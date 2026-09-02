---
title: "How Coding Agents Use System Prompts and Tools"
date: 2026-08-28
categories: [ai, agent]
tags: [copilot, claude, cline]
---

Coding agents often appear to have a simple interface: enter a request and receive an answer or a code change. Behind that interface, the agent usually combines instructions, conversation context, and a set of tools that can inspect or modify the working environment.

The exact implementation differs between Claude Code, GitHub Copilot, Cline, and other products. The general model, however, is useful for understanding why an agent sometimes reads a file or runs a command and sometimes answers immediately.

## What the Model Receives

The text entered by the user is only one part of an agent request. A coding-agent harness may add:

- System instructions that define the agent's role, constraints, and workflow
- Project instructions supplied by the user or repository
- Conversation history and selected source code
- Tool definitions, including names, descriptions, and input schemas

These instructions are assembled by the client, extension, or service that runs the agent. They are not necessarily shown in the normal chat transcript. Some products provide diagnostic views or logs that expose parts of the assembled request, but their visibility depends on the product, configuration, and permissions.

This is why the visible prompt is not a complete description of the agent's behavior. The surrounding harness can add context and constraints before the model responds.

## Tools Are Structured Calls

A skill is usually implemented as a tool or function. The tool definition describes what it does and which arguments it accepts. For example, a file-reading tool might define a path and a line range in its input schema.

The model does not execute that function directly. A typical client-side tool loop looks like this:

```text
User request
	-> model chooses a response or emits a tool call
	-> agent runtime validates and executes the call
	-> runtime sends the tool result back to the model
	-> model continues or produces the final response
```

Anthropic documents this exchange as a `tool_use` block followed by a `tool_result`. The same broad pattern is used by many tool-enabled agent systems, although the names and execution boundaries vary.

## Why an Agent May Skip a Tool

Tool use is commonly conditional. With an automatic tool-choice policy, the model decides whether the request matches a tool and whether the existing context is sufficient.

An agent may answer directly when:

- The question is conversational or concerns stable knowledge
- The required information is already present in the context
- No available tool clearly matches the request
- The tool description or required arguments are ambiguous
- The agent's instructions allow a direct answer

This does not prove that the model ignored the tool definition. It may have judged that calling the tool was unnecessary or that the request did not satisfy the tool's contract. Long context, competing instructions, and incomplete user requests can also make that judgment less reliable.

Some APIs allow the application to influence this boundary. A client can leave tool choice automatic, request that a particular tool be used, or prohibit tool calls. A prompt such as "investigate with the available tools before answering" can increase tool use, but it is not equivalent to an API-level guarantee unless the product supports an explicit forced-tool policy.

## How to Write Better Agent Requests

When fresh information or an environment action is required, make that requirement explicit:

```text
Inspect the relevant source files first. Do not infer the current behavior from the prompt alone.
Then explain the cause and propose the smallest fix.
```

A good request identifies:

- The outcome you want
- The information the agent must inspect
- Any tool or data source that is required
- Important constraints, such as read-only investigation

For example, "Review this error" leaves the scope unclear. "Inspect the relevant logs and source code, then identify the first failing operation" gives the agent a concrete investigation target without depending on a product-specific tool name.

## A Practical Mental Model

The most reliable mental model is not "the prompt contains a hidden script that always runs." It is:

1. The harness assembles instructions, context, and tool definitions.
2. The model selects either a direct response or a structured tool call.
3. The runtime executes permitted calls and returns their results.
4. The model uses those results to continue the task or finish the response.

Claude Code, GitHub Copilot, Cline, and similar products expose different controls and diagnostics around this loop. Understanding the loop helps explain their behavior without assuming that one product's prompt format or logging behavior applies to all of them.

## References

- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) - documents tool definitions, `tool_use` and `tool_result` blocks, automatic tool choice, and forced tool selection.
- [Use chat in VS Code](https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode) - documents agent context selection, tools, and the Chat Debug view for inspecting prompts and tool payloads.