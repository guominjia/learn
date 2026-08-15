---
layout: post
title: "LiteLLM Proxy: One model_list, Both OpenAI and Anthropic Endpoints"
date: 2026-08-15
categories: [ai, litellm]
tags: [litellm, anthropic, openai, claude, bedrock, gateway]
---

LiteLLM Proxy exposes the OpenAI-compatible `/v1/chat/completions` route, but it also natively implements the Anthropic `/v1/messages` route. That route can forward to **any** backend LiteLLM supports — OpenAI, Azure, Bedrock, Vertex — not just Anthropic models. In other words, a client that only speaks the Anthropic wire format can drive a non-Claude model.

## Why this matters

Claude Code always sends requests in Anthropic's format. Pointing it at LiteLLM Proxy lets it call, say, GPT-4o, while Claude Code itself never knows the backend changed.

```yaml
# config.yaml
model_list:
  - model_name: claude-opus-4-7      # name exposed to clients (select this in Claude Code)
    litellm_params:
      model: openai/gpt-4o           # actual backend model
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Start the proxy:

```bash
litellm --config config.yaml   # listens on http://0.0.0.0:4000
```

Point the client at it:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
claude --model claude-opus-4-7   # requests are forwarded to openai/gpt-4o
```

Or hit `/v1/messages` directly with curl:

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
```

The request body is Anthropic-shaped. LiteLLM translates messages, tool calls, and streaming into whatever the backend provider (OpenAI in this case) expects, then translates the response back to Anthropic's format on the way out.

## Both endpoints, same model_list, no extra config

`/v1/chat/completions` and `/v1/messages` are two protocol "shells" over the same backend model list — nothing extra needs to be configured to use both at once.

Using the `claude-opus-4-7` → `openai/gpt-4o` mapping above, both calls work against the same deployment:

```bash
# OpenAI format
curl -X POST http://0.0.0.0:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-7","messages":[{"role":"user","content":"hi"}]}'

# Anthropic format
curl -X POST http://0.0.0.0:4000/v1/messages \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
```

Both requests target the same `model_name` and land on the same backend model. Only the request/response wire format differs:

| Client | Endpoint | Backend can be |
|---|---|---|
| OpenAI SDK / `openai` library | `/v1/chat/completions` | OpenAI, Anthropic, Bedrock, Vertex, Azure |
| Anthropic SDK / Claude Code | `/v1/messages` | OpenAI, Anthropic, Bedrock, Vertex, Azure |

This is LiteLLM's core value as a gateway: one `model_list`, multiple client protocols, interchangeable backends.

## Reference

- [Anthropic API - LiteLLM docs](https://docs.litellm.ai/docs/anthropic_completion)