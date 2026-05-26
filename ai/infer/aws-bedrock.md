# Calling Claude on AWS Bedrock with OpenAI SDK

## Overview

AWS Bedrock hosts foundation models from multiple providers, including Anthropic's Claude. If you already have `AWS_REGION`, `AWS_BEARER_TOKEN_BEDROCK`, and `CLAUDE_CODE_USE_BEDROCK` configured, this guide explains how to call Claude through the OpenAI-compatible interface and clarifies what each variable does.

## Environment Variables

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Bedrock service region (e.g. `us-east-2`) |
| `AWS_BEARER_TOKEN_BEDROCK` | Bearer token for Bedrock authentication |
| `CLAUDE_CODE_USE_BEDROCK` | **Claude Code CLI only** — tells the `claude` CLI tool to route through Bedrock instead of `api.anthropic.com` |

> `CLAUDE_CODE_USE_BEDROCK` is exclusively used by the Claude Code CLI agent. It has no effect when you call models programmatically with the OpenAI SDK.

## Two Bedrock Endpoints

AWS Bedrock exposes two distinct endpoint styles. Understanding the difference is critical.

### `bedrock-mantle.*.api.aws` — OpenAI-Compatible

This is the **OpenAI-compatible** endpoint. It accepts the standard `/v1/chat/completions` format and uses **Bearer Token** authentication.

```
https://bedrock-mantle.us-east-2.api.aws/v1
```

- **Auth:** `Authorization: Bearer <token>`
- **Request format:** Same as OpenAI (`{"model": "...", "messages": [...]}`)
- **Compatible with OpenAI SDK:** Yes

### `bedrock-runtime.*.amazonaws.com` — Native Bedrock

This is the **native Bedrock** endpoint. It uses a completely different API format and requires **AWS Signature V4** authentication.

```
https://bedrock-runtime.us-east-2.amazonaws.com
```

- **Auth:** AWS SigV4 (HMAC-based request signing with `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`)
- **Request format:** `POST /model/{modelId}/invoke` with provider-specific body
- **Compatible with OpenAI SDK:** No

| Feature | `bedrock-mantle` | `bedrock-runtime` |
|---|---|---|
| Auth method | Bearer Token | AWS SigV4 Signature |
| API format | OpenAI-compatible | Bedrock-native |
| OpenAI SDK works? | Yes | No |
| SDK required | `openai` | `boto3` |

## Calling Claude with OpenAI SDK

Since `bedrock-mantle` is OpenAI-compatible, you can use the OpenAI Python SDK directly:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://bedrock-mantle.us-east-2.api.aws/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)

response = client.chat.completions.create(
    model="anthropic.claude-sonnet-4-20250514-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Listing Available Models

Not all models may be available in your region or account. Query the endpoint to check:

```python
models = client.models.list()
for m in models.data:
    print(m.id)
```

Or via `curl`:

```bash
curl -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" \
     https://bedrock-mantle.us-east-2.api.aws/v1/models
```

If Claude models are missing from the list, possible causes:

1. **Region availability** — Claude may not be enabled in `us-east-2`. Common regions: `us-east-1`, `us-west-2`.
2. **Model access not granted** — In the AWS Console → Bedrock → Model access, you must explicitly request access to Anthropic Claude models.
3. **Endpoint limitations** — The `bedrock-mantle` endpoint may only expose a subset of models.

## Calling Claude with boto3 (Native Endpoint)

If you need to use the native `bedrock-runtime` endpoint (e.g. when `bedrock-mantle` doesn't list Claude), use `boto3`:

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.invoke_model(
    modelId="anthropic.claude-sonnet-4-20250514-v1:0",
    contentType="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 1024,
    }),
)
result = json.loads(response["body"].read())
print(result)
```

This requires IAM credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) rather than a Bearer token.

## Using LiteLLM as a Proxy

LiteLLM can translate OpenAI-format requests to Bedrock-native calls, acting as a local proxy:

```bash
# Start the proxy
litellm --model bedrock/anthropic.claude-sonnet-4-20250514-v1:0
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-anything",  # any value works
)

response = client.chat.completions.create(
    model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

LiteLLM reads `AWS_REGION` and AWS credentials from environment variables automatically and handles SigV4 signing internally.

## Summary

```
┌─────────────────┐     Bearer Token      ┌──────────────────────────────────┐
│   OpenAI SDK    │ ───────────────────▶   │  bedrock-mantle.*.api.aws/v1   │
└─────────────────┘                        └──────────────────────────────────┘

┌─────────────────┐     SigV4 Signing      ┌──────────────────────────────────┐
│     boto3       │ ───────────────────▶   │  bedrock-runtime.*.amazonaws.com │
└─────────────────┘                        └──────────────────────────────────┘

┌─────────────────┐     OpenAI format      ┌─────────┐   SigV4   ┌──────────┐
│   OpenAI SDK    │ ───────────────────▶   │ LiteLLM │ ────────▶ │ Bedrock  │
└─────────────────┘                        └─────────┘           └──────────┘
```

- **Have a Bearer Token?** → Use `bedrock-mantle` with OpenAI SDK directly.
- **Have IAM credentials?** → Use `boto3` or LiteLLM as a proxy.
- **Using Claude Code CLI?** → Set `CLAUDE_CODE_USE_BEDROCK=1` and it handles everything.
