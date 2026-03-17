# LiteLLM: More Than Just a Router

> **Published:** March 17, 2026

## Introduction

LiteLLM is often misunderstood as a simple router that forwards requests to different LLM backends. In reality, it is a **full-featured LLM infrastructure layer** that covers API unification, load balancing, fallback handling, caching, cost tracking, and a production-ready proxy server.

---

## Background

When working with multiple LLM providers — OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, Ollama — each one exposes a different API shape. You end up writing provider-specific code, managing multiple API keys, and duplicating retry/fallback logic across your codebase.

LiteLLM solves this by sitting in front of all those providers and exposing a single, consistent interface.

---

## What LiteLLM Actually Does

### 1. Unified Interface (Adapter Layer)

LiteLLM translates every supported model into the **OpenAI `chat.completions` format**. This means you write your code once and can swap models by changing a single string.

```python
import litellm

# OpenAI
response = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "Hello"}])

# Anthropic — same code, different model string
response = litellm.completion(model="claude-3-5-sonnet-20241022", messages=[{"role": "user", "content": "Hello"}])

# Google Gemini — same code again
response = litellm.completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "Hello"}])
```

Supported providers include OpenAI, Anthropic, Azure, Google (Gemini/Vertex), Cohere, Mistral, Ollama, Hugging Face, and 100+ more.

---

### 2. Router with Load Balancing

When you need to scale across multiple deployments of the same model (e.g., multiple Azure regions, multiple API keys), LiteLLM's `Router` handles it automatically.

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o-us",
                "api_base": "https://us.openai.azure.com",
                "api_key": "...",
            },
        },
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o-eu",
                "api_base": "https://eu.openai.azure.com",
                "api_key": "...",
            },
        },
    ],
    routing_strategy="least-busy",  # or "latency-based", "cost-based", "simple-shuffle"
)

response = router.completion(model="gpt-4o", messages=[...])
```

Available routing strategies:

| Strategy | Description |
|---|---|
| `simple-shuffle` | Round-robin across deployments |
| `least-busy` | Route to the deployment with fewest active requests |
| `latency-based` | Route to the historically fastest deployment |
| `cost-based` | Route to the cheapest option |

---

### 3. Fallback and Retry

A production system must handle provider outages or rate limits gracefully. LiteLLM provides first-class fallback support:

```python
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this document..."}],
    fallbacks=["claude-3-5-sonnet-20241022", "gemini/gemini-2.0-flash"],
    num_retries=3,
    timeout=30,
)
```

If `gpt-4o` fails (rate limit, timeout, server error), LiteLLM automatically retries and then cascades through the fallback list — no try/except boilerplate in your application code.

---

### 4. LiteLLM Proxy — A Standalone OpenAI-Compatible Server

LiteLLM ships with a **proxy server** that you can deploy as a microservice. Any OpenAI-compatible client (LangChain, OpenAI SDK, curl, etc.) can point to it without modification.

```bash
# Install and launch
pip install 'litellm[proxy]'
litellm --model gpt-4o --port 4000
```

```bash
# Call it exactly like OpenAI
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

The proxy is configured via a `config.yaml`:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_base: https://my-azure.openai.azure.com
      api_key: os.environ/AZURE_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: sk-my-master-key   # auth header for proxy clients
```

---

### 5. Cost Tracking

LiteLLM tracks token usage and computes cost automatically for every call:

```python
response = litellm.completion(model="gpt-4o", messages=[...])

print(litellm.completion_cost(response))
# 0.000245  (USD)
```

Over the proxy, cost is aggregated per virtual key, per user, and per team — useful for internal chargeback and budget enforcement.

---

### 6. Caching

Identical requests can be served from cache, reducing latency and cost:

```python
import litellm
from litellm.caching import Cache

litellm.cache = Cache(type="redis", host="localhost", port=6379)

# First call hits the model
r1 = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "What is 2+2?"}])

# Second identical call is served from Redis cache instantly
r2 = litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "What is 2+2?"}])
```

Supported cache backends: `redis`, `redis-semantic` (embedding-based similarity), `s3`, and in-memory.

---

### 7. Observability and Logging

LiteLLM integrates with major observability platforms out of the box:

```python
litellm.success_callback = ["langfuse", "helicone", "datadog"]
```

Every call is logged with model name, latency, token counts, cost, and custom metadata — with zero boilerplate.

---

## Architecture Summary

```
Your App / Any OpenAI-compatible client
          │
          ▼
  ┌───────────────────┐
  │   LiteLLM Proxy   │  ← auth, rate limiting, budget enforcement
  │   (FastAPI server) │
  └────────┬──────────┘
           │
  ┌────────▼──────────┐
  │   LiteLLM Router  │  ← load balancing, fallback, retries
  └────────┬──────────┘
           │
  ┌────────▼──────────┐
  │  Adapter / SDK    │  ← unified OpenAI-format translation
  └────────┬──────────┘
           │
  ┌────────▼─────────────────────────────────────┐
  │  OpenAI │ Anthropic │ Azure │ Gemini │ Ollama │ ...
  └──────────────────────────────────────────────┘
```

---

## LiteLLM vs. A Simple Router

| Capability | Simple Router | LiteLLM |
|---|:---:|:---:|
| Forward request to backend | ✅ | ✅ |
| Unified API format | ❌ | ✅ |
| Load balancing strategies | ❌ | ✅ |
| Automatic fallback | ❌ | ✅ |
| Cost tracking per call | ❌ | ✅ |
| Response caching | ❌ | ✅ |
| Streaming support | ❌ | ✅ |
| Observability integrations | ❌ | ✅ |
| Standalone proxy server | ❌ | ✅ |
| Virtual key management | ❌ | ✅ |

---

## When to Use LiteLLM

- You need **model-agnostic code** that can switch providers without refactoring.
- You operate multiple LLM deployments and need **load balancing** or **failover**.
- You want **cost visibility** across teams or projects.
- You need a **drop-in OpenAI-compatible gateway** in front of your private or on-prem models.
- You're building a **multi-tenant platform** where different users/teams should use different models or have different budgets.

---

## References

- [LiteLLM Docs](https://docs.litellm.ai)
- [GitHub: BerriAI/litellm](https://github.com/BerriAI/litellm)
- [Supported Providers](https://docs.litellm.ai/docs/providers)
- [LiteLLM Proxy Guide](https://docs.litellm.ai/docs/proxy/quick_start)
