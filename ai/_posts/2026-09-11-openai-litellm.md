---
layout: post
title: "OpenAI vs LiteLLM: API Cheat Sheet"
date: 2026-09-11
categories: [ai]
tags: [openai, litellm, python]
---

The following table maps common OpenAI Python client calls to LiteLLM SDK calls.

The LiteLLM model name usually includes the provider prefix, for example `openai/gpt-5.5`.

## API mapping

### `client.responses.create`

OpenAI:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5.5",
    input="Say hello.",
)
print(response.output_text)
```

LiteLLM:

```python
import litellm

response = litellm.responses(
    model="openai/gpt-5.5",
    input="Say hello.",
)
print(response)
```

### `client.models.list`

OpenAI:

```python
models = client.models.list()
print([model.id for model in models.data])
```

LiteLLM SDK has no universal equivalent because the available models belong to different providers. With LiteLLM Proxy, query its OpenAI-compatible `/models` endpoint using the OpenAI client:

```python
proxy_client = OpenAI(
    api_key="your-proxy-key",
    base_url="http://localhost:4000/v1",
)
models = proxy_client.models.list()
print([model.id for model in models.data])
```

### `client.embeddings.create`

OpenAI:

```python
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Text to embed",
)
print(len(embedding_response.data[0].embedding))
```

LiteLLM:

```python
embedding_response = litellm.embedding(
    model="openai/text-embedding-3-small",
    input=["Text to embed"],
)
print(len(embedding_response.data[0]["embedding"]))
```

### `client.chat.completions.create`

OpenAI:

```python
completion = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello."},
    ],
)
print(completion.choices[0].message.content)
```

LiteLLM:

```python
response = litellm.completion(
    model="openai/gpt-5.5",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hello."},
    ],
)
print(response.choices[0].message.content)
```

The request shape is similar, but provider support and provider-specific parameters can differ. Check the selected model and provider before assuming complete compatibility.

## References

- [OpenAI Python library](https://github.com/openai/openai-python) - official Python client examples for Responses, Chat Completions, models, and embeddings.
- [OpenAI model list API](https://developers.openai.com/api/reference/resources/models/methods/list) - OpenAI model listing endpoint.
- [OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create) - OpenAI embedding endpoint.
- [LiteLLM Responses API](https://docs.litellm.ai/docs/response_api) - `litellm.responses()` and the OpenAI-compatible Responses format.
- [LiteLLM Chat Completions](https://docs.litellm.ai/docs/completion) - `litellm.completion()` and its common response format.
- [LiteLLM embeddings](https://docs.litellm.ai/docs/embedding/supported_embedding) - `litellm.embedding()` and supported embedding providers.
- [LiteLLM Proxy quick start](https://docs.litellm.ai/docs/proxy/quick_start) - the Proxy `/models` endpoint.