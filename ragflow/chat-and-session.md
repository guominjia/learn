# Building Q&A with `ragflow_sdk`: Chat, Session, and `ask()`

If you are using `ragflow_sdk` and wondering how to send real Q&A requests, the key detail is this: chat objects manage sessions, but the actual question answering happens through `Session.ask()`.

This article walks through the full flow, including non-streaming and streaming responses, plus a quick look at what happens under the hood.

## Why Sessions Matter

In RAGFlow, you typically:

1. Get a chat assistant (`Chat` object)
2. Create a session from that assistant
3. Send questions through `session.ask()`

Without creating a session first, you cannot perform Q&A in this pattern.

## End-to-End Example

```python
from ragflow_sdk import RAGFlow

# 1) Initialize client
rag = RAGFlow(api_key="<YOUR_API_KEY>", base_url="http://<YOUR_BASE_URL>:9380")

# 2) Get a chat assistant
assistant = rag.list_chats(name="Miss R")[0]

# 3) Create a session
session = assistant.create_session()

# 4) Ask a question (non-streaming)
response = session.ask("What is retrieval-augmented generation?", stream=False)
print(response.content)
```

## `Session.ask()` Parameters

The `ask()` method supports:

- `question`: The user query (required)
- `stream`: Whether to return a stream (`True`) or a single final message (`False`)
- `**kwargs`: Additional variables passed through to the backend request

This makes it easy to inject runtime options (for example, custom style or business context fields).

## Streaming Response Example

Use streaming when you want token-by-token or chunk-by-chunk output in real time:

```python
for part in session.ask("Explain vector databases in simple terms.", stream=True):
    print(part.content, end="", flush=True)
```

This pattern is useful for CLI tools, chat UIs, and any interface where latency perception matters.

## Under the Hood: Which API Endpoint Is Called?

`Session.ask()` dispatches by session type:

- Chat session → `POST /chats/{chat_id}/completions`
- Agent session → `POST /agents/{agent_id}/completions`

Internally, the payload includes:

- `question`
- `stream`
- `session_id`
- any extra `kwargs`

The returned `Message` object typically includes:

- `content`: model response text
- `reference`: retrieved document chunks (when available)

## Practical Tips

- Create one session per conversation thread for cleaner context boundaries.
- Use `stream=False` for simple backend workflows and batch jobs.
- Use `stream=True` for interactive UX.
- Check `reference` when you need provenance or source transparency.

## Summary

`chat.py` gives you the session lifecycle, but `Session.ask()` is the true Q&A entry point. Once you follow the assistant → session → ask flow, you can switch between standard and streaming responses with one flag and pass custom runtime parameters as needed.

## References

- [ragflow_sdk/modules/session.py:36-47](https://github.com/infiniflow/ragflow/blob/ce71d878/sdk/python/ragflow_sdk/modules/session.py#L36-L47)
- <https://deepwiki.com/search/datasetragflowknowledgebase-kn_06736f77-7afb-4b2c-8afa-7885dffcf5f3>