# Upload Is Not Enough: Understanding RAGFlow's Two-Step Document Processing Pipeline

When integrating RAGFlow into an application, a common mistake is assuming that uploading a document automatically makes it searchable. It does not. RAGFlow deliberately separates document ingestion into two distinct phases — upload and parsing — and understanding why this design exists makes the entire SDK much easier to work with.

---

## The Misconception

If you call `Dataset.upload_documents()` and then immediately try to query the dataset, you will likely get no results. The method name implies the job is done, but from RAGFlow's perspective, the document has only arrived on disk. It has not been chunked, embedded, or indexed yet — it is not yet part of the retrieval system.

---

## Phase 1: Uploading Documents

`upload_documents()` is a straightforward HTTP multipart upload. It sends the raw file bytes to the RAGFlow backend and returns a list of `Document` objects representing what was accepted.

### Source Implementation

```python
def upload_documents(self, document_list: list[dict]):
    url = f"/datasets/{self.id}/documents"
    files = [("file", (ele["display_name"], ele["blob"])) for ele in document_list]
    res = self.post(path=url, json=None, files=files)
    res = res.json()
    if res.get("code") == 0:
        doc_list = []
        for doc in res["data"]:
            document = Document(self.rag, doc)
            doc_list.append(document)
        return doc_list
    raise Exception(res.get("message"))
```

The returned `Document` objects have a `run` attribute that starts as `"UNSTART"`. This is your signal that the document is sitting in storage, waiting to be processed.

**What upload does:**
- Transfers file bytes to the RAGFlow server
- Registers the document in the dataset's metadata
- Returns `Document` objects with status `"UNSTART"`

**What upload does NOT do:**
- Chunk the document into retrievable segments
- Build text or vector embeddings
- Create any searchable index

---

## Phase 2: Parsing Documents

Parsing is where the real work happens. RAGFlow's file parsing has two responsibilities:

1. **Chunking** — splitting the document into segments based on the dataset's configured `chunk_method` (e.g., by paragraph, by sentence, by fixed token window)
2. **Indexing** — building both embedding-based (vector) and full-text (keyword) indexes on those chunks

RAGFlow exposes two parsing methods depending on whether you need to block and inspect results.

### `async_parse_documents()` — Fire and Forget

```python
DataSet.async_parse_documents(document_ids: list[str]) -> None
```

Initiates parsing for the specified document IDs and returns immediately. Use this when you want to kick off processing in the background and check status later via the `run` attribute on each `Document`.

```python
dataset.async_parse_documents(doc_ids)
print("Parsing initiated.")
```

The `run` attribute transitions through these states:

| Value | Meaning |
|---|---|
| `"UNSTART"` | Not yet processed |
| `"RUNNING"` | Actively being parsed |
| `"DONE"` | Successfully chunked and indexed |
| `"FAIL"` | Parsing encountered an error |
| `"CANCEL"` | Parsing was cancelled |

### `parse_documents()` — Block Until Complete

```python
DataSet.parse_documents(document_ids: list[str]) -> list[tuple[str, str, int, int]]
```

This method wraps `async_parse_documents()` and blocks until all documents finish processing. It returns a list of tuples with detailed results per document:

```python
(document_id: str, status: str, chunk_count: int, token_count: int)
```

`chunk_count` tells you how many retrievable segments were created. `token_count` gives you a sense of the indexing cost. If a `KeyboardInterrupt` occurs during the wait, all pending tasks are cancelled gracefully — a thoughtful detail for interactive scripts.

---

## The Complete Two-Step Workflow

Putting both phases together, a minimal working pipeline looks like this:

```python
from ragflow_sdk import RAGFlow

rag = RAGFlow(api_key="<YOUR_API_KEY>", base_url="http://<YOUR_BASE_URL>:9380")
dataset = rag.create_dataset(name="kb_name")

# Phase 1: Upload — files land on disk, not yet searchable
documents = [
    {'display_name': 'report.txt', 'blob': open('./report.txt', 'rb').read()},
    {'display_name': 'notes.txt',  'blob': open('./notes.txt',  'rb').read()},
]
dataset.upload_documents(documents)

# Phase 2: Parse — chunk, embed, and index
docs = dataset.list_documents(keywords="")
doc_ids = [doc.id for doc in docs]

try:
    results = dataset.parse_documents(doc_ids)
    for doc_id, status, chunk_count, token_count in results:
        print(f"{doc_id}: {status} | chunks={chunk_count} | tokens={token_count}")
except KeyboardInterrupt:
    print("Cancelled.")
except Exception as e:
    print(f"Error: {e}")
```

---

## Why the Separation Exists

This two-phase design is intentional and has practical advantages.

**Batch efficiency.** You can upload a large number of files first — across multiple calls if needed — then trigger parsing in a single batch operation. This avoids the overhead of spinning up the embedding pipeline for each file individually.

**Flexible scheduling.** Parsing is computationally expensive (embedding generation, index writes). Decoupling upload from parsing lets you schedule processing during off-peak hours without blocking the upload path.

**Selective reprocessing.** If the dataset's chunking strategy or embedding model changes, you can re-parse specific documents by ID without re-uploading the raw files.

**Progress observability.** Because parsing is a separate, trackable operation with explicit `run` states, you get fine-grained visibility into what is processing, what failed, and what is ready.

---

## Key Takeaways

| Phase | Method | Database / Index Impact |
|---|---|---|
| Upload | `upload_documents()` | File stored; metadata recorded; nothing indexed |
| Parse (async) | `async_parse_documents()` | Chunks created; embeddings built; indexes updated |
| Parse (blocking) | `parse_documents()` | Same as above; returns per-document results |

Never assume a document is retrievable after upload alone. The document is only part of the retrieval system after parsing completes with status `"DONE"`. Build your integration logic around this two-step reality and you will avoid a class of subtle, hard-to-debug search failures.

---

## References

- <https://deepwiki.com/search/ragflowsdkdatasetuploaddocumen_ce970cf2-8fac-4a1a-8655-5507f91d5fb3?mode=fast>

## Opens

1. `async_parse_documents` will parse all documents in datasets? can i choose specific one to parse?