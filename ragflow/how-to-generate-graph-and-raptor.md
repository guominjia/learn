# How RAGFlow Generates Knowledge Graphs and RAPTOR Hierarchies

RAGFlow is an open-source RAG engine that goes beyond simple vector similarity search. Two of its most powerful features are **GraphRAG** (knowledge graph construction) and **RAPTOR** (recursive abstractive processing for tree-organized retrieval). This post walks through how these features work under the hood — from the API endpoint all the way to the task executor.

---

## Background: Not All Tasks Are "dataflow"

A common misconception when exploring RAGFlow's source code is that the web interface always creates tasks of type `"dataflow"`. In reality, the system supports several distinct task types:

| Task Type   | Purpose                                  |
|-------------|------------------------------------------|
| `dataflow`  | Pipeline-based document processing       |
| `graphrag`  | Knowledge graph construction             |
| `raptor`    | Hierarchical summarization               |
| `mindmap`   | Mind map generation                      |
| `memory`    | Memory-related processing                |

The mapping from task type to internal pipeline type is defined in `rag/svr/task_executor.py`:

```python
TASK_TYPE_TO_PIPELINE_TASK_TYPE = {
    "dataflow": PipelineTaskType.PARSE,
    "raptor": PipelineTaskType.RAPTOR,
    "graphrag": PipelineTaskType.GRAPH_RAG,
    "mindmap": PipelineTaskType.MINDMAP,
    "memory": PipelineTaskType.MEMORY,
}
```

---

## The Entry Point: API Endpoints

GraphRAG and RAPTOR tasks are not triggered during normal document parsing. They are initiated explicitly through dedicated REST API endpoints defined in `api/apps/kb_app.py`:

- `POST /run_graphrag` — triggers knowledge graph construction for a knowledge base
- `POST /run_raptor` — triggers hierarchical summarization for a knowledge base

### GraphRAG Endpoint

```python
@manager.route("/run_graphrag", methods=["POST"])
@login_required
async def run_graphrag():
    req = await get_request_json()
    kb_id = req.get("kb_id", "")

    ok, kb = KnowledgebaseService.get_by_id(kb_id)

    # Prevent duplicate runs — check if a task is already in progress
    task_id = kb.graphrag_task_id
    if task_id:
        ok, task = TaskService.get_by_id(task_id)
        if task and task.progress not in [-1, 1]:
            return get_error_data_result(
                message=f"Task {task_id} in progress. A Graph Task is already running."
            )

    # Collect all documents in the knowledge base
    documents, _ = DocumentService.get_by_kb_id(kb_id=kb_id, ...)
    sample_document = documents[0]
    document_ids = [doc["id"] for doc in documents]

    # Queue the task
    task_id = queue_raptor_o_graphrag_tasks(
        sample_doc_id=sample_document,
        ty="graphrag",
        priority=0,
        fake_doc_id=GRAPH_RAPTOR_FAKE_DOC_ID,
        doc_ids=list(document_ids),
    )

    KnowledgebaseService.update_by_id(kb.id, {"graphrag_task_id": task_id})
    return get_json_result(data={"graphrag_task_id": task_id})
```

The RAPTOR endpoint (`run_raptor`) mirrors this structure exactly, just with `ty="raptor"` and `raptor_task_id`.

Key observations:
- Both endpoints operate at the **knowledge base level**, not the document level.
- A **fake document ID** (`GRAPH_RAPTOR_FAKE_DOC_ID`) is used to bypass per-document task restrictions, since these tasks span all documents in the KB.
- Progress tracking is built in — the task ID is stored back on the knowledge base record.

---

## Queuing the Task: `queue_raptor_o_graphrag_tasks`

Both endpoints call the same helper function in `api/db/services/document_service.py`:

```python
def queue_raptor_o_graphrag_tasks(sample_doc_id, ty, priority, fake_doc_id="", doc_ids=[]):
    assert ty in ["graphrag", "raptor", "mindmap"]

    chunking_config = DocumentService.get_chunking_config(sample_doc_id["id"])
    hasher = xxhash.xxh64()
    for field in sorted(chunking_config.keys()):
        hasher.update(str(chunking_config[field]).encode("utf-8"))

    task = {
        "id": get_uuid(),
        "doc_id": sample_doc_id["id"],
        "from_page": 100000000,   # sentinel value — not a real page range
        "to_page":   100000000,
        "task_type": ty,           # "graphrag" or "raptor"
        "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task " + ty,
        "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for field in ["doc_id", "from_page", "to_page"]:
        hasher.update(str(task.get(field, "")).encode("utf-8"))
    hasher.update(ty.encode("utf-8"))
    task["digest"] = hasher.hexdigest()

    bulk_insert_into_db(Task, [task], True)

    task["doc_id"] = fake_doc_id
    task["doc_ids"] = doc_ids

    DocumentService.begin2parse(sample_doc_id["id"], keep_progress=True)
    REDIS_CONN.queue_product(settings.get_svr_queue_name(priority), message=task)
    return task["id"]
```

Notable design choices:
- `from_page = 100000000` and `to_page = 100000000` are sentinel values indicating this is not a chunk-level task.
- A content **digest (xxhash)** is computed from the chunking config and task fields — this enables deduplication and change detection.
- The task is pushed onto a **Redis queue** for asynchronous processing by the task executor.
- `doc_ids` carries the full list of document IDs in the KB, since GraphRAG and RAPTOR need to reason across all documents.


---

## The Task Executor: Processing GraphRAG and RAPTOR

The worker process picks tasks off the Redis queue and dispatches them in `do_handle_task()` inside `rag/svr/task_executor.py`. Here is how each type is handled:

### RAPTOR Execution

```python
if task_type == "raptor":
    ok, kb = KnowledgebaseService.get_by_id(task_dataset_id)

    kb_parser_config = kb.parser_config
    # Apply default RAPTOR config if not already set
    if not kb_parser_config.get("raptor", {}).get("use_raptor", False):
        kb_parser_config.update({
            "raptor": {
                "use_raptor": True,
                "prompt": "Please summarize the following paragraphs...\n{cluster_content}",
                "max_token": 256,
                "threshold": 0.1,
                "max_cluster": 64,
                "random_seed": 0,
                "scope": "file",
            }
        })
        KnowledgebaseService.update_by_id(kb.id, {"parser_config": kb_parser_config})

    # Skip structured data (e.g., tables, spreadsheets)
    if should_skip_raptor(file_type, parser_id, task_parser_config, raptor_config):
        progress_callback(prog=1.0, msg=f"Raptor skipped: {get_skip_reason(...)}")
        return

    # Bind chat model and run RAPTOR
    chat_model = LLMBundle(task_tenant_id, chat_model_config, lang=task_language)
    async with kg_limiter:
        chunks, token_count = await run_raptor_for_kb(
            row=task,
            kb_parser_config=kb_parser_config,
            chat_mdl=chat_model,
            embd_mdl=embedding_model,
            vector_size=vector_size,
            callback=progress_callback,
            doc_ids=task.get("doc_ids", []),
        )
```

RAPTOR recursively clusters document chunks by semantic similarity, then generates abstractive summaries for each cluster using an LLM. These summaries are themselves embedded and stored as new chunks, forming a tree structure that enables multi-granularity retrieval.

### GraphRAG Execution

```python
elif task_type == "graphrag":
    ok, kb = KnowledgebaseService.get_by_id(task_dataset_id)

    kb_parser_config = kb.parser_config
    # Apply default GraphRAG config if not already set
    if not kb_parser_config.get("graphrag", {}).get("use_graphrag", False):
        kb_parser_config.update({
            "graphrag": {
                "use_graphrag": True,
                "entity_types": ["organization", "person", "geo", "event", "category"],
                "method": "light",
            }
        })
        KnowledgebaseService.update_by_id(kb.id, {"parser_config": kb_parser_config})

    graphrag_conf = kb_parser_config.get("graphrag", {})
    chat_model = LLMBundle(task_tenant_id, chat_model_config, lang=task_language)
    with_resolution = graphrag_conf.get("resolution", False)
    with_community = graphrag_conf.get("community", False)

    async with kg_limiter:
        result = await run_graphrag_for_kb(
            row=task,
            doc_ids=task.get("doc_ids", []),
            language=task_language,
            kb_parser_config=kb_parser_config,
            chat_model=chat_model,
            embedding_model=embedding_model,
            callback=progress_callback,
            with_resolution=with_resolution,
            with_community=with_community,
        )

    progress_callback(prog=1.0, msg="Knowledge Graph done ({:.2f}s)".format(timer() - start_ts))
```

GraphRAG extracts entities and relationships from document chunks using an LLM, builds a knowledge graph, and optionally resolves entity coreferences (`with_resolution`) and detects communities (`with_community`). This enables graph-traversal-based retrieval that captures multi-hop relationships across documents.

---

## Concurrency Control

Both GraphRAG and RAPTOR use an async semaphore `kg_limiter` to limit concurrent knowledge graph / RAPTOR tasks. This prevents the system from being overwhelmed when multiple knowledge bases trigger these tasks simultaneously.

```python
async with kg_limiter:
    result = await run_graphrag_for_kb(...)
```

---

## Web UI Integration

The frontend exposes these features directly in the dataset settings page (`web/src/pages/dataset/dataset-setting/index.tsx`):

```tsx
<GraphRagItems
  className="border-none p-0"
  data={graphRagGenerateData as IGenerateLogButtonProps}
  onDelete={() => handleDeletePipelineTask(GenerateType.KnowledgeGraph)}
/>

<RaptorFormFields
  data={raptorGenerateData as IGenerateLogButtonProps}
  onDelete={() => handleDeletePipelineTask(GenerateType.Raptor)}
/>
```

Each component shows a generation button, live progress, and a delete/cancel option — all backed by polling the task status from the API.

---

## End-to-End Flow Summary

```
User clicks "Generate Knowledge Graph" in UI
        ↓
POST /api/v1/datasets/{kb_id}/run_graphrag
        ↓
queue_raptor_o_graphrag_tasks(ty="graphrag")
  - Inserts Task record into DB (task_type = "graphrag")
  - Pushes task message onto Redis queue
        ↓
Task executor (do_handle_task) picks up from Redis
  - Detects task_type == "graphrag"
  - Validates KB and sets default config if needed
  - Calls run_graphrag_for_kb() with all doc_ids
        ↓
Entities and relationships extracted → Knowledge Graph stored
```

A standard document parse task uses a completely different path — `queue_tasks()` creates per-chunk parsing tasks, while GraphRAG and RAPTOR tasks operate at the KB level on already-parsed chunks.

---

## Key Takeaways

- GraphRAG and RAPTOR are **post-parsing** enhancements — they operate on chunks that already exist in the vector store.
- Both use the **`task_type`** field to differentiate from normal parsing tasks in the same task queue.
- The **fake document ID** pattern elegantly bypasses the document-level task model, allowing KB-wide tasks to fit into the existing task infrastructure.
- Default configs are **lazily applied** — if the KB has no GraphRAG or RAPTOR config, the executor writes sensible defaults before proceeding.
- **Concurrency is rate-limited** via an async semaphore to protect LLM and compute resources.

Understanding this task dispatch architecture is essential if you want to extend RAGFlow with custom post-processing pipelines or build tooling around its knowledge graph capabilities.

## References

- <https://deepwiki.com/search/does-web-alway-use-tasktype-da_6848caed-a23c-412c-839f-8867d4b844b1?mode=fast>