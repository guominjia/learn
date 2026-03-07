# How RAGFlow Resolves `index_name` and `knowledgebase_id` During Chat

When users select one or more datasets in RAGFlow chat, retrieval only works if the system can map each dataset to the correct vector storage layout. In practice, that means answering two key questions:

1. Which index namespace should be used?
2. Which vector dimension should be used for query and document matching?

This article walks through the internal flow and explains why `index_name`, `knowledgebase_id`, and embedding model consistency are tightly coupled.

## The Problem Behind the Scenes

In a multi-tenant system, each tenant must be isolated at storage level, and each knowledge base may carry its own metadata. At retrieval time, RAGFlow must guarantee:

- tenant-level isolation,
- schema compatibility for vector fields,
- and consistent embedding behavior across selected datasets.

If any of these assumptions break, retrieval can fail or produce invalid similarity scores.

## Step 1: Build `index_name` from `tenant_id`

RAGFlow derives the index namespace from `tenant_id` via `search.index_name(tenant_id)`, then initializes the storage index with the current `kb_id` and vector size.

```python
def init_kb(row, vector_size: int):
    idxnm = search.index_name(row["tenant_id"])
    parser_id = row.get("parser_id", None)
    return settings.docStoreConn.create_idx(idxnm, row.get("kb_id", ""), vector_size, parser_id)
```

Why this matters:

- `tenant_id` controls namespace boundaries.
- `kb_id` identifies the target knowledge base inside that namespace.
- The resulting storage object is prepared with the expected vector schema.

## Step 2: Infer Vector Dimension from Embeddings

Vector dimension is not hard-coded. It is inferred from embedding output and therefore depends on the selected embedding model (`embd_id`) configured on the knowledge base.

```python
vector_size = 0
for i, d in enumerate(docs):
    v = vects[i].tolist()
    vector_size = len(v)
    d["q_%d_vec" % len(v)] = v
```

This has an important consequence: schema and query vectors are model-dependent. If datasets use different embedding models, their vector spaces are not directly compatible.

## Step 3: Retrieval Uses the Same Embedding Context

During chat retrieval, RAGFlow passes the embedding model context, tenant scope, and selected knowledge base IDs into the retriever.

```python
kbinfos = await retriever.retrieval(
    " ".join(questions),
    embd_mdl,
    tenant_ids,
    dialog.kb_ids,
    1,
    dialog.top_n,
    dialog.similarity_threshold,
    dialog.vector_similarity_weight,
    doc_ids=attachments,
    top=dialog.top_k,
    aggs=True,
    rerank_mdl=rerank_mdl,
    rank_feature=label_question(" ".join(questions), kbs),
)
```

At this point, the system expects vectors in the target stores to match the embedding model used for the query.

## How Table Names Are Derived

The physical table naming rule is:

`{index_name}_{knowledgebase_id}`

Example:

- `tenant_id = tenant123` → `index_name = ragflow_tenant123`
- `kb_id = kb456` → table name `ragflow_tenant123_kb456`

This naming strategy combines tenant isolation with knowledge-base-level granularity.

## Why Mixed Embedding Models Are Rejected

RAGFlow validates selected datasets before chat. If the datasets do not share the same embedding model family, the request is rejected.

That safeguard prevents:

- cross-space similarity comparisons,
- vector dimension conflicts,
- and retrieval quality degradation caused by incompatible embedding spaces.

## Key Takeaways

- `index_name` is derived from `tenant_id` to enforce tenant isolation.
- Vector dimension comes from the embedding model output, not manual configuration.
- Storage tables follow `{index_name}_{knowledgebase_id}`.
- Chat retrieval is valid only when selected datasets use compatible embedding models.

## References

- [`rag/svr/task_executor.py:564-567`](https://github.com/infiniflow/ragflow/blob/ce71d878/rag/svr/task_executor.py#L564-L567) (`init_kb`, vector size inference)
- [`api/db/db_models.py:833`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L833-L833) (`embd_id` in knowledge base model)
- [`api/apps/chunk_app.py:408`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/chunk_app.py#L408-L408) (embedding model bundle creation)
- [`api/db/services/dialog_service.py:560-574`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/services/dialog_service.py#L560-L574) (chat retrieval call)
- [`rag/utils/infinity_conn.py:318-319`](https://github.com/infiniflow/ragflow/blob/ce71d878/rag/utils/infinity_conn.py#L318-L319) (table naming)
- [`api/apps/dialog_app.py:103-106`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/dialog_app.py#L103-L106) (embedding consistency validation)
- <https://deepwiki.com/search/datasetragflowknowledgebase-kn_06736f77-7afb-4b2c-8afa-7885dffcf5f3>