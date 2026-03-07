# Where Does a Dataset Actually Live in RAGFlow?

When people first inspect RAGFlow, they often assume that a dataset is physically stored in the `knowledgebase` table. That assumption is understandable—but incomplete.

In practice, RAGFlow splits dataset state into two layers:

- **Metadata layer** in relational storage (`knowledgebase`)
- **Chunk/content layer** in a document/vector engine (for example Infinity/OpenSearch-compatible backends)

This separation is intentional and is one of the key reasons RAGFlow can support both management-friendly dataset metadata and retrieval-friendly chunk search at scale.

## The Mental Model: Dataset Metadata vs. Retrieval Payload

Think of a dataset as a control plane + data plane design:

- The **control plane** (`knowledgebase`) tracks who owns the dataset, which parser/embedding settings it uses, and aggregate stats.
- The **data plane** stores chunked text, tokenized variants, vector embeddings, and retrieval-specific features.

That means creating a dataset does **not** write all chunk text into `knowledgebase`; it seeds metadata and configuration that later guide indexing.

## What Is Stored in `knowledgebase`

`Knowledgebase` maps to the `knowledgebase` table and keeps dataset-level attributes such as:

- Identity and ownership (`id`, `tenant_id`, `created_by`, `name`)
- Processing defaults (`embd_id`, `parser_id`, `parser_config`)
- Aggregates (`doc_num`, `token_num`, `chunk_num`)
- Long-running task tracking (`graphrag_task_id`, `raptor_task_id`, etc.)

In short, this table answers **what the dataset is** and **how it should be processed**, not **where every chunk body is stored**.

## Where Chunks Are Stored

Chunks are written to the document/vector backend, not to `knowledgebase` rows.

RAGFlow builds chunk table names dynamically using:

`{index_name}_{kb_id}`

Typical behavior:

- `index_name` reflects the retrieval index namespace (often vector-field related, such as `q_1024_vec`)
- `kb_id` isolates data per dataset

This naming strategy gives tenant/dataset isolation while keeping lookup routes deterministic.

## What a Chunk Record Contains

Chunk documents are richer than plain text blobs. Depending on backend schema and parser mode, they may include:

- Core content fields (`content_with_weight`, tokenized forms)
- Metadata (`kb_id`, `doc_id`, document name/type)
- Embedding vectors (`q_{size}_vec`)
- Retrieval features (keywords, questions, tags, rank-related fields)
- Positional/page structure (`position_int`, `page_num_int`)
- Optional graph/table-specific extensions (`chunk_data`, entity/edge fields, etc.)

This is why retrieval remains fast and expressive: the online query path reads directly from retrieval-optimized chunk records.

## End-to-End Data Flow

1. **Dataset creation** writes a new `knowledgebase` row with validated defaults.
2. **Document ingestion/parsing** generates chunks.
3. **Chunk indexing** writes those chunks into backend tables like `{index_name}_{kb_id}`.
4. **Search/retrieval** reads chunk records from the backend table set, not from `knowledgebase`.

If you only inspect SQL tables and do not check the retrieval backend, it can look like “dataset content is missing”—while in reality content lives in the vector/document engine.

## Practical Debugging Checklist

When you need to verify where your dataset content went:

1. Confirm a row exists in `knowledgebase` for the expected `kb_id`.
2. Confirm ingestion ran successfully and produced chunk counts.
3. Check backend table naming: `index_name + '_' + kb_id`.
4. Verify your query path is searching the correct `knowledgebase_ids` set.
5. If a table is missing in one engine, check whether this dataset was created/indexed in another configured engine.

## Why This Design Works

- Keeps metadata operations simple and transactional.
- Keeps retrieval storage specialized for vector/full-text workloads.
- Supports extensible schemas for advanced parsers and GraphRAG-style workflows.
- Maintains clear dataset isolation through deterministic index/table naming.

## References

- [`api/db/db_models.py:826-860`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L826-L860) (`Knowledgebase` model and `knowledgebase` table mapping)
- [`api/db/services/knowledgebase_service.py:376-430`](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/services/knowledgebase_service.py#L376-L430) (`create_with_name` dataset creation flow)
- [`rag/utils/infinity_conn.py:233-234`](https://github.com/infiniflow/ragflow/blob/ce71d878/rag/utils/infinity_conn.py#L233-L234) (dynamic table naming, insert/get path)
- [`rag/utils/ob_conn.py:49-94`](https://github.com/infiniflow/ragflow/blob/ce71d878/rag/utils/ob_conn.py#L49-L94) (chunk schema definition)
- <https://deepwiki.com/search/datasetragflowknowledgebase-kn_06736f77-7afb-4b2c-8afa-7885dffcf5f3>