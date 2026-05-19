# How RAGFlow Uses Elasticsearch to Filter by Dataset

In RAGFlow, when a user performs a retrieval query, the system doesn't search the entire document store — it narrows results to only the specified datasets (knowledge bases). Elasticsearch achieves this by injecting a `kb_id` field filter into every query. This post walks through the architecture, the code path from API to storage, and how other backends handle the same problem differently.

## The Core Mechanism: Single Index + Field Filtering

Elasticsearch in RAGFlow follows a **single-index, field-filtered** design:

1. **One index per tenant** — All document chunks belonging to a tenant live in the same Elasticsearch index.
2. **Field-level isolation** — Every chunk carries a `kb_id` field that identifies which dataset (knowledge base) it belongs to.
3. **Query-time filtering** — At search time, a `terms` filter on `kb_id` restricts results to the requested datasets.

This is a classic multi-tenant pattern in Elasticsearch: rather than creating separate indices per dataset (which would increase cluster overhead), RAGFlow keeps a flat structure and relies on efficient filter caching.

## Code Walkthrough

### Step 1 — The Search Interface Receives Dataset IDs

The entry point is `ESConnection.search()` in `rag/utils/es_conn.py`. The method accepts a `knowledgebase_ids` parameter and immediately injects it into the condition dictionary:

```python
def search(
    self, select_fields: list[str],
    highlight_fields: list[str],
    condition: dict,
    match_expressions: list[MatchExpr],
    order_by: OrderByExpr,
    offset: int,
    limit: int,
    index_names: str | list[str],
    knowledgebase_ids: list[str],   # list of dataset IDs to search
    ...
):
    ...
    condition["kb_id"] = knowledgebase_ids  # inject into filter conditions
```

### Step 2 — Building the Elasticsearch Bool Query

The condition dictionary is then iterated to construct an Elasticsearch [bool query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html). For list-type values (like `kb_id`), the code appends a `terms` filter:

```python
bool_query = Q("bool", must=[])
condition["kb_id"] = knowledgebase_ids

for k, v in condition.items():
    ...
    if isinstance(v, list):
        bool_query.filter.append(Q("terms", **{k: v}))  # terms filter for kb_id
    elif isinstance(v, str) or isinstance(v, int):
        bool_query.filter.append(Q("term", **{k: v}))
```

The resulting Elasticsearch query looks conceptually like this:

```json
{
  "query": {
    "bool": {
      "filter": [
        { "terms": { "kb_id": ["dataset_id_1", "dataset_id_2"] } }
      ],
      "must": [ ... ]
    }
  }
}
```

Because `terms` queries are placed in the `filter` context, Elasticsearch can cache them and skip scoring — making dataset filtering essentially free in terms of relevance computation.

### Step 3 — The Full Call Chain (API → Service → Storage)

The dataset IDs flow through several layers before reaching Elasticsearch:

| Layer | Entry Point | Role |
|-------|------------|------|
| **HTTP API** | `POST /api/v1/retrieval` | Accepts `dataset_ids` in the request body |
| **Python SDK** | `ragflow_sdk.RAGFlow.retrieve()` | Wraps the HTTP call with typed parameters |
| **Service Layer** | `settings.retriever.retrieval()` | Orchestrates retrieval logic |
| **Storage Layer** | `ESConnection.search()` | Constructs and executes the Elasticsearch query |

## How Other Backends Handle Dataset Isolation

RAGFlow's document store abstraction supports multiple backends, each with its own isolation strategy:

| Backend | Isolation Strategy | Implementation |
|---------|--------------------|----------------|
| **Elasticsearch** | Single index + `kb_id` field filter | `rag/utils/es_conn.py` |
| **Infinity** | Separate table per dataset (`{index_name}_{kb_id}`) | `rag/utils/infinity_conn.py` |
| **OceanBase** | Single table + `kb_id` field filter | `rag/utils/ob_conn.py` |

The Infinity backend takes a fundamentally different approach: instead of filtering on a field, it creates a **dedicated table for each knowledge base**. The table name follows the pattern `{index_name}_{kb_id}`, so the query is physically scoped to the correct data. The trade-off is more tables to manage but zero filtering overhead.

OceanBase mirrors the Elasticsearch pattern — a single table with a `kb_id` column and query-time filtering.

## Usage Example

To search specific datasets via the RAGFlow API:

```bash
curl --request POST \
     --url http://{address}/api/v1/retrieval \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer <YOUR_API_KEY>' \
     --data '{
          "question": "What is advantage of ragflow?",
          "dataset_ids": ["b2a62730759d11ef987d0242ac120004"]
     }'
```

You can pass multiple dataset IDs to search across several knowledge bases simultaneously. If you omit `dataset_ids`, you must provide `document_ids` instead — at least one scoping parameter is required.

## Key Takeaways

- **Efficient multi-dataset search**: The `terms` filter on `kb_id` supports querying across multiple datasets in a single request with minimal overhead.
- **Filter caching**: Since `terms` queries in the `filter` context are cacheable, repeated searches against the same datasets benefit from Elasticsearch's filter cache.
- **Backend-agnostic design**: RAGFlow's storage abstraction means you can swap Elasticsearch for Infinity or OceanBase without changing the retrieval API — only the isolation mechanism differs under the hood.
- **Single-index simplicity**: By avoiding per-dataset indices, RAGFlow keeps cluster management simple while still achieving strong data isolation at query time.

## References

- <https://deepwiki.com/search/elasticsearchdataset_cef52542-7a64-428d-9013-d46102a852b8?mode=fast>