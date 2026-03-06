# How RAGFlow Creates a Dataset: A Code Walkthrough

RAGFlow exposes datasets (also called knowledge bases) as a first-class resource. Understanding how one gets created end-to-end — from an HTTP request all the way to a database row — is a good entry point into the codebase. This post traces that path through three layers: the API handler, the service layer, and the persistence call.

## The API Layer: Accepting the Request

The entry point for dataset creation is the `create` function in `api/apps/sdk/dataset.py` [dataset.py:55-L127](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/sdk/dataset.py#L55-L127). It is registered to handle `POST /api/v1/datasets` and is responsible for:

- Parsing and validating the JSON request body
- Resolving the authenticated tenant from the bearer token
- Forwarding the sanitised parameters to the service layer

Key fields accepted at this layer include `name`, `embedding_model`, `chunk_method`, `parser_config`, and `permission`. Any unsupported key is rejected before the call goes deeper.

## The Service Layer: Core Creation Logic

The actual work happens in `create_with_name` inside `api/db/services/knowledgebase_service.py` [knowledgebase_service.py:374-L430](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/services/knowledgebase_service.py#L374-L430). This method:

1. **Validates the name** — empty or whitespace-only names are rejected outright.
2. **Handles name collisions** — if a dataset with the same name already exists under the tenant, a numeric suffix is appended automatically (e.g. `my-docs (2)`).
3. **Builds the payload** — default values are merged with the caller-supplied fields, producing a complete record ready for persistence.
4. **Constructs `parser_config`** — chunk method-specific defaults are applied here; for instance, `naive` and `book` carry different splitting parameters.

## Saving to the Database

Once the payload is assembled, `KnowledgebaseService.save()` commits it. A unique ID is generated at this point, and the record becomes queryable immediately. The dataset is scoped to the tenant, so the name uniqueness constraint is per-tenant, not global.

## Configuration Reference

The following fields can be set when creating a dataset [dataset.py:55-L127](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/sdk/dataset.py#L55-L127):

| Field | Description |
|---|---|
| `name` | Human-readable label for the dataset (required, unique per tenant) |
| `embedding_model` | Model used to embed chunks for retrieval |
| `chunk_method` | Splitting strategy: `naive`, `book`, `qa`, and others |
| `parser_config` | Method-specific parser options |
| `permission` | Access control: `me` or `team` |

## Quick Start: Python SDK

The Python SDK wraps the HTTP call with a thin client. A minimal example from the repo [dataset_example.py:27-L32](https://github.com/infiniflow/ragflow/blob/ce71d878/example/sdk/dataset_example.py#L27-L32):

```python
client.create_dataset(
    name="my-knowledge-base",
    embedding_model="text-embedding-ada-002",
    chunk_method="naive",
)
```

Or directly via HTTP [http_api_reference.md:440-L450](https://github.com/infiniflow/ragflow/blob/ce71d878/docs/references/http_api_reference.md#L440-L450):

```bash
curl -X POST http://localhost/api/v1/datasets \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-knowledge-base", "chunk_method": "naive"}'
```

## Key Takeaways

- **Dataset = KnowledgeBase** internally; the two terms are used interchangeably throughout the codebase.
- **IDs are auto-generated** — callers never supply a dataset ID on creation.
- **Name uniqueness is tenant-scoped** — the service layer handles collisions automatically with a suffix strategy rather than returning an error.
- **`parser_config` is derived** — you rarely need to set it manually; the service builds sensible defaults from `chunk_method`.

## References

- <https://deepwiki.com/search/datasets_0b931efd-c718-4e6e-b839-e37354b4abec?mode=fast>