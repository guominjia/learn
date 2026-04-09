# How RAGFlow Creates and Persists a Knowledge Base: A Service-Layer Walkthrough

When reading RAGFlow internals, one common question is: **where does a new knowledge base actually get stored?**

The short answer is simple: `KnowledgebaseService.save()` writes data into the `knowledgebase` table.

Reference: [db_models.py:859-860](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L859-L860)

## The Core Idea

RAGFlow uses a clean service-layer pattern:

- The `Knowledgebase` model declares the target table.
- `KnowledgebaseService` handles creation and persistence logic.
- The app endpoints and SDK trigger a high-level create flow.

This keeps route handlers thin and data logic centralized.

## 1) Table Mapping: `Knowledgebase` → `knowledgebase`

The ORM model `Knowledgebase` maps directly to the `knowledgebase` table.

Reference: [db_models.py:826-860](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/db_models.py#L826-L860)

From a data-path perspective, this is the final storage destination for knowledge base metadata.

## 2) Save Logic: Inherited from `CommonService`

`KnowledgebaseService` extends `CommonService` and uses the parent `save` implementation.

Reference: [common_service.py:141-154](https://github.com/infiniflow/ragflow/blob/ce71d878/api/db/services/common_service.py#L141-L154)

The save path uses `force_insert=True`, which enforces an insert operation for new records. In practice, that means a new knowledge base creation will generate a new row in `knowledgebase` rather than attempting an update-first behavior.

## 3) Creation Entry Point: `create_with_name`

In the codebase, the practical creation API is `create_with_name`.

It is invoked from:

- App layer: [kb_app.py:60-65](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/kb_app.py#L60-L65)
- SDK layer: [dataset.py:122-127](https://github.com/infiniflow/ragflow/blob/ce71d878/api/apps/sdk/dataset.py#L122-L127)

This method prepares the dataset/knowledge-base payload and then persists it through the service save pipeline.

## End-to-End Flow

1. Call `KnowledgebaseService.create_with_name()` to construct and validate initial knowledge base data.
2. Call `KnowledgebaseService.save()` (via service flow) to persist the record.
3. The ORM writes a new row into the `knowledgebase` table.

## Why This Design Works

- **Consistency:** all persistence goes through a shared service abstraction.
- **Reusability:** both API and SDK can reuse the same creation behavior.
- **Maintainability:** business logic stays in services, not duplicated in route handlers.

## References

- <https://deepwiki.com/search/knowledgebasecreatewithsavekno_838cd1fa-72e3-47fc-8cfe-b5129ac7e3dc?mode=fast>

