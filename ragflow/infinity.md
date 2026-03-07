# Why Infinity Is a Strong Document Engine for RAGFlow

If you are building a Retrieval-Augmented Generation (RAG) system, database choice directly impacts recall quality, latency, and operational complexity. In the RAGFlow ecosystem, **Infinity** is designed as an AI-native document engine that blends vector retrieval and structured querying in one system.

This article explains what Infinity is, how it is integrated into RAGFlow, and why it differs from traditional systems such as MySQL.

## What Is Infinity?

Infinity is an AI-native vector database optimized for RAG workloads. It is not a classic transactional relational database, and it is not only a keyword search engine either. Instead, it focuses on the retrieval patterns that modern LLM applications need:

- semantic retrieval over embedding vectors
- keyword and metadata filtering
- scalable indexing for large chunk collections

In short, Infinity is built for retrieval quality and speed in AI applications.

## Core Capabilities

### 1) Native Vector Storage

Infinity supports vector columns natively, for example `vector,1024,float`. In RAGFlow, vector fields are created dynamically based on embedding dimensions.

### 2) Efficient Vector Indexing

Infinity uses HNSW indexing for approximate nearest-neighbor search. Typical index parameters include values such as:

- `M: 16`
- `ef_construction: 50`
- `metric: cosine`
- `encode: lvq`

This gives a good balance between retrieval performance and accuracy in practical RAG scenarios.

### 3) Table-Based Organization

RAGFlow organizes knowledge data in table form when using Infinity. A common naming pattern is:

`{index_name}_{knowledgebase_id}`

That model keeps data isolated per knowledge base and simplifies multi-tenant operations.

### 4) Multi-Protocol Access

Infinity can be accessed through multiple protocols, including:

- Thrift RPC on port `23817`
- HTTP API on port `23820`
- PostgreSQL-compatible protocol on port `5432`

This flexibility helps integration with both application services and operations tooling.

## Infinity vs. MySQL for RAG

| Dimension | Infinity | MySQL |
|---|---|---|
| Primary goal | Vector + hybrid retrieval | Transactional relational data |
| Vector support | Native with ANN index support | Usually requires extensions/plugins |
| Search model | Semantic + keyword hybrid | SQL-centric filtering and joins |
| Optimization target | AI and RAG pipelines | General-purpose business workloads |
| Typical document workload fit | High | Medium to low without add-ons |

MySQL is excellent for transactional systems, but Infinity is usually a better fit for high-quality document retrieval in LLM applications.

## Infinity Inside RAGFlow

In RAGFlow, Infinity is one of the pluggable document engines (alongside options like Elasticsearch/OpenSearch). Its primary responsibilities include:

1. **Chunk storage**: save content chunks, vectors, and metadata.
2. **Hybrid retrieval**: combine semantic similarity with textual constraints.
3. **Tenant isolation**: maintain separate logical tables per knowledge base.

This architecture lets teams scale ingestion and retrieval while keeping strong boundaries between datasets.

## Deployment Snapshot

A typical Docker service configuration for Infinity exposes three ports:

```yaml
infinity:
  image: infiniflow/infinity:v0.7.0-dev2
  ports:
    - ${INFINITY_THRIFT_PORT}:23817
    - ${INFINITY_HTTP_PORT}:23820
    - ${INFINITY_PSQL_PORT}:5432
```

In practice, RAGFlow also configures runtime parameters such as the Infinity URI, database name, health checks, and resource limits.

## Practical Takeaways

- Infinity is purpose-built for AI retrieval, not as a universal OLTP replacement.
- It combines vector and structured retrieval in a way that maps naturally to RAG pipelines.
- In RAGFlow, it acts as a drop-in document engine with strong support for multi-knowledge-base isolation.
- SQL-style access exists, but the system is tuned primarily for semantic search and hybrid recall.

If your priority is robust retrieval for LLM applications, Infinity is a practical and focused choice.

## References

- <https://deepwiki.com/search/datasetragflowknowledgebase-kn_06736f77-7afb-4b2c-8afa-7885dffcf5f3>