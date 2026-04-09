# Deep Dive into RAGFlow's Search Architecture: Document Storage, Hybrid Retrieval, and Similarity Computation

RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine that provides a sophisticated search pipeline. This post explores three critical aspects of its search architecture: the pluggable document store design, hybrid retrieval mechanism, and dynamic vector field naming — along with a comparison to ChromaDB's approach.

---

## 1. Pluggable Document Store Architecture

RAGFlow uses a **plugin-based architecture** for document storage. At any given time, only one document store engine is active. The system does **not** write to multiple backends simultaneously.

### How the Engine Is Selected

During startup, `common/settings.py` reads the `DOC_ENGINE` environment variable and instantiates exactly one connection:

```python
DOC_ENGINE = os.environ.get("DOC_ENGINE", "elasticsearch").strip()
lower_case_doc_engine = DOC_ENGINE.lower()

if lower_case_doc_engine == "elasticsearch":
    docStoreConn = rag.utils.es_conn.ESConnection()
elif lower_case_doc_engine == "infinity":
    docStoreConn = rag.utils.infinity_conn.InfinityConnection()
elif lower_case_doc_engine == "opensearch":
    docStoreConn = rag.utils.opensearch_conn.OSConnection()
elif lower_case_doc_engine == "oceanbase":
    docStoreConn = rag.utils.ob_conn.OBConnection()
else:
    raise Exception(f"Not supported doc engine: {DOC_ENGINE}")
```

Every engine implements the same `DocStoreConnection` interface, so the rest of the codebase is storage-agnostic. In `task_executor.py`, the insert call is simply:

```python
doc_store_result = await thread_pool_exec(
    settings.docStoreConn.insert,
    chunks[b:b + settings.DOC_BULK_SIZE],
    search.index_name(task_tenant_id),
    task_dataset_id,
)
```

### Switching Engines

To switch from Elasticsearch to Infinity (or another engine):

1. Stop all containers: `docker compose -f docker/docker-compose.yml down -v`
2. Set `DOC_ENGINE=infinity` in `docker/.env`
3. Restart: `docker compose -f docker-compose.yml up -d`

> **Warning:** The `-v` flag deletes container volumes. Existing data will be cleared, and you may need to re-ingest your documents.

### Engine-Specific Insert Behavior

Each backend handles insertion differently:

- **Elasticsearch** uses the [Bulk API](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html), batching index operations with retry logic and connection-timeout handling.
- **Infinity** maps RAGFlow's internal field names (e.g., `content_with_weight` → `content`, `docnm_kwd` → `docnm`) before calling the native `table.insert()`. It also manages a connection pool and handles table creation as a fallback.

---

## 2. Hybrid Retrieval: Combining Keywords and Vectors

The core value proposition of RAGFlow's search is **hybrid retrieval** — blending keyword (BM25-style) similarity with vector (embedding) similarity so that results capture both exact matches and semantic relevance.

### The Retrieval Pipeline

The `retrieval` method in `rag/nlp/search.py` orchestrates the flow:

```python
async def retrieval(
    self, question, embd_mdl, tenant_ids, kb_ids,
    page, page_size,
    similarity_threshold=0.2,
    vector_similarity_weight=0.3,
    top=1024,
    rerank_mdl=None,
    rank_feature: dict | None = {PAGERANK_FLD: 10},
    ...
):
```

Key steps:
1. Build a search request with both text and vector components.
2. Dispatch the request to the configured document store.
3. Fuse and re-rank results.

### Fusion Strategies per Engine

Each storage backend implements fusion differently, but they all honor the same `vector_similarity_weight` parameter.

**Elasticsearch** — combines a `bool` query (text) with a `knn` query (vector), applying boost weights:

```go
// Text weight
textWeight := 1.0 - req.VectorSimilarityWeight
boolQuery := buildESKeywordQuery(matchText, filterClauses, 1.0)
boolMap["boost"] = textWeight

// Vector kNN query
knnQuery := map[string]interface{}{
    "field":          vectorFieldName,
    "query_vector":   req.Vector,
    "k":              k,
    "num_candidates": k * 2,
    "similarity":     req.SimilarityThreshold,
}
```

**Infinity** — uses a built-in `weighted_sum` fusion method:

```go
searchReq.Fusion = &FusionExpr{
    Method: "weighted_sum",
    TopN:   topK,
    Weights: []float64{
        1.0 - vectorSimilarityWeight, // text weight
        vectorSimilarityWeight,       // vector weight
    },
}
```

**OceanBase** — performs fusion in SQL via a `FULL OUTER JOIN` between full-text and vector result sets:

```sql
SELECT COALESCE(f.id, v.id) AS id,
       (f.relevance * (1 - weight) + v.similarity * weight + pagerank) AS score
FROM fulltext_results f
FULL OUTER JOIN vector_results v ON f.id = v.id
ORDER BY score DESC
```

### Scoring Formula

After retrieving candidates, RAGFlow computes the final similarity score:

```python
sim = tkweight * np.array(tksim) + vtweight * vtsim + rank_fea
```

| Component    | Default Weight | Description                          |
|-------------|---------------|--------------------------------------|
| `tkweight`  | 0.7           | Keyword (term) similarity weight     |
| `vtweight`  | 0.3           | Vector cosine similarity weight      |
| `rank_fea`  | varies        | PageRank or other feature scores     |

### Re-ranking

When a re-rank model is configured, RAGFlow replaces the simple weighted combination with model-based scoring. For the Infinity engine specifically, re-ranking is skipped because Infinity normalizes each component score before fusion internally:

```python
if settings.DOC_ENGINE_INFINITY:
    # Infinity normalizes each way score before fusion — no rerank needed
    sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
else:
    sim, tsim, vsim = self.rerank(sres, question, ...)
```

### Configurable Parameters

| Parameter                   | Default | Description                                    |
|----------------------------|---------|------------------------------------------------|
| `vector_similarity_weight` | 0.3     | Weight for vector cosine similarity             |
| `similarity_threshold`     | 0.2     | Minimum similarity to include a result          |
| `top_k`                    | 1024    | Number of candidate chunks to retrieve          |

---

## 3. Dynamic Vector Field Naming

RAGFlow uses a dynamic naming convention for vector fields: `q_{dimension}_vec`. This allows a single index to store embeddings of different dimensions from different models.

### How It Works

**At index time**, the vector field name is derived from the embedding dimension:

```python
# In infinity_conn_base.py
vector_name = f"q_{vector_size}_vec"
schema[vector_name] = {"type": f"vector,{vector_size},float"}
```

**At query time**, the same naming pattern is reconstructed from the query embedding:

```python
# In rag/nlp/search.py
embedding_data = [get_float(v) for v in qv]
vector_column_name = f"q_{len(embedding_data)}_vec"
return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, ...)
```

The Go-based engines follow the same convention:

```go
// Both Elasticsearch and Infinity engines
fieldBuilder.WriteString("q_")
fieldBuilder.WriteString(strconv.Itoa(dimension))
fieldBuilder.WriteString("_vec")
```

### Index Mapping Support

Elasticsearch and OpenSearch use dynamic templates to handle multiple vector dimensions:

```json
// Elasticsearch (conf/mapping.json)
{ "match": "*_768_vec",  "mapping": { "type": "dense_vector", "dims": 768,  "similarity": "cosine" } },
{ "match": "*_1024_vec", "mapping": { "type": "dense_vector", "dims": 1024, "similarity": "cosine" } },
{ "match": "*_1536_vec", "mapping": { "type": "dense_vector", "dims": 1536, "similarity": "cosine" } }
```

```json
// OpenSearch (conf/os_mapping.json)
{ "match": "*_768_vec",  "mapping": { "type": "knn_vector", "dimension": 768,  "space_type": "cosinesimil" } },
{ "match": "*_1024_vec", "mapping": { "type": "knn_vector", "dimension": 1024, "space_type": "cosinesimil" } }
```

This design means:
- The same index can hold embeddings from models of different output dimensions.
- The query embedding's length automatically determines which field to search.
- No manual configuration of vector field names is needed.

---

## 4. RAGFlow vs. ChromaDB: Similarity Computation

It's worth contrasting RAGFlow's approach with ChromaDB, which relies on HNSW (Hierarchical Navigable Small World) graphs for approximate nearest neighbor search.

### Algorithm Comparison

| Aspect               | RAGFlow                              | ChromaDB (HNSW)                  |
|----------------------|--------------------------------------|----------------------------------|
| **Algorithm**        | Exact cosine similarity (+ kNN)      | HNSW approximate nearest neighbor|
| **Recall**           | 100% (exact search)                  | < 100% (approximate)             |
| **Query Complexity** | O(n) for exact; engine-optimized     | O(log n) amortized               |
| **Hybrid Search**    | Native (text + vector + PageRank)    | Vector-only                      |
| **Storage Backends** | ES, Infinity, OpenSearch, OceanBase  | Built-in (single backend)        |

### RAGFlow's Cosine Similarity

In the Go reranker, RAGFlow computes exact cosine similarity:

```go
vsim = make([]float64, len(bvecs))
for i, bvec := range bvecs {
    vsim[i] = cosineSimilarity(avec, bvec)
}
```

This guarantees perfect recall but can be slower on very large datasets.

### When to Choose Which

**Choose RAGFlow when:**
- You need hybrid retrieval (keyword + semantic + PageRank).
- High recall is critical and cannot tolerate missed results.
- Your data volume is small to medium (tens of millions of chunks).
- You want flexibility to swap storage backends.

**Choose ChromaDB when:**
- You need the fastest possible vector-only queries.
- Approximate results are acceptable.
- Your application is primarily semantic search without keyword matching needs.
- You want a lightweight, embedded solution.

---

## Key Takeaways

1. **Pluggable storage**: RAGFlow's `DocStoreConnection` interface abstracts away four different backends behind a single API. Switching engines is a one-line config change.
2. **Hybrid retrieval**: By fusing keyword and vector scores with configurable weights, RAGFlow avoids the recall limitations of pure vector search.
3. **Dynamic vector fields**: The `q_{dim}_vec` naming convention elegantly supports multiple embedding models within a single index.
4. **Precision vs. speed**: RAGFlow trades some query speed for exact results and hybrid search capability — a worthwhile tradeoff for RAG applications where answer quality is paramount.

## References

- <https://deepwiki.com/search/docstoreresult-await-threadpoo_50593a6e-9bc2-468b-ac53-4d1c1714438b?mode=fast>