# How RAGFlow Searches Across Multiple Datasets

When a Chat or Search request in RAGFlow involves multiple datasets, the system retrieves results from all of them **in parallel (batch lookup)** rather than querying each dataset sequentially. This design keeps latency low and ensures that the final answer is drawn from the best chunks across every dataset in a single pass.

---

## Retrieval Architecture

### 1. Unified Search Entry Point

Whether triggered by a Chat dialog or the Search API, every query ultimately lands in **`Dealer.search`**. This method accepts a list of dataset IDs (`kb_ids`) and fans the query out to all of them at once:

```python
# rag/nlp/search.py L132-137
async def search(self, req, idx_names: str | list[str],
           kb_ids: list[str],
           emb_mdl=None,
           highlight: bool | list | None = None,
           rank_feature: dict | None = None):
```

On the API side, the request model explicitly supports a list of datasets:

```python
# api/utils/validation_utils.py L913-918
class SearchDatasetsReq(BaseModel):
    """Request model for searching multiple datasets."""
    dataset_ids: Annotated[list[str], Field(..., min_length=1)]
```

### 2. Batch Query at the Storage Layer

The full `kb_ids` list is forwarded directly to the underlying document store (Elasticsearch, Infinity, or OceanBase):

```python
# rag/nlp/search.py L164
res = self.dataStore.search(
    src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids
)
```

How each storage engine handles the list differs in implementation, but the intent is the same — **cover all target datasets in a single round-trip**.

| Engine | Strategy |
|---|---|
| **Infinity** | Iterates over `index_names × memory_ids`, opens a table per combination (`{index}_{dataset_id}`), and aggregates results. |
| **OceanBase** | Builds a SQL `IN` clause from `kb_ids` so one query scans every relevant partition. |
| **Elasticsearch** | Uses the native multi-index search capability with `kb_ids` as a filter. |

**Infinity example:**

```python
# memory/utils/infinity_conn.py L226-233
for indexName in index_names:
    for memory_id in memory_ids:
        table_name = f"{indexName}_{memory_id}"
        try:
            table_instance = db_instance.get_table(table_name)
        except Exception:
            continue
        table_list.append(table_name)
```

### 3. Merging and Reranking

After retrieval, candidate chunks from **all** datasets are pooled together for a single, unified post-processing pipeline:

1. **Hybrid scoring** — A weighted sum fuses vector similarity and keyword (BM25) similarity into one score. The default weight split is 5 % keyword / 95 % vector:

    ```python
    # rag/nlp/search.py L186-187
    fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
    matchExprs = [matchText, matchDense, fusionExpr]
    ```

2. **Cross-dataset reranking** — If a rerank model is configured, all candidate chunks (regardless of source dataset) are passed through `rerank_by_model` together. This ensures globally optimal ordering.

3. **Threshold filtering** — Chunks whose final score falls below `similarity_threshold` are discarded. The threshold, vector weight, and top-N are configured at the dialog level:

    ```python
    # api/db/services/dialog_service.py L178-180
    cls.model.similarity_threshold,
    cls.model.vector_similarity_weight,
    cls.model.top_n,
    ```

---

## Key Constraint: Embedding-Model Consistency

All datasets queried together **must use the same embedding model**. Without this rule, vectors live in different spaces and cosine-similarity comparisons become meaningless:

```python
# api/apps/services/dataset_api_service.py L1315-1318
embd_nms = list(set([
    TenantLLMService.split_model_name_and_factory(kb.embd_id)[0]
    for kb in kbs
]))
if len(embd_nms) != 1:
    return False, "Datasets use different embedding models."
```

> *"Select one or multiple datasets, but ensure that they use the same embedding model, otherwise an error would occur."*
> — RAGFlow documentation

---

## Influencing Result Priority with Page Rank

RAGFlow provides a **Page Rank** knob per dataset that lets you bias the unified scoring stage toward specific datasets. It works by adding a bonus to every chunk that originates from a higher-ranked dataset.

**Example:** Suppose you have two datasets — *Dataset A (2024 news, page rank = 1)* and *Dataset B (2023 news, page rank = 0)*. A chunk from Dataset A with an initial score of 50 receives a boost of $1 \times 100 = 100$ points, giving it a final score of $50 + 100 = 150$. Chunks from Dataset B receive no boost, so 2024 news is effectively always surfaced first.

```python
# rag/nlp/search.py L335-340
if not query_rfea:
    return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
```

---

## Summary

| Aspect | Detail |
|---|---|
| **Query dispatch** | All datasets are searched in a single batch call via `Dealer.search`. |
| **Storage layer** | `kb_ids` are passed as-is; each engine parallels or batches internally. |
| **Post-processing** | Hybrid scoring → reranking → threshold filtering, all cross-dataset. |
| **Constraint** | Every dataset in the query must share the same embedding model. |
| **Priority tuning** | Use the per-dataset Page Rank to boost or suppress specific sources. |

## References

- <https://deepwiki.com/search/chatsearchdataset_4931b523-15e1-42b9-be9f-39f72d53bc62?mode=fast>