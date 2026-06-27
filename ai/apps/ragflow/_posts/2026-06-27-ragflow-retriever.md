---
title: Understanding the Retriever Flow in RAGFlow
categories: [ai, rag, ragflow]
tags: [ragflow, retriever, retrieval, elasticsearch, rag]
---

In every Retrieval-Augmented Generation (RAG) system, the retriever is one of the most important components.

The generator can only answer with the knowledge it receives. If the retriever misses the right chunks, returns noisy content, or ranks documents poorly, the final answer will also be weak. RAGFlow follows the same principle: the retriever is central to the quality of the whole RAG pipeline.

This post walks through the core retrieval path in RAGFlow and explains how the main objects are connected.

## High-Level Retrieval Flow

The core retrieval flow involves both the `common` directory and the `rag` directory.

At a high level, the chain looks like this:

```text
common/settings.py
	common.settings.retriever
		-> rag.nlp.search.Dealer()
			-> Dealer.retrieval()
				-> Dealer.search()
					-> self.dataStore.search()
						-> rag.utils.es_conn.ESConnection()
							-> common.doc_store.es_conn_base
```

The important idea is that `Dealer` owns the retrieval logic, but the actual search backend is delegated to `dataStore`.

## Retriever Initialization in `common.settings`

RAGFlow creates the global retriever instance in `common/settings.py`.

The `common.settings` module initializes `retriever` as:

```python
retriever = rag.nlp.search.Dealer()
```

So the central retriever object is an instance of `Dealer`, which is implemented in:

```text
rag/nlp/search.py
```

This means most high-level retrieval behavior should be understood by reading `rag/nlp/search.py` first.

## `Dealer`: The Main Retrieval Entry Point

The `Dealer` class provides two important functions:

- `retrieval()`
- `search()`

The `retrieval()` function is the higher-level API. It prepares retrieval parameters, calls `search()`, and then formats or post-processes the result.

The `search()` function performs the lower-level search operation. Inside `search()`, RAGFlow eventually calls:

```python
self.dataStore.search(...)
```

This is the key boundary between retrieval orchestration and the document store implementation.

## `dataStore`: The Search Backend

The `Dealer` object receives a `dataStore` parameter. In the `common.settings` instance, this backend is currently:

```python
rag.utils.es_conn.ESConnection()
```

The implementation is located in:

```text
rag/utils/es_conn.py
```

`ESConnection` is the Elasticsearch-based document store connection used by the retriever.

It also inherits from the base implementation in:

```text
common/doc_store/es_conn_base.py
```

So the actual search call travels from `Dealer.search()` into the Elasticsearch connection layer.

## Why This Design Matters

This design separates two responsibilities:

1. `Dealer` decides how retrieval should be performed.
2. `ESConnection` decides how documents should be searched from the backend store.

This separation is useful because the retrieval layer can focus on RAG-specific logic, while the storage layer can focus on Elasticsearch queries, filters, ranking, pagination, and aggregations.

In practice, if you want to understand or debug RAGFlow retrieval quality, you usually need to inspect both sides:

- `rag/nlp/search.py` for retrieval orchestration.
- `rag/utils/es_conn.py` and `common/doc_store/es_conn_base.py` for backend search behavior.

## Result Structure Returned by `retrieval()`

At the end of the retrieval process, `retrieval()` returns `ranks`.

The returned object contains three major fields:

```text
total
chunks
doc_aggs
```

These fields are related, but they represent different levels of information.

### `total`

`total` is the total number of documents after search and filtering.

It does not only mean the number of chunks returned on the current page. Instead, it represents the total matched result count after the search conditions and filters are applied.

This field is useful for pagination and for understanding the full size of the matched result set.

### `chunks`

`chunks` contains only the documents returned for the requested page.

For example, if the caller asks for a specific page with a limited page size, `chunks` only includes that slice of the result set.

This is the field most directly used by downstream RAG logic because these chunks are the actual retrieved context candidates.

### `doc_aggs`

`doc_aggs` contains document-level aggregation information.

This matters because multiple chunks can come from the same document. A RAG system often retrieves at the chunk level, but users and applications may still need document-level grouping or statistics.

For example, if five returned chunks come from the same PDF, `chunks` will show the individual chunk records, while `doc_aggs` can help identify their shared source document.

## Mental Model

A useful mental model is:

```text
retrieval() returns the final retrieval package
search() performs the search request
dataStore.search() talks to the actual backend
ESConnection implements the Elasticsearch behavior
```

So when analyzing RAGFlow retriever behavior, the first question should be:

> Is the issue in retrieval orchestration, backend search, filtering, ranking, pagination, or result aggregation?

The answer determines which file to inspect first.

## Summary

RAGFlow's retriever is centered around `Dealer` in `rag/nlp/search.py`.

The global `common.settings.retriever` instance is initialized as `rag.nlp.search.Dealer()`. `Dealer.retrieval()` calls `Dealer.search()`, and `Dealer.search()` delegates the backend query to `self.dataStore.search()`.

In the current settings, `dataStore` is `rag.utils.es_conn.ESConnection()`, implemented in `rag/utils/es_conn.py` and based on `common/doc_store/es_conn_base.py`.

The final retrieval result includes three key fields:

- `total`: total matched documents after search and filtering.
- `chunks`: the paginated chunk results returned for the current request.
- `doc_aggs`: document-level aggregations, useful because many chunks may come from the same document.

Understanding these connections makes it easier to debug retrieval quality, pagination behavior, document aggregation, and backend search performance in RAGFlow.

## References

- [DeepWiki discussion about `total`, `chunks`, and `doc_aggs`](https://deepwiki.com/search/chatreferencefiledtotalchunksd_468764bf-458d-46f5-9a14-78490946c263?mode=fast)