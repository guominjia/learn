# Flowchart

`Dealer.search` is router,
`FulltextQueryer.question` convert question to query,
`ESConnection.search` is instance,
`ElasticSearchConnectionPool.get_conn` return `Elasticsearch` instance

```mermaid
graph LR
  A[Dealer.search]
    A --> B[FulltextQueryer.question]
    A --> C[ESConnection.search] --> D[ESConnectionBase.search] --> E[ElasticSearchConnectionPool.get_conn]

  click A "https://github.com/infiniflow/ragflow/tree/main/rag/nlp/search.py#L74"
  click B "https://github.com/infiniflow/ragflow/tree/main/rag/nlp/query.py#L114"
  click C "https://github.com/infiniflow/ragflow/tree/main/rag/utils/es_conn.py#L39"
  click D "https://github.com/infiniflow/ragflow/tree/main/common/doc_store/es_conn_base.py#L45"
  click E "https://github.com/infiniflow/ragflow/tree/main/common/doc_store/es_conn_pool.py#L66"
```