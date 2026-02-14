# Chromadb Python Package

```python
client = chromadb.PersistentClient(path=chroma_path)
client = chromadb.HttpClient(host=chroma_host, port=chroma_port, ssl=False)
collection = client.get_collection(name=chroma_collection)
collection = client.get_or_create_collection(
    name=chroma_collection,
    embedding_function=EmbeddingFunction(),
    metadata={"hnsw:space": "cosine"}
)
client.list_collections()
collection.get(where, where_document)
```