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
collection.query(query_texts, query_embeddings)
```

## Max size and Max batch size limitation


- Below code will fail for 10000 and show `exceeds maximum batch size 5461`
- Below code Will fail and show `Payload too large` when payload is too large

```python
import sys
from chromadb import HttpClient

def test_payload_size():
    client = HttpClient(host="localhost", port=8000)
    collection = client.get_or_create_collection("test")

    sizes = [1, 10, 100, 1000, 10000]
    for size in sizes:
        try:
            docs = ["x" * 1000] * size  # 1KB per document
            collection.add(documents=docs, ids=[str(i) for i in range(size)])
            print(f"Size {size} documents: OK ({size * 1000 / 1024:.2f}KB)")
        except Exception as e:
            print(f"Size {size} documents: FAILED")
            print(f"Max size is around {(size-1) * 1000 / 1024:.2f}KB")
            break
```

## References

- <https://docs.trychroma.com/>, <https://github.com/chroma-core/chroma>