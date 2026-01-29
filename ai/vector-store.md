## ChromaDB传参

### 1. **输入参数类型不同**

```python
# 方式1：传入列表
embedding_function([query_text])  # 输入: ["your query"]

# 方式2：传入字符串
embedding_function(query_text)    # 输入: "your query"
```

### 2. **返回结果结构不同**

```python
# 方式1：返回二维列表/数组
query_embeddings = embedding_function(["query1", "query2"])
# 返回: [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # shape: (2, embedding_dim)

# 方式2：基于每个字符返回嵌入，query1会返回6个嵌入
query_embeddings = embedding_function("query1")
# 返回: [[0.1, 0.2, ...], ...]  # shape: (6, embedding_dim)
```

**ChromaDB 的 `collection.query()` 方法要求 `query_embeddings` 参数必须是二维结构**，即使只查询一个文本也需要：

```python
# ✅ 正确：二维列表
query_embeddings = [[0.1, 0.2, 0.3, ...]]  # shape: (1, embedding_dim)

# ❌ 错误：一维列表
query_embeddings = [0.1, 0.2, 0.3, ...]    # shape: (embedding_dim,)
```

## 为什么是二维结构？

ChromaDB 设计为支持**批量查询**：

```python
# 批量查询示例（3个查询）
results = collection.query(
    query_embeddings=[
        [0.1, 0.2, ...],  # 查询1
        [0.3, 0.4, ...],  # 查询2
        [0.5, 0.6, ...]   # 查询3
    ],
    n_results=5
)

# 返回结构：
{
    'documents': [
        ["doc1", "doc2", ...],  # 查询1的5个结果
        ["doc6", "doc7", ...],  # 查询2的5个结果
        ["doc11", "doc12", ...]  # 查询3的5个结果
    ],
    'metadatas': [...],  # 同样是二维
    'distances': [...]   # 同样是二维
}
```

## ChromaDB返回

ChromaDB 的 `collection.query()` **始终返回支持批量查询的结构**，即使只查询一个文本：

```python
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],  # 输入：1个查询
    n_results=5
)

# 返回结构：
{
    'documents': [
        ["doc1", "doc2", "doc3", "doc4", "doc5"]  # 第1个查询的结果列表
    ],
    'metadatas': [
        [meta1, meta2, meta3, meta4, meta5]       # 第1个查询的元数据列表
    ],
    'distances': [
        [0.12, 0.15, 0.18, 0.21, 0.24]            # 第1个查询的距离列表
    ]
}
```

## 总结

| 操作 | 结构 | 原因 |
|------|------|------|
| `embedding_function([query_text])` | 输入必须是列表 | ChromaDB 要求 `query_embeddings` 是二维 |
| `results['documents'][i]` | 必须用 `[i]` 索引 | ChromaDB 返回值**永远是批量结构** |

这是 ChromaDB 的 API 设计决策，为了统一处理单查询和多查询场景，**始终使用批量格式**。