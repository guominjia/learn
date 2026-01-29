## `cosine_similarity` 函数三种实现方式对比：

### 1️⃣ **NumPy 实现**（推荐，无需额外依赖）
```python
# 方法1: 归一化 + 点积
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms
similarity_matrix = np.dot(normalized, normalized.T)

# 方法2: 手动计算（更清晰但稍慢）
def cosine_similarity_np(embeddings):
    dot_product = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1)
    similarity_matrix = dot_product / np.outer(norms, norms)
    return similarity_matrix
```

### 2️⃣ **sklearn 实现**
```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
```

### 3️⃣ **PyTorch 实现**（如果使用 GPU）
```python
import torch
embeddings_tensor = torch.tensor(embeddings)
similarity_matrix = torch.nn.functional.cosine_similarity(
    embeddings_tensor.unsqueeze(1), 
    embeddings_tensor.unsqueeze(0), 
    dim=2
).numpy()
```

## 性能对比：

| 方法 | 速度 | 内存 | 依赖 |
|------|------|------|------|
| NumPy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 无额外依赖 |
| sklearn | ⭐⭐⭐ | ⭐⭐⭐ | 需要 scikit-learn |
| PyTorch | ⭐⭐⭐⭐⭐ (GPU) | ⭐⭐⭐ | 需要 PyTorch |