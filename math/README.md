# Mathematics

## [Tensor](tensor.md)

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

## 余弦相似度公式

**余弦相似度公式：**
```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

### 📊 完整计算示例：

```python
import numpy as np

# 3个简单的2维embeddings
embeddings = np.array([
    [3, 4],   # embedding A, ‖A‖ = 5
    [0, 1],   # embedding B, ‖B‖ = 1
    [5, 0]    # embedding C, ‖C‖ = 5
])

# 步骤1: 计算点积矩阵
dot_product = np.dot(embeddings, embeddings.T)
print("点积矩阵 (A·B):")
print(dot_product)
# [[ 25   4  15]   # A·A=25, A·B=4,  A·C=15
#  [  4   1   0]   # B·A=4,  B·B=1,  B·C=0
#  [ 15   0  25]]  # C·A=15, C·B=0,  C·C=25

# 步骤2: 计算范数
norms = np.linalg.norm(embeddings, axis=1)
print("\n范数 (‖·‖):")
print(norms)
# [5. 1. 5.]  # ‖A‖=5, ‖B‖=1, ‖C‖=5

# 步骤3: 计算范数外积 (‖A‖ × ‖B‖)
norms_outer = np.outer(norms, norms)
print("\n范数外积 (‖A‖×‖B‖):")
print(norms_outer)
# [[25.  5. 25.]   # 5×5  5×1  5×5
#  [ 5.  1.  5.]   # 1×5  1×1  1×5
#  [25.  5. 25.]]  # 5×5  5×1  5×5

# 步骤4: 计算余弦相似度
cosine_sim = dot_product / norms_outer
print("\n余弦相似度 (A·B / ‖A‖×‖B‖):")
print(cosine_sim)
# [[1.   0.8  0.6 ]   # cos(A,A)=1.0, cos(A,B)=0.8, cos(A,C)=0.6
#  [0.8  1.   0.  ]   # cos(B,A)=0.8, cos(B,B)=1.0, cos(B,C)=0.0
#  [0.6  0.   1.  ]]  # cos(C,A)=0.6, cos(C,B)=0.0, cos(C,C)=1.0
```