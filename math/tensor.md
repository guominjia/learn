# Tensor

## 1. 张量的定义
张量是一个多维数组，可以有任意数量的维度。根据维度的不同，张量可以分为以下几类：

- **标量（Scalar）**：零阶张量，即一个单一的数值。例如，3 或 -1.5。
- **向量（Vector）**：一阶张量，即一维数组。例如，[1, 2, 3]。
- **矩阵（Matrix）**：二阶张量，即二维数组。例如，[[1, 2], [3, 4]]。
- **高阶张量（Higher-order Tensor）**：三维及以上的数组。例如，三阶张量 [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]。

## 2. 张量与向量的关系
向量是张量的一种特殊形式。具体来说，向量是一阶张量。张量可以看作是向量的推广，向量是张量的一种特例。

- **向量**：一阶张量，只有一个维度。例如，[1, 2, 3] 是一个包含三个元素的向量。
- **张量**：可以有任意数量的维度。例如，[[1, 2], [3, 4]] 是一个二维张量（矩阵），[[[1, 2], [3, 4]], [[5, 6], [7, 8]]] 是一个三维张量。

## 3. 张量的表示
在 PyTorch 中，张量是通过 torch.Tensor 类来表示的。

## 4. 张量的操作
```python
vector1 = torch.tensor([1.0, 2.0, 3.0])
vector2 = torch.tensor([4.0, 5.0, 6.0])

vector_sum = vector1 + vector2
print(vector_sum)  # 输出: tensor([5., 7., 9.])

dot_product = torch.dot(vector1, vector2)
print(dot_product)  # 输出: tensor(32.)

matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
matrix2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

matrix_sum = matrix1 + matrix2
print(matrix_sum)  # 输出: tensor([[ 6.,  8.], [10., 12.]])

matrix_product = torch.matmul(matrix1, matrix2)
print(matrix_product)  # 输出: tensor([[19., 22.], [43.,
```

## 5. 梯度的定义
梯度是某个标量函数（通常是损失函数）相对于其输入变量的导数。

### 1. backward() 方法
y.backward() 方法用于计算 y 对所有叶子节点（即 requires_grad=True 的张量）的梯度，并将结果存储在这些叶子节点的 grad 属性中。