# Pytorch
PyTorch 是一个开源的深度学习框架，由 Facebook 的人工智能研究团队开发和维护。

## 1. 基本概念

### 1.1 张量（Tensor）
张量是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但增加了对 GPU 的支持。张量可以是多维数组，支持各种数学运算。

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
```

### 1.2 自动微分（Autograd）
PyTorch 提供了自动微分功能，通过 autograd 模块实现。它可以自动计算张量的梯度，方便进行反向传播。

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```

backward() 方法用于计算标量函数相对于其输入张量的梯度。如果你直接对一个非标量张量调用 backward()，PyTorch 会报错，因为它不知道如何处理非标量张量的梯度计算。

```python
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = a ** 2

b.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs

grad_output = torch.ones_like(b)
b.backward(grad_output) # Should provide grad for vector
```

## 2. 模块和层（Modules and Layers）
PyTorch 提供了丰富的神经网络模块和层，方便构建复杂的神经网络模型。常用的模块包括 nn.Module、nn.Linear、nn.Conv2d 等。

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
print(model)
```

## 3. 优化器（Optimizers）
PyTorch 提供了多种优化器，用于更新模型参数。常用的优化器包括 SGD、Adam 等。

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

## 4. 数据加载（Data Loading）
PyTorch 提供了强大的数据加载和预处理工具，包括 Dataset 和 DataLoader。这些工具可以方便地加载和处理大规模数据集。

```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_data, batch_labels in dataloader:
    output = model(batch_data)
    loss = loss_fn(output, batch_labels)
    # Training...
```

## 5. GPU 加速
PyTorch 支持 GPU 加速，可以利用 CUDA 进行高效的并行计算。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
data = data.to(device)
labels = labels.to(device)
```