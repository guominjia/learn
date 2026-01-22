## 架构师
> **架构师核心价值**:
>   - 设计可演进的系统，降低未来变更成本
>       - 定义清晰的抽象层，在其下封装具体的实现
>       - 基于当前需求和团队情况，选择一种具体实现
>       - 明确记录当前选择、切换成本、切换触发条件和备选方案
>       - 通过配置和工厂模式来组装系统，使得未来更换组件时只需修改少量配置代码

### 实例：*基于* **架构师核心价值** *设计及实现RGA系统*
假设业务需求：企业内部知识库，支持多种格式文档，需要高准确率和可接受的响应时间。  
首先定义以下抽象接口：

- DocumentLoader: 文档加载
- TextSplitter: 文本分割
- VectorStore: 向量存储
- Retriever: 检索器
- Generator: 生成器
- RAGSystem: 整个RAG系统

然后，为每个接口提供一个基于LlamaIndex的具体实现，但通过依赖注入的方式，使得可以轻松替换为其他实现（如LangChain、Haystack等）。
最后，提供一个配置，通过配置可以切换不同的实现，并探讨切换成本。

[PYTHON代码参考链接](https://github.com/guominjia/learn/tree/code_study/rag_architect)

#### 架构师在这个例子中的核心工作

##### 1. **设计抽象，而非实现**
```python
# 坏架构：直接使用具体框架
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)  # 直接绑定LlamaIndex

# 好架构：先定义抽象接口
class VectorStore(ABC):
    @abstractmethod
    def search(self, query: str) -> List[Document]: ...
```

##### 2. **量化切换成本，而非猜测未来**
```python
# 在配置中明确记录
switch_cost_estimate = {
    "llamaindex_to_langchain": {
        "estimated_hours": 8,  # 明确的工作量
        "data_migration": "无需迁移",  # 明确的数据影响
        "risk_level": "低"  # 明确的风险
    }
}
```

##### 3. **建立反馈循环，而非一次性决策**
```python
# 基于实际数据做决策
def _should_switch_framework(self):
    # 不是基于"我觉得"，而是基于"数据表明"
    error_rate = self._calculate_error_rate()
    avg_time = self._calculate_avg_response_time()
    return error_rate > 0.1 or avg_time > 3.0
```

##### 4. **设计逃生通道，而非祈祷不出错**
```python
def _try_fallback_framework(self, query: str):
    """当主框架失败时，有明确的备用方案"""
    return {
        "fallback_triggered": True,
        "switch_advice": self._get_switch_advice()  # 具体的切换指南
    }
```

##### 架构师的价值清单

| 交付物 | 价值 |
|--------|------|
| **抽象接口** | 保护业务代码不受框架变更影响 |
| **适配器层** | 让框架切换成为配置变更，而非代码重写 |
| **成本清单** | 让每个决策的代价透明、可评估 |
| **监控指标** | 用数据驱动决策，而非主观判断 |
| **切换协议** | 明确的切换步骤和回滚方案 |

> 总结：**架构师不预测哪个框架会赢，而是设计一个无论哪个框架赢了我们都能轻松跟上的系统。**

## [个人理解](personal-thinking-about-architect.md)
## 参考

- [与AI元宝关于架构师讨论](https://yuanbao.tencent.com/chat/naQivTmsDa/0cdf2ce3-6eff-4899-b5e2-5b4a45aae676)