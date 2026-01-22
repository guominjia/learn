
"""
架构师的核心工作：不是选择框架，而是设计一个让框架选择可逆的架构
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# ==================== 1. 抽象层定义 ====================
# 这部分代码一旦确定，未来3年都不应该改动
# 它定义了系统"做什么"，而不是"怎么做"

class Document(ABC):
    """文档抽象，不依赖任何具体框架"""
    @property
    @abstractmethod
    def content(self) -> str: ...
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]: ...


class VectorStore(ABC):
    """向量存储抽象"""
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Document]: ...
    
    @abstractmethod
    def add_documents(self, documents: List[Document]): ...
    
    @classmethod
    @abstractmethod
    def create_from_documents(cls, documents: List[Document], **kwargs) -> 'VectorStore': ...


class RAGGenerator(ABC):
    """RAG生成器抽象"""
    @abstractmethod
    def generate(self, query: str, context: List[Document]) -> str: ...


class RAGSystem(ABC):
    """完整的RAG系统抽象"""
    @abstractmethod
    def query(self, query: str) -> Dict[str, Any]: ...
    
    @abstractmethod
    def ingest(self, documents: List[Document]): ...


# ==================== 2. 适配器层 ====================
# 这部分代码可以替换，是框架选择的具体实现
# 每个适配器都是独立的，可插拔

# ---------- LlamaIndex 适配器 ----------
class LlamaIndexDocument(Document):
    """LlamaIndex的文档适配器"""
    def __init__(self, llama_document):
        self._doc = llama_document
    
    @property
    def content(self) -> str:
        return self._doc.text
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._doc.metadata or {}


class LlamaIndexVectorStore(VectorStore):
    """LlamaIndex向量存储适配器"""
    
    def __init__(self, index):
        # 这里隐藏了LlamaIndex的具体实现
        self._index = index
        self._retriever = index.as_retriever(similarity_top_k=5)
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        # 适配器模式：将框架接口转换为我们的抽象接口
        nodes = self._retriever.retrieve(query)
        return [LlamaIndexDocument(node) for node in nodes[:top_k]]
    
    def add_documents(self, documents: List[Document]):
        # 如果LlamaIndex不支持动态添加，这里会抛出明确异常
        raise NotImplementedError("LlamaIndex不支持动态添加文档")
    
    @classmethod
    def create_from_documents(cls, documents: List[Document], **kwargs):
        # 从文档创建索引
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import TextNode
        
        # 将我们的Document转换为LlamaIndex的TextNode
        nodes = []
        for doc in documents:
            if isinstance(doc, LlamaIndexDocument):
                nodes.append(doc._doc)
            else:
                # 创建适配节点
                node = TextNode(
                    text=doc.content,
                    metadata=doc.metadata
                )
                nodes.append(node)
        
        index = VectorStoreIndex(nodes, **kwargs)
        return cls(index)


# ---------- LangChain 适配器 ----------
class LangChainDocument(Document):
    """LangChain的文档适配器"""
    def __init__(self, lc_document):
        self._doc = lc_document
    
    @property
    def content(self) -> str:
        return self._doc.page_content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._doc.metadata


class LangChainVectorStore(VectorStore):
    """LangChain向量存储适配器"""
    
    def __init__(self, vectorstore, retriever):
        self._vectorstore = vectorstore
        self._retriever = retriever
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        docs = self._retriever.get_relevant_documents(query)[:top_k]
        return [LangChainDocument(doc) for doc in docs]
    
    def add_documents(self, documents: List[Document]):
        # LangChain通常支持动态添加
        lc_docs = []
        for doc in documents:
            if isinstance(doc, LangChainDocument):
                lc_docs.append(doc._doc)
            else:
                from langchain_core.documents import Document as LCDocument
                lc_docs.append(LCDocument(
                    page_content=doc.content,
                    metadata=doc.metadata
                ))
        self._vectorstore.add_documents(lc_docs)
    
    @classmethod
    def create_from_documents(cls, documents: List[Document], **kwargs):
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document as LCDocument
        
        # 转换为LangChain文档
        lc_docs = []
        for doc in documents:
            if isinstance(doc, LangChainDocument):
                lc_docs.append(doc._doc)
            else:
                lc_docs.append(LCDocument(
                    page_content=doc.content,
                    metadata=doc.metadata
                ))
        
        # 创建向量存储
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=lc_docs,
            embedding=embeddings,
            **kwargs
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        return cls(vectorstore, retriever)


# ==================== 3. 配置与工厂 ====================
# 这部分决定使用哪个框架，但可以随时切换

@dataclass
class RAGConfig:
    """配置类：所有可切换的决策点都在这里"""
    framework: str = "llamaindex"  # 可切换：llamaindex, langchain, haystack
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo"
    vector_store_type: str = "chroma"
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # 切换成本记录
    switch_cost_estimate: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.switch_cost_estimate is None:
            self.switch_cost_estimate = {
                "llamaindex_to_langchain": {
                    "code_changes": "适配器层重写",
                    "estimated_hours": 8,
                    "data_migration": "无需迁移",
                    "risk_level": "低"
                },
                "langchain_to_llamaindex": {
                    "code_changes": "适配器层重写",
                    "estimated_hours": 8,
                    "data_migration": "可能需要重建索引",
                    "risk_level": "中"
                }
            }


class RAGFactory:
    """工厂类：根据配置创建具体组件"""
    
    @staticmethod
    def create_vector_store(documents: List[Document], config: RAGConfig) -> VectorStore:
        """创建向量存储，具体实现由配置决定"""
        if config.framework == "llamaindex":
            return LlamaIndexVectorStore.create_from_documents(documents)
        elif config.framework == "langchain":
            return LangChainVectorStore.create_from_documents(documents)
        else:
            raise ValueError(f"不支持的框架: {config.framework}")
    
    @staticmethod
    def create_generator(config: RAGConfig) -> RAGGenerator:
        """创建生成器，具体实现由配置决定"""
        if config.framework == "llamaindex":
            from .llamaindex_generator import LlamaIndexGenerator
            return LlamaIndexGenerator(config.llm_model)
        elif config.framework == "langchain":
            from .langchain_generator import LangChainGenerator
            return LangChainGenerator(config.llm_model)
        else:
            raise ValueError(f"不支持的框架: {config.framework}")


# ==================== 4. 具体的RAG系统实现 ====================

class SimpleRAGSystem(RAGSystem):
    """具体的RAG系统实现，但通过抽象层隔离框架依赖"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store: Optional[VectorStore] = None
        self.generator: Optional[RAGGenerator] = None
        self.query_history = []
        
        # 监控指标
        self.metrics = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "framework_errors": 0,
            "last_framework_check": datetime.now()
        }
    
    def ingest(self, documents: List[Document]):
        """摄取文档"""
        self.vector_store = RAGFactory.create_vector_store(documents, self.config)
        self.generator = RAGFactory.create_generator(self.config)
    
    def query(self, query: str) -> Dict[str, Any]:
        """查询"""
        start_time = datetime.now()
        
        try:
            # 1. 检索
            context_docs = self.vector_store.search(query)
            
            # 2. 生成
            answer = self.generator.generate(query, context_docs)
            
            # 3. 记录结果
            response_time = (datetime.now() - start_time).total_seconds()
            self._record_query(query, answer, response_time, True)
            
            return {
                "answer": answer,
                "sources": [doc.metadata for doc in context_docs],
                "response_time": response_time,
                "framework": self.config.framework,
                "success": True
            }
            
        except Exception as e:
            # 记录框架错误
            self.metrics["framework_errors"] += 1
            self._record_query(query, str(e), 0, False)
            
            # 检查是否需要触发框架切换
            if self._should_switch_framework():
                return self._try_fallback_framework(query)
            
            raise
    
    def _record_query(self, query: str, answer: str, response_time: float, success: bool):
        """记录查询历史，用于监控和决策"""
        self.query_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "answer": answer,
            "response_time": response_time,
            "success": success,
            "framework": self.config.framework
        })
        
        # 更新指标
        self.metrics["total_queries"] += 1
        if success:
            total_time = self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1)
            self.metrics["avg_response_time"] = (total_time + response_time) / self.metrics["total_queries"]
    
    def _should_switch_framework(self) -> bool:
        """判断是否需要切换框架的决策逻辑"""
        # 基于实际指标的决策，而不是猜测
        recent_queries = [q for q in self.query_history[-100:] if q["timestamp"] > datetime.now().timestamp() - 3600]
        
        if len(recent_queries) < 10:
            return False
        
        # 计算错误率
        error_rate = len([q for q in recent_queries if not q["success"]]) / len(recent_queries)
        
        # 检查性能
        avg_time = sum(q["response_time"] for q in recent_queries if q["success"]) / len(recent_queries)
        
        # 切换条件（可配置）
        switch_conditions = {
            "error_rate_too_high": error_rate > 0.1,  # 错误率超过10%
            "response_too_slow": avg_time > 3.0,  # 平均响应超过3秒
            "framework_errors_high": self.metrics["framework_errors"] > 10  # 框架错误超过10次
        }
        
        return any(switch_conditions.values())
    
    def _try_fallback_framework(self, query: str) -> Dict[str, Any]:
        """尝试切换到备用框架"""
        print(f"⚠️ 检测到框架问题，尝试切换到备用框架...")
        
        # 这里可以实现热切换逻辑
        # 当前简单返回错误信息
        return {
            "answer": "系统正在优化，请稍后再试",
            "sources": [],
            "response_time": 0,
            "framework": self.config.framework,
            "success": False,
            "fallback_triggered": True,
            "switch_advice": self._get_switch_advice()
        }
    
    def _get_switch_advice(self) -> Dict[str, Any]:
        """提供框架切换的具体建议"""
        current = self.config.framework
        target = "langchain" if current == "llamaindex" else "llamaindex"
        
        return {
            "current_framework": current,
            "recommended_framework": target,
            "estimated_effort": self.config.switch_cost_estimate.get(f"{current}_to_{target}", {}),
            "steps": [
                "1. 更新配置中的framework字段",
                "2. 重启服务（适配器会自动切换）",
                "3. 监控新框架的性能指标"
            ],
            "rollback_steps": [
                "1. 恢复原配置",
                "2. 重启服务"
            ]
        }
    
    def switch_framework(self, new_framework: str):
        """动态切换框架（演示用，生产环境需要更复杂的迁移）"""
        print(f"🔄 切换框架: {self.config.framework} -> {new_framework}")
        
        # 记录切换决策
        switch_log = {
            "timestamp": datetime.now(),
            "from": self.config.framework,
            "to": new_framework,
            "reason": self.metrics,
            "history_size": len(self.query_history)
        }
        
        # 更新配置
        old_config = self.config
        self.config = RAGConfig(
            framework=new_framework,
            embedding_model=old_config.embedding_model,
            llm_model=old_config.llm_model
        )
        
        # 在实际系统中，这里需要重新初始化组件
        # 但因为我们有抽象层，只需用新框架重新创建即可
        if self.vector_store and hasattr(self.vector_store, '_documents'):
            documents = getattr(self.vector_store, '_documents', [])
            self.ingest(documents)
        
        return switch_log


# ==================== 5. 使用示例 ====================

def main():
    """演示如何使用这个可演进的RAG系统"""
    
    # 1. 定义配置 - 这是唯一的决策点
    config = RAGConfig(
        framework="llamaindex",  # 今天选择LlamaIndex
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4-turbo"
    )
    
    print(f"🎯 初始配置: 使用 {config.framework}")
    print(f"📊 切换成本预估: {json.dumps(config.switch_cost_estimate, indent=2, ensure_ascii=False)}")
    
    # 2. 创建RAG系统
    rag = SimpleRAGSystem(config)
    
    # 3. 创建一些示例文档
    class SimpleDocument(Document):
        def __init__(self, content: str, source: str = ""):
            self._content = content
            self._metadata = {"source": source, "id": hash(content)}
        
        @property
        def content(self) -> str:
            return self._content
        
        @property
        def metadata(self) -> Dict[str, Any]:
            return self._metadata
    
    documents = [
        SimpleDocument("LangChain是一个用于开发大语言模型应用的框架", "doc1"),
        SimpleDocument("LlamaIndex是一个专门为RAG任务优化的框架", "doc2"),
        SimpleDocument("向量数据库用于存储和检索嵌入向量", "doc3")
    ]
    
    # 4. 摄取文档
    rag.ingest(documents)
    
    # 5. 查询
    result = rag.query("什么是RAG框架？")
    print(f"\n🔍 查询结果:")
    print(f"   答案: {result['answer'][:100]}...")
    print(f"   响应时间: {result['response_time']:.2f}秒")
    print(f"   使用框架: {result['framework']}")
    
    # 6. 演示切换决策
    print(f"\n📈 当前指标:")
    print(f"   总查询数: {rag.metrics['total_queries']}")
    print(f"   平均响应时间: {rag.metrics['avg_response_time']:.2f}秒")
    print(f"   框架错误数: {rag.metrics['framework_errors']}")
    
    # 7. 如果需要切换，获取具体建议
    if rag._should_switch_framework():
        advice = rag._get_switch_advice()
        print(f"\n🔄 切换建议:")
        print(f"   当前框架: {advice['current_framework']}")
        print(f"   推荐框架: {advice['recommended_framework']}")
        print(f"   预估工作量: {advice['estimated_effort'].get('estimated_hours', '未知')} 小时")
    
    # 8. 架构师的价值体现
    print(f"\n🏗️ 架构师设计的价值:")
    print(f"   1. 抽象层保护: 业务逻辑不依赖具体框架")
    print(f"   2. 可逆决策: 切换框架只需改配置，不需重写业务代码")
    print(f"   3. 数据驱动: 基于实际指标决定是否切换，而非猜测")
    print(f"   4. 成本透明: 每个决策的切换成本都有明确预估")


# ==================== 6. 架构师的额外工作 ====================

class RAGSystemMonitor:
    """监控系统，收集数据以支持架构决策"""
    
    def __init__(self, rag_system: SimpleRAGSystem):
        self.rag = rag_system
        self.decision_log = []
    
    def evaluate_framework_decision(self) -> Dict[str, Any]:
        """基于数据评估当前框架选择是否正确"""
        
        if len(self.rag.query_history) < 20:
            return {"status": "insufficient_data", "recommendation": "继续收集数据"}
        
        # 分析性能指标
        recent = self.rag.query_history[-20:]
        success_rate = len([q for q in recent if q["success"]]) / len(recent)
        avg_time = sum(q["response_time"] for q in recent if q["success"]) / len(recent)
        
        # 与SLO对比
        meets_slo = {
            "success_rate": success_rate >= 0.95,
            "response_time": avg_time <= 3.0,
            "framework_stability": self.rag.metrics["framework_errors"] < 5
        }
        
        recommendation = "保持当前框架"
        if not all(meets_slo.values()):
            recommendation = f"考虑切换到{self.rag._get_switch_advice()['recommended_framework']}"
        
        evaluation = {
            "timestamp": datetime.now(),
            "current_framework": self.rag.config.framework,
            "metrics": {
                "success_rate": success_rate,
                "avg_response_time": avg_time,
                "framework_errors": self.rag.metrics["framework_errors"]
            },
            "meets_slo": meets_slo,
            "recommendation": recommendation,
            "data_points": len(recent)
        }
        
        self.decision_log.append(evaluation)
        return evaluation


if __name__ == "__main__":
    main()
