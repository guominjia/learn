"""
Agent Memory System - 基于Forms-Functions-Dynamics框架的完整实现
参考论文: Memory in the Age of AI Agents: A Survey
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib


# ============================================================================
# FORMS: 记忆的形式/载体
# ============================================================================

@dataclass
class MemoryUnit:
    """基础记忆单元"""
    id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5
    access_count: int = 0
    

class TokenLevelMemory:
    """Token-level Memory: 显式的、可检索的记忆"""
    
    def __init__(self):
        self.flat_memory: List[MemoryUnit] = []  # 1D: 平面记忆
        self.planar_memory: Dict[str, List[str]] = defaultdict(list)  # 2D: 图结构
        self.hierarchical_memory: Dict[str, Any] = {  # 3D: 层次结构
            'abstract': [],
            'detailed': [],
            'links': {}
        }
    
    def add_flat(self, unit: MemoryUnit):
        """添加到平面记忆"""
        self.flat_memory.append(unit)
    
    def add_planar(self, unit_id: str, related_ids: List[str]):
        """添加到图结构记忆"""
        self.planar_memory[unit_id].extend(related_ids)
    
    def add_hierarchical(self, unit: MemoryUnit, level: str = 'detailed'):
        """添加到层次记忆"""
        self.hierarchical_memory[level].append(unit)
    
    def get_all_flat(self) -> List[MemoryUnit]:
        return self.flat_memory
    
    def get_related(self, unit_id: str) -> List[str]:
        """获取相关记忆"""
        return self.planar_memory.get(unit_id, [])


class ParametricMemory:
    """Parametric Memory: 参数化记忆（模拟权重更新）"""
    
    def __init__(self, dim: int = 128):
        self.parameters = np.random.randn(dim, dim) * 0.01
        self.update_history = []
    
    def update(self, gradient: np.ndarray, lr: float = 0.001):
        """模拟参数更新"""
        self.parameters += lr * gradient
        self.update_history.append({
            'timestamp': datetime.now(),
            'magnitude': np.linalg.norm(gradient)
        })
    
    def encode(self, input_vec: np.ndarray) -> np.ndarray:
        """通过参数编码信息"""
        return np.tanh(self.parameters @ input_vec)


class LatentMemory:
    """Latent Memory: 潜在状态记忆"""
    
    def __init__(self, hidden_dim: int = 256):
        self.hidden_state = np.zeros(hidden_dim)
        self.cell_state = np.zeros(hidden_dim)
        self.attention_keys = []
        self.attention_values = []
    
    def update_state(self, input_vec: np.ndarray):
        """更新隐状态（简化的LSTM式更新）"""
        forget_gate = 1 / (1 + np.exp(-np.dot(self.hidden_state, input_vec[:len(self.hidden_state)])))
        self.cell_state = forget_gate * self.cell_state + (1 - forget_gate) * input_vec[:len(self.cell_state)]
        self.hidden_state = np.tanh(self.cell_state)
    
    def add_attention(self, key: np.ndarray, value: np.ndarray):
        """添加注意力键值对"""
        self.attention_keys.append(key)
        self.attention_values.append(value)


# ============================================================================
# FUNCTIONS: 记忆的功能
# ============================================================================

class FactualMemory:
    """事实记忆: 记录来自用户与环境交互的知识"""
    
    def __init__(self):
        self.user_preferences: Dict[str, Any] = {}
        self.environment_facts: Dict[str, Any] = {}
        self.interaction_history: List[Dict] = []
    
    def add_user_preference(self, key: str, value: Any):
        """添加用户偏好"""
        self.user_preferences[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'confidence': 1.0
        }
    
    def add_environment_fact(self, key: str, value: Any, source: str = 'interaction'):
        """添加环境事实"""
        self.environment_facts[key] = {
            'value': value,
            'source': source,
            'timestamp': datetime.now()
        }
    
    def get_user_preference(self, key: str) -> Optional[Any]:
        return self.user_preferences.get(key, {}).get('value')
    
    def get_environment_fact(self, key: str) -> Optional[Any]:
        return self.environment_facts.get(key, {}).get('value')


class ExperientialMemory:
    """经验记忆: 从任务执行中增量提升解决问题的能力"""
    
    def __init__(self):
        self.case_based: List[Dict] = []  # 基于案例的记忆
        self.strategy_based: Dict[str, List[str]] = {}  # 基于策略的记忆
        self.skill_based: Dict[str, Any] = {}  # 基于技能的记忆
    
    def add_case(self, task: str, actions: List[str], outcome: str, success: bool):
        """添加案例记忆"""
        self.case_based.append({
            'task': task,
            'actions': actions,
            'outcome': outcome,
            'success': success,
            'timestamp': datetime.now()
        })
    
    def add_strategy(self, task_type: str, strategy: str):
        """添加策略记忆"""
        if task_type not in self.strategy_based:
            self.strategy_based[task_type] = []
        self.strategy_based[task_type].append(strategy)
    
    def add_skill(self, skill_name: str, code: str, description: str):
        """添加技能记忆"""
        self.skill_based[skill_name] = {
            'code': code,
            'description': description,
            'usage_count': 0,
            'success_rate': 0.0
        }
    
    def get_similar_cases(self, task: str, top_k: int = 3) -> List[Dict]:
        """获取相似案例"""
        # 简化实现：返回最近的成功案例
        successful_cases = [c for c in self.case_based if c['success']]
        return sorted(successful_cases, key=lambda x: x['timestamp'], reverse=True)[:top_k]


class WorkingMemory:
    """工作记忆: 管理单个任务实例中的工作区信息"""
    
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.current_context: List[str] = []
        self.compressed_context: Optional[str] = None
        self.task_state: Dict[str, Any] = {}
    
    def add_to_context(self, item: str):
        """添加到当前上下文"""
        self.current_context.append(item)
        if len(self.current_context) > self.capacity:
            self._compress()
    
    def _compress(self):
        """压缩上下文"""
        # 简化实现：只保留最近的items
        overflow = self.current_context[:-self.capacity]
        self.compressed_context = " | ".join(overflow)
        self.current_context = self.current_context[-self.capacity:]
    
    def get_context(self) -> str:
        """获取完整上下文"""
        parts = []
        if self.compressed_context:
            parts.append(f"[压缩历史] {self.compressed_context}")
        parts.extend(self.current_context)
        return "\n".join(parts)
    
    def update_task_state(self, key: str, value: Any):
        """更新任务状态"""
        self.task_state[key] = value
    
    def clear(self):
        """清空工作记忆"""
        self.current_context = []
        self.compressed_context = None
        self.task_state = {}


# ============================================================================
# DYNAMICS: 记忆的动态机制
# ============================================================================

class MemoryFormation:
    """记忆形成: 从原始上下文到可存可取的知识"""
    
    @staticmethod
    def semantic_summarization(text: str, max_length: int = 100) -> str:
        """语义总结"""
        # 简化实现：截取前N个字符
        summary = text[:max_length]
        if len(text) > max_length:
            summary += "..."
        return summary
    
    @staticmethod
    def knowledge_distillation(examples: List[str]) -> str:
        """知识蒸馏"""
        # 简化实现：提取关键模式
        return f"从{len(examples)}个示例中提取的知识模式"
    
    @staticmethod
    def structure_construction(data: Dict) -> Dict:
        """结构化构建"""
        # 将非结构化数据转为结构化
        structured = {
            'entities': [],
            'relations': [],
            'attributes': {}
        }
        # 简化实现
        for key, value in data.items():
            structured['attributes'][key] = value
        return structured


class MemoryEvolution:
    """记忆演化: 整合、冲突消解与剪枝"""
    
    def __init__(self):
        self.conflict_resolution_strategy = 'latest'  # 'latest', 'voting', 'confidence'
    
    def merge_memories(self, mem1: MemoryUnit, mem2: MemoryUnit) -> MemoryUnit:
        """合并相似记忆"""
        merged_content = f"{mem1.content} + {mem2.content}"
        merged_metadata = {**mem1.metadata, **mem2.metadata}
        
        return MemoryUnit(
            id=f"merged_{mem1.id}_{mem2.id}",
            content=merged_content,
            timestamp=max(mem1.timestamp, mem2.timestamp),
            metadata=merged_metadata,
            importance=max(mem1.importance, mem2.importance)
        )
    
    def resolve_conflict(self, conflicting_units: List[MemoryUnit]) -> MemoryUnit:
        """冲突消解"""
        if self.conflict_resolution_strategy == 'latest':
            return max(conflicting_units, key=lambda x: x.timestamp)
        elif self.conflict_resolution_strategy == 'confidence':
            return max(conflicting_units, key=lambda x: x.importance)
        else:
            return conflicting_units[0]
    
    def prune(self, memories: List[MemoryUnit], threshold: float = 0.3) -> List[MemoryUnit]:
        """剪枝低重要性记忆"""
        return [m for m in memories if m.importance >= threshold]
    
    def update_importance(self, memory: MemoryUnit, decay_rate: float = 0.95):
        """更新重要性（时间衰减）"""
        time_diff = (datetime.now() - memory.timestamp).days
        memory.importance *= (decay_rate ** time_diff)


class MemoryRetrieval:
    """记忆检索: 决定记忆是否真的能帮助决策"""
    
    def __init__(self):
        self.retrieval_strategies = ['similarity', 'recency', 'importance', 'hybrid']
    
    def similarity_based(self, query: str, memories: List[MemoryUnit], top_k: int = 3) -> List[MemoryUnit]:
        """基于相似度检索"""
        # 简化实现：基于字符串匹配
        scored = [(m, self._simple_similarity(query, m.content)) for m in memories]
        sorted_memories = sorted(scored, key=lambda x: x[1], reverse=True)
        return [m for m, _ in sorted_memories[:top_k]]
    
    def recency_based(self, memories: List[MemoryUnit], top_k: int = 3) -> List[MemoryUnit]:
        """基于最近性检索"""
        sorted_memories = sorted(memories, key=lambda x: x.timestamp, reverse=True)
        return sorted_memories[:top_k]
    
    def importance_based(self, memories: List[MemoryUnit], top_k: int = 3) -> List[MemoryUnit]:
        """基于重要性检索"""
        sorted_memories = sorted(memories, key=lambda x: x.importance, reverse=True)
        return sorted_memories[:top_k]
    
    def hybrid_retrieval(self, query: str, memories: List[MemoryUnit], 
                        weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
                        top_k: int = 3) -> List[MemoryUnit]:
        """混合检索策略"""
        w_sim, w_rec, w_imp = weights
        
        scored = []
        for m in memories:
            sim_score = self._simple_similarity(query, m.content)
            rec_score = self._recency_score(m.timestamp)
            imp_score = m.importance
            
            final_score = w_sim * sim_score + w_rec * rec_score + w_imp * imp_score
            scored.append((m, final_score))
        
        sorted_memories = sorted(scored, key=lambda x: x[1], reverse=True)
        return [m for m, _ in sorted_memories[:top_k]]
    
    @staticmethod
    def _simple_similarity(query: str, content: str) -> float:
        """简单的相似度计算"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        intersection = query_words & content_words
        return len(intersection) / len(query_words)
    
    @staticmethod
    def _recency_score(timestamp: datetime) -> float:
        """计算最近性得分"""
        days_old = (datetime.now() - timestamp).days
        return np.exp(-days_old / 30)  # 30天半衰期


# ============================================================================
# 完整的Agent记忆系统
# ============================================================================

class AgentMemorySystem:
    """集成的Agent记忆系统"""
    
    def __init__(self):
        # Forms
        self.token_memory = TokenLevelMemory()
        self.parametric_memory = ParametricMemory()
        self.latent_memory = LatentMemory()
        
        # Functions
        self.factual_memory = FactualMemory()
        self.experiential_memory = ExperientialMemory()
        self.working_memory = WorkingMemory()
        
        # Dynamics
        self.formation = MemoryFormation()
        self.evolution = MemoryEvolution()
        self.retrieval = MemoryRetrieval()
        
        self.memory_log = []
    
    def process_interaction(self, user_input: str, agent_response: str, 
                          task_type: str = 'general', success: bool = True):
        """处理一次交互，更新各类记忆"""
        
        # 1. Formation: 形成记忆
        summary = self.formation.semantic_summarization(
            f"User: {user_input}\nAgent: {agent_response}"
        )
        
        # 2. 创建记忆单元
        memory_id = hashlib.md5(summary.encode()).hexdigest()[:8]
        memory_unit = MemoryUnit(
            id=memory_id,
            content=summary,
            timestamp=datetime.now(),
            metadata={'task_type': task_type, 'success': success},
            importance=0.8 if success else 0.5
        )
        
        # 3. 存储到不同形式
        self.token_memory.add_flat(memory_unit)
        
        # 4. 更新功能性记忆
        self.factual_memory.interaction_history.append({
            'user_input': user_input,
            'agent_response': agent_response,
            'timestamp': datetime.now()
        })
        
        # 5. 更新工作记忆
        self.working_memory.add_to_context(f"[{task_type}] {summary}")
        
        # 6. 如果成功，添加到经验记忆
        if success:
            self.experiential_memory.add_case(
                task=task_type,
                actions=[agent_response],
                outcome='success',
                success=True
            )
        
        # 7. 记录日志
        self.memory_log.append({
            'timestamp': datetime.now(),
            'memory_id': memory_id,
            'action': 'process_interaction'
        })
        
        return memory_id
    
    def query_memory(self, query: str, memory_type: str = 'all') -> Dict[str, Any]:
        """查询记忆系统"""
        results = {}
        
        if memory_type in ['all', 'token']:
            # 从token记忆检索
            token_results = self.retrieval.hybrid_retrieval(
                query, 
                self.token_memory.get_all_flat(),
                top_k=4
            )
            results['token_memory'] = [
                {'id': m.id, 'content': m.content, 'importance': m.importance}
                for m in token_results
            ]
        
        if memory_type in ['all', 'factual']:
            # 从事实记忆检索
            results['user_preferences'] = self.factual_memory.user_preferences
            results['recent_interactions'] = self.factual_memory.interaction_history[-5:]
        
        if memory_type in ['all', 'experiential']:
            # 从经验记忆检索
            similar_cases = self.experiential_memory.get_similar_cases(query)
            results['similar_cases'] = similar_cases
        
        if memory_type in ['all', 'working']:
            # 从工作记忆获取
            results['current_context'] = self.working_memory.get_context()
        
        return results
    
    def maintain_memory(self):
        """定期维护记忆（演化）"""
        all_memories = self.token_memory.get_all_flat()
        
        # 1. 更新重要性
        for memory in all_memories:
            self.evolution.update_importance(memory)
        
        # 2. 剪枝
        pruned = self.evolution.prune(all_memories, threshold=0.3)
        self.token_memory.flat_memory = pruned
        
        # 3. 合并相似记忆（简化实现）
        # 在实际应用中，这里需要更复杂的相似度判断
        
        print(f"维护完成: 保留 {len(pruned)}/{len(all_memories)} 条记忆")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'token_memory_count': len(self.token_memory.get_all_flat()),
            'factual_memory': {
                'user_preferences': len(self.factual_memory.user_preferences),
                'environment_facts': len(self.factual_memory.environment_facts),
                'interactions': len(self.factual_memory.interaction_history)
            },
            'experiential_memory': {
                'cases': len(self.experiential_memory.case_based),
                'strategies': sum(len(v) for v in self.experiential_memory.strategy_based.values()),
                'skills': len(self.experiential_memory.skill_based)
            },
            'working_memory': {
                'context_items': len(self.working_memory.current_context),
                'compressed': self.working_memory.compressed_context is not None
            }
        }


# ============================================================================
# 使用示例
# ============================================================================

def demo():
    """演示Agent记忆系统的使用"""
    print("=" * 70)
    print("Agent记忆系统演示 - 基于Forms-Functions-Dynamics框架")
    print("=" * 70)
    
    # 创建记忆系统
    agent_memory = AgentMemorySystem()
    
    # 模拟多次交互
    interactions = [
        ("帮我写一个Python排序函数", "好的，我来写一个快速排序...", "coding", True),
        ("我喜欢简洁的代码风格", "好的，我会记住您的偏好", "preference", True),
        ("这个Bug怎么修复？", "让我检查一下...", "debugging", False),
        ("再帮我优化一下性能", "我建议使用缓存...", "optimization", True),
    ]
    
    print("\n📝 处理交互...")
    for user_input, agent_response, task_type, success in interactions:
        memory_id = agent_memory.process_interaction(
            user_input, agent_response, task_type, success
        )
        print(f"  ✓ 已记录: [{task_type}] {memory_id}")
    
    # 添加用户偏好
    print("\n👤 添加用户偏好...")
    agent_memory.factual_memory.add_user_preference("code_style", "简洁")
    agent_memory.factual_memory.add_user_preference("language", "Python")
    print("  ✓ 偏好已保存")
    
    # 添加技能
    print("\n🛠️ 添加技能记忆...")
    agent_memory.experiential_memory.add_skill(
        "quick_sort",
        "def quick_sort(arr): ...",
        "快速排序实现"
    )
    print("  ✓ 技能已保存")
    
    # 查询记忆
    print("\n🔍 查询记忆: '代码优化'")
    query_results = agent_memory.query_memory("代码优化")
    
    print("\n📊 查询结果:")
    if 'token_memory' in query_results:
        print(f"  Token记忆: 找到 {len(query_results['token_memory'])} 条相关记忆")
        for mem in query_results['token_memory'][:2]:
            print(f"    - {mem['content'][:50]}... (重要性: {mem['importance']:.2f})")
    
    if 'similar_cases' in query_results:
        print(f"  经验记忆: 找到 {len(query_results['similar_cases'])} 个相似案例")
    
    # 维护记忆
    print("\n🔧 执行记忆维护...")
    agent_memory.maintain_memory()
    
    # 显示系统状态
    print("\n📈 系统状态:")
    status = agent_memory.get_system_status()
    print(f"  Token记忆数量: {status['token_memory_count']}")
    print(f"  事实记忆: {status['factual_memory']}")
    print(f"  经验记忆: {status['experiential_memory']}")
    print(f"  工作记忆: {status['working_memory']}")
    
    print("\n" + "=" * 70)
    print("✅ 演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    demo()