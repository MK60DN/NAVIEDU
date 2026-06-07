"""
认知状态分析模块

该模块实现了对群体成员认知状态的独立分析能力。
旨在从多个维度（知识图谱、行为模式、混合特征、神经网络嵌入）量化群体的认知状态。

核心方法：
1. knowledge_state (knowledge_state)：基于预设知识图谱的认知状态分析
   核心逻辑：根据成员在知识点图谱中的位置，分析理解水平、认知负荷和认知阶段
2. behavior_state (behavior_state)：基于行为模式的认知状态分析
   核心逻辑：根据成员的交互行为（发言频率、提问、回答等），推断认知状态
3. hybrid_state (hybrid_state)：混合认知状态分析
   核心逻辑：融合知识图谱信息和行为模式，获得更全面的认知画像
4. neural_state (neural_state)：基于神经网络的隐状态分析
   核心逻辑：使用预训练的轻量级神经网络，直接从文本中提取认知状态嵌入

可替换引擎：
- knowledge_state（默认）：基于知识图谱的认知状态分析
               优势：高可解释性，能够精确判断认知阶段和知识点掌握情况
- behavior_state：基于行为模式的认知状态分析
               优势：无需预设知识图谱，适用于开放域场景
- hybrid_state：混合认知状态分析
               优势：融合知识图谱和行为信息，综合分析更全面
- neural_state：基于神经网络的隐状态分析
               优势：端到端训练，能够捕获深层语义特征

学术基础：
- 认知诊断模型 (Corbett & Anderson, 1995): 知识追踪与认知状态推断
- 知识图谱嵌入 (Bordes et al., 2013): 将知识图谱节点映射到低维向量空间
- 行为认知模型 (Baker et al., 2010): 基于行为模式的认知状态推断
"""

from tender.cognition.base import (
    BaseCognitionAnalyzer,
    CognitionState,
    KnowledgeNode,
    BehaviorProfile,
    CognitionTrace,
    KnowledgeGraphConfig,
)
from tender.cognition.knowledge_state import KnowledgeStateAnalyzer
from tender.cognition.behavior_state import BehaviorStateAnalyzer
from tender.cognition.hybrid_state import HybridStateAnalyzer
from tender.cognition.neural_state import NeuralStateAnalyzer
from tender.cognition.config import (
    DEFAULT_CONFIG,
    ENGINE_MAP,
    get_cognition_config,
    get_cognition_analyzer,
)

__all__ = [
    "BaseCognitionAnalyzer",
    "CognitionState",
    "KnowledgeNode",
    "BehaviorProfile",
    "CognitionTrace",
    "KnowledgeGraphConfig",
    "KnowledgeStateAnalyzer",
    "BehaviorStateAnalyzer",
    "HybridStateAnalyzer",
    "NeuralStateAnalyzer",
    "DEFAULT_CONFIG",
    "ENGINE_MAP",
    "get_cognition_config",
    "get_cognition_analyzer",
]
