"""空间拓扑分析模块

该模块负责分析群聊成员情绪状态的空间拓扑结构。
它使用持续拉普拉斯算子（Persistent Laplacian）来检测情绪点云中的
聚类、环状结构和离群点，揭示群体情绪的"形状"。

学术基础：
- 持续拉普拉斯算子 (Mémoli, 2011): 比经典持续同调提供更精细的拓扑流形信息
- 谱图理论 (Chung, 1997): 通过拉普拉斯算子的特征值分析图的结构
- HDBSCAN 聚类算法 (Campello, Moulavi & Sander, 2013)

可替换引擎：
- persistent_laplacian（默认）：基于持续拉普拉斯算子的谱分析方法
               优势：能检测到经典持续同调无法发现的细微结构差异
- topological_gradient_flow：基于拓扑梯度流的方法
               优势：能够追踪情绪结构在时间尺度上的动态演变过程
"""

from tender.topology_analysis.base import BaseTopologyAnalyzer, TopologyResult
from tender.topology_analysis.persistent_laplacian import (
    PersistentLaplacianAnalyzer,
)
from tender.topology_analysis.topological_gradient_flow import (
    TopologicalGradientFlowAnalyzer,
)
from tender.topology_analysis.config import (
    DEFAULT_CONFIG,
    ENGINE_MAP,
    get_topology_config,
)

__all__ = [
    "BaseTopologyAnalyzer",
    "TopologyResult",
    "PersistentLaplacianAnalyzer",
    "TopologicalGradientFlowAnalyzer",
    "DEFAULT_CONFIG",
    "ENGINE_MAP",
    "get_topology_config",
]
