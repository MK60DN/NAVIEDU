"""时空融合模块

该模块负责将空间拓扑分析结果和时间因果分析结果融合为统一的特征向量，
并进行下一时间窗口的情绪预测。

学术基础：
- 图卷积网络 (Kipf & Welling, 2017): 将卷积操作推广到图结构数据
- 动态图网络 (Pareja et al., 2020): 处理随时间演化的图结构
- 可微时序逻辑 (Li et al., 2020): 将逻辑规则转化为可微分的数学表达式

可替换引擎：
- dct_gnn（默认）：动态因果图神经网络融合
               优势：端到端学习时空依赖关系，预测精度高
- neural_temporal_logic：神经时序逻辑融合
               优势：高可解释性，每个规则有明确的语义含义
"""

from tender.fusion.base import BaseFusionModule, FusionResult
from tender.fusion.dct_gnn import DCTGNN
from tender.fusion.neural_temporal_logic import NeuralTemporalLogicFusion
from tender.fusion.config import (
    DEFAULT_CONFIG,
    ENGINE_MAP,
    get_fusion_config,
)

__all__ = [
    "BaseFusionModule",
    "FusionResult",
    "DCTGNN",
    "NeuralTemporalLogicFusion",
    "DEFAULT_CONFIG",
    "ENGINE_MAP",
    "get_fusion_config",
]
