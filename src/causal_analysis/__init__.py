"""时间因果分析模块

该模块负责分析群聊成员情绪的因果影响关系，构建情绪因果网络。
它通过检测成员间情绪状态的时间依赖关系，识别超级传播者和关键接收者。

学术基础：
- 收敛交叉映射 (Sugihara et al., 2012): 基于状态空间重构的非线性因果检测
- 结构因果模型 (Pearl, 2009): 基于有向无环图的因果推断框架
- PC+LiNGAM (Shimizu et al., 2006): 结合条件独立性与非高斯性的混合方法

可替换引擎：
- convergent_cross_mapping（默认）：收敛交叉映射分析器
               优势：适合短时间窗口，能捕捉非线性因果关系
- structural_causal_model：结构因果模型分析器
               优势：支持干预模拟和反事实推理
- pc_lingsam：PC算法+LiNGAM分析器
               优势：无需时间序列结构，适合横截面数据
"""

from tender.causal_analysis.base import BaseCausalAnalyzer, CausalResult, CausalEdge
from tender.causal_analysis.convergent_cross_mapping import (
    ConvergentCrossMappingAnalyzer,
)
from tender.causal_analysis.structural_causal_model import (
    StructuralCausalModelAnalyzer,
)
from tender.causal_analysis.pc_lingsam import (
    PCLiNGAMAnalyzer,
)
from tender.causal_analysis.config import (
    DEFAULT_CONFIG,
    ENGINE_MAP,
    get_causal_config,
)

__all__ = [
    "BaseCausalAnalyzer",
    "CausalResult",
    "CausalEdge",
    "ConvergentCrossMappingAnalyzer",
    "StructuralCausalModelAnalyzer",
    "PCLiNGAMAnalyzer",
    "DEFAULT_CONFIG",
    "ENGINE_MAP",
    "get_causal_config",
]
