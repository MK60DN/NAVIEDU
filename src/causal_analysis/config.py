"""时间因果分析模块配置文件"""

from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "convergent_cross_mapping",      # 默认引擎
    "embedding_dimension": 5,
    "tau": 1,
    "lib_size_ratio": 0.8,
    "significance_level": 0.05,
    "emotion_dimension": "composite",
    "max_lag": 5,
    "num_lib_sizes": 10,
    "seed": 42,
}

# 引擎映射表
ENGINE_MAP: Dict[str, str] = {
    "convergent_cross_mapping": "ConvergentCrossMappingAnalyzer",
    "structural_causal_model": "StructuralCausalModelAnalyzer",
    "pc_lingsam": "PCLiNGAMAnalyzer",
}

def get_causal_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取因果分析配置（合并默认配置与用户覆盖）"""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
