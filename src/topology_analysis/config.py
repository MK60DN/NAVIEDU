"""空间拓扑分析模块配置文件"""

from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "persistent_laplacian",         # 默认引擎
    "normalize": True,
    "spectral_gap_threshold": 0.1,
    "eigenvalue_count": 10,
    "laplacian_type": "normalized",
    "min_cluster_size": 2,
    "min_samples": 1,
    "h1_threshold_ratio": 0.3,
    "metric": "euclidean",
    "standardize": True,
    "max_edge_length": 2.0,
    "num_scale_steps": 20,
}

# 引擎映射表（已移除旧的 PersistentHomologyAnalyzer）
ENGINE_MAP: Dict[str, str] = {
    "persistent_laplacian": "PersistentLaplacianAnalyzer",
    "topological_gradient_flow": "TopologicalGradientFlowAnalyzer",
}

def get_topology_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取拓扑分析配置（合并默认配置与用户覆盖）"""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
