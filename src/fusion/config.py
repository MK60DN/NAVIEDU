"""时空融合模块配置文件"""

from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "dct_gnn",                      # 默认引擎
    "gnn_hidden_dim": 64,
    "gnn_num_layers": 3,
    "gnn_learning_rate": 0.001,
    "gnn_dropout": 0.2,
    "gnn_num_epochs": 200,
    "gnn_weight_decay": 0.0005,
    "spatial_feature_dim": 8,
    "temporal_feature_dim": 8,
    "output_dim": 16,
    "forecast_horizon": 1,
    "forecast_method": "gcn",
}

# 引擎映射表
# 已更新为新的 PyTorch 版 DCTGNNModule 类名
ENGINE_MAP: Dict[str, str] = {
    "dct_gnn": "DCTGNNModule",
    "neural_temporal_logic": "NeuralTemporalLogicFusion",
}

def get_fusion_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取融合模块配置（合并默认配置与用户覆盖）"""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
