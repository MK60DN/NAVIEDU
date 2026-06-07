"""策略推理模块配置文件"""

from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "causal_rl",                    # 默认引擎
    "state_dim": 16,
    "action_dim": 7,
    "rl_hidden_dim": 128,                     # DQN 隐藏层维度（新增）
    "rl_num_hidden_layers": 3,                # DQN 隐藏层数量（新增）
    "rl_dropout": 0.2,                        # DQN Dropout 比例（新增）
    "rl_learning_rate": 0.001,                # DQN 学习率（重命名）
    "rl_discount_factor": 0.99,               # 折扣因子（重命名）
    "rl_batch_size": 64,                      # 训练批次大小（重命名）
    "rl_buffer_size": 10000,                  # 经验回放池容量（重命名）
    "rl_tau": 0.005,                          # 目标网络软更新系数（新增）
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "target_update": 10,                      # 目标网络更新频率
    "enable_reconciliation": False,           # 是否启用共识化过滤层
}

# 引擎映射表
# 已更新为新的 PyTorch 版 DQNStrategyEngine 类名
ENGINE_MAP: Dict[str, str] = {
    "causal_rl": "DQNStrategyEngine",
    "llm_strategist": "LLMStrategistEngine",
}

# 共识化过滤层配置
RECONCILIATION_DEFAULT_CONFIG: Dict[str, Any] = {
    "min_confidence": 0.5,
    "max_retries": 3,
    "conflict_resolution": "majority_vote",
}

def get_strategy_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取策略模块配置（合并默认配置与用户覆盖）"""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config

def get_reconciliation_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取共识化过滤层配置（合并默认配置与用户覆盖）"""
    config = RECONCILIATION_DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
