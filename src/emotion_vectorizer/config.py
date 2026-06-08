"""情绪向量化模块配置文件"""

from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG: Dict[str, Any] = {
    "engine": "neuro_symbolic",          # 默认引擎
    "model_name": "deepseek",
    "api_url": "https://api.deepseek.com",
    "api_key": "sk-your-api-key-here",
    "temperature": 0.1,
    "batch_size": 16,
    "rule_weight": 0.7,                  # 符号规则权重
    "use_causal_chain": False,           # 是否构建事件因果链
    "text_weight": 0.5,                  # 多模态文本权重
    "behavior_weight": 0.3,              # 多模态行为权重
    "social_weight": 0.2,                # 多模态社交权重
}

# 引擎映射表
ENGINE_MAP: Dict[str, str] = {
    "neuro_symbolic": "NeuroSymbolicVectorizer",
    "multimodal": "MultimodalVectorizer",
}

def get_emotion_vectorizer_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """获取情绪向量化器配置（合并默认配置与用户覆盖）"""
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config
