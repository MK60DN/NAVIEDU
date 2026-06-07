""" 情绪-认知协同模块 - 配置与引擎映射（已更新以适配认知模块）

该模块定义了情绪-认知协同模块的配置结构和引擎映射关系。
它遵循 Tender 框架的统一配置规范，支持通过 config.yaml 进行灵活的引擎切换。

主要功能：
1. 定义协同模块的默认配置
2. 建立引擎名称到具体实现类的映射
3. 提供配置加载与验证工具方法
4. 支持基于配置字典创建指定引擎实例

使用示例（配合认知模块）：
    from tender.cognition.config import get_cognition_analyzer
    from tender.synergy.config import get_synergy_engine

    # 1. 先配置认知分析引擎
    cognition_config = {
        "engine": "hybrid_state",
        "feature_dim": 16,
        "use_knowledge_graph": True,
        "knowledge_graph_path": "/path/to/kg.yaml",
    }
    cognition_analyzer = get_cognition_analyzer(cognition_config)

    # 2. 再配置协同引擎
    synergy_config = {
        "engine": "layered_reasoning",
        "emotion_dim": 16,
        "cognition_dim": 16,
        "cognition_source": "external",  # 使用外部认知模块
    }
    synergy_engine = get_synergy_engine(synergy_config)

    # 3. 协同分析时，将认知分析引擎的输出传入
    # cognition_states = cognition_analyzer.analyze(member_messages)
    # result = synergy_engine.fuse(emotion_features, cognition_features)
"""

from typing import Dict, Any, Optional

# ============================================================================
# 默认配置（已更新）
# ============================================================================

DEFAULT_CONFIG = {
    "engine": "weighted_fusion",                 # 默认引擎：加权融合

    # === 核心维度参数 ===
    "emotion_dim": 16,                           # 情绪特征维度（从情绪分析管道获取）
    "cognition_dim": 16,                         # 认知特征维度（从认知分析模块获取）
    "output_dim": 32,                            # 融合后的输出维度

    # === 认知模块对接参数（新增） ===
    "cognition_source": "internal",              # 认知状态来源
                                                 # "internal": 使用协同模块自带的认知分析（旧有模式）
                                                 # "external": 使用新认知模块（tender.cognition）
    "cognition_engine": None,                    # 指定认知分析引擎名称（仅在 cognition_source="external" 时生效）
                                                 # 可选: None(自动检测), "knowledge_state", "behavior_state",
                                                 #       "hybrid_state", "neural_state"
    "enable_cognition_metadata": True,           # 是否将 CognitionState 对象的元数据传播到 SynergyResult

    # === 加权融合引擎参数（W1） ===
    "emotion_weight": 0.5,                       # 情绪权重 α
    "cognition_weight": 0.5,                     # 认知权重 β

    # === 门控融合引擎参数（W2） ===
    "gate_hidden_dim": 16,                       # 门控网络隐藏层维度
    "gate_activation": "sigmoid",                # 门控激活函数

    # === 分层推理引擎参数（W3） ===
    "cognition_first": True,                     # 是否先分析认知再分析情绪
    "cognition_threshold": 0.5,                  # 认知状态判断阈值
    "emotion_threshold": 0.5,                    # 情绪状态判断阈值

    # === 因果协调引擎参数（W4） ===
    "causal_method": "ccm",                      # 因果分析方法
    "causal_lag": 1,                             # 因果滞后步数
    "significance_level": 0.05,                  # 显著性水平
    "max_emotion_features": 5,                   # 情绪特征数量上限
    "max_cognition_features": 5,                 # 认知特征数量上限
}


def _validate_config(config: Dict[str, Any]) -> None:
    """验证配置参数的合法性和一致性（已更新）

    新增了对 cognition_source 等相关参数的验证。

    Args:
        config: 待验证的配置字典

    Raises:
        ValueError: 如果配置参数无效
    """
    # 已有验证逻辑保持不变（检查维度等）
    emotion_dim = config.get("emotion_dim", 16)
    cognition_dim = config.get("cognition_dim", 16)
    output_dim = config.get("output_dim", 32)

    if emotion_dim < 1:
        raise ValueError(f"情绪特征维度必须大于 0: {emotion_dim}")
    if cognition_dim < 1:
        raise ValueError(f"认知特征维度必须大于 0: {cognition_dim}")
    if output_dim < 1:
        raise ValueError(f"输出维度必须大于 0: {output_dim}")

    # 新增：验证认知模块对接参数
    cognition_source = config.get("cognition_source", "internal")
    if cognition_source not in ["internal", "external"]:
        raise ValueError(
            f"不支持的认知状态来源 '{cognition_source}'。"
            f"可选: ['internal', 'external']"
        )

    if cognition_source == "external":
        engine = config.get("cognition_engine", None)
        if engine is not None:
            valid_engines = [
                "knowledge_state", "behavior_state",
                "hybrid_state", "neural_state"
            ]
            if engine not in valid_engines:
                raise ValueError(
                    f"不支持的认知分析引擎 '{engine}'。"
                    f"可选: {valid_engines}"
                )

    # 验证因果协调引擎参数
    if config.get("engine") == "causal_coordination":
        causal_method = config.get("causal_method", "ccm")
        if causal_method not in ["ccm", "granger", "pearson"]:
            raise ValueError(
                f"不支持的因果分析方法 '{causal_method}'。"
                f"可选: ['ccm', 'granger', 'pearson']"
            )


def get_synergy_config(
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """获取协同模块的完整配置

    将用户提供的自定义配置与默认配置合并。

    Args:
        custom_config: 用户自定义配置字典

    Returns:
        Dict[str, Any]: 合并后的完整配置字典
    """
    config = DEFAULT_CONFIG.copy()

    if custom_config is not None:
        for key, value in custom_config.items():
            if key in config and value is not None:
                config[key] = value

    # 验证关键参数的有效性
    _validate_config(config)

    return config


def get_synergy_engine(
    config_or_engine: Dict[str, Any],
    custom_config: Optional[Dict[str, Any]] = None,
) -> "BaseSynergyEngine":
    """根据配置自动创建并初始化协同引擎实例

    这是工厂方法函数，根据配置中的 engine 字段动态创建对应的引擎实例。
    支持两种调用方式：
    1. 直接传入完整配置字典
    2. 分两次传入（先传引擎配置，再传自定义配置）

    Args:
        config_or_engine: 完整配置字典或引擎名称字符串
        custom_config: 可选的自定义配置

    Returns:
        BaseSynergyEngine: 初始化后的协同引擎实例

    Raises:
        ValueError: 如果引擎名称无效或导入失败
    """
    # 解析引擎名称
    if isinstance(config_or_engine, dict):
        engine_name = config_or_engine.get("engine", "weighted_fusion")
        merged_config = get_synergy_config(config_or_engine)
    elif isinstance(config_or_engine, str):
        engine_name = config_or_engine
        merged_config = get_synergy_config(custom_config)
    else:
        raise ValueError(f"不支持的参数类型: {type(config_or_engine)}")

    # 从映射表中获取引擎类名
    if engine_name not in ENGINE_MAP:
        raise ValueError(
            f"不支持的协同引擎 '{engine_name}'。"
            f"可选引擎: {list(ENGINE_MAP.keys())}"
        )

    class_name = ENGINE_MAP[engine_name]

    # 动态导入并实例化
    try:
        import importlib

        module = importlib.import_module("tender.synergy")
        engine_cls = getattr(module, class_name)
        engine = engine_cls(merged_config)

        # 初始化日志：记录认知模块对接状态
        cognition_source = merged_config.get("cognition_source", "internal")
        if cognition_source == "external":
            print(
                f"[协同模块] 引擎 {engine_name} 已配置为使用外部认知模块。"
                f"请确保在调用 fuse() 前已初始化认知分析引擎。"
            )

        return engine

    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"无法导入协同引擎 {class_name}: {e}。"
            f"请确保 tender.synergy 中已注册对应的引擎文件。"
        )
    except Exception as e:
        raise RuntimeError(f"初始化协同引擎 {class_name} 失败: {e}")


# ============================================================================
# 引擎映射表（保持不变）
# ============================================================================

ENGINE_MAP = {
    "weighted_fusion": "WeightedFusionEngine",
    "gated_fusion": "GatedFusionEngine",
    "layered_reasoning": "LayeredReasoningEngine",
    "causal_coordination": "CausalCoordinationEngine",
}
