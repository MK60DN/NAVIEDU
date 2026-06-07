""" 认知状态分析模块 - 配置与引擎映射

该模块定义了认知状态分析模块的配置结构和引擎映射关系。
它遵循 Tender 框架的统一配置规范，支持通过 config.yaml 进行灵活的引擎切换。

主要功能：
1. 定义认知分析模块的默认配置
2. 建立引擎名称到具体实现类的映射
3. 提供配置加载与验证工具方法
4. 支持基于配置字典创建指定引擎实例

使用示例：
    from tender.cognition.config import get_cognition_analyzer

    config = {
        "engine": "knowledge_state",
        "feature_dim": 16,
        "use_knowledge_graph": True,
        "knowledge_graph_path": "/path/to/knowledge_graph.yaml",
    }
    analyzer = get_cognition_analyzer(config)   # 返回 KnowledgeStateAnalyzer 实例
"""

from typing import Dict, Any, Optional

from tender.cognition.base import BaseCognitionAnalyzer


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    "engine": "knowledge_state",                # 默认引擎：基于知识图谱的认知状态分析
                                                # 原因：高可解释性，适合教育场景

    # === 通用参数（所有引擎均支持） ===
    "feature_dim": 16,                          # 认知特征维度
                                                # 此值将影响知识图谱嵌入的维度
    "output_dim": 16,                           # 输出维度
                                                # 此值应与 synergy 模块的 cognition_dim 一致
    "use_knowledge_graph": True,                # 是否使用预设知识图谱
                                                # 如果为 False，引擎将尝试从数据中推断知识结构

    # === 知识图谱相关参数（仅在引擎为 knowledge_state 或 hybrid_state 时生效） ===
    "knowledge_graph_path": None,               # 知识图谱配置文件路径
                                                # 支持 .yaml 或 .json 格式
    "kg_embedding_method": "node2vec",           # 知识图谱嵌入方法
                                                # 可选: "node2vec", "graphsage", "onehot"
    "kg_embedding_dim": 16,                      # 知识图谱嵌入维度
    "node_difficulty_key": "difficulty",         # 节点字典中的难度字段名
    "node_prerequisites_key": "prerequisites",   # 节点字典中的前置字段名

    # === 行为分析相关参数（仅在引擎为 behavior_state 或 hybrid_state 时生效） ===
    "window_size_seconds": 300,                  # 行为分析的时间窗口大小（秒）
    "min_messages_for_analysis": 3,              # 分析所需的最少消息数量
    "question_keywords": [                       # 提问关键词列表
        "?", "？", "为什么", "怎么", "如何",
        "什么是", "能不能", "是什么意思",
    ],
    "response_time_threshold": 60,               # 响应时间阈值（秒）

    # === 神经网络相关参数（仅在引擎为 neural_state 时生效） ===
    "neural_model_path": None,                   # 预训练模型路径
    "neural_hidden_dim": 64,                     # 神经网络隐藏层维度
    "neural_dropout": 0.2,                       # Dropout 比率
    "neural_use_pretrained": False,              # 是否使用预训练模型
    "batch_size": 32,                            # 批处理大小
    "max_seq_length": 128,                       # 最大序列长度

    # === 聚合参数（所有引擎通用） ===
    "aggregation_strategy": "weighted",           # 成员到群体的聚合策略
                                                  # 可选: "weighted", "average", "min", "max"
    "enable_attention_bias": True,                # 是否根据参与度加权
    "min_confidence_threshold": 0.1,              # 最低置信度阈值（低于此值的结果将被标记为低置信度）
}

# ============================================================================
# 引擎映射表
# ============================================================================

ENGINE_MAP = {
    "knowledge_state": "KnowledgeStateAnalyzer",    # 基于知识图谱的认知状态分析
    "behavior_state": "BehaviorStateAnalyzer",      # 基于行为模式的认知状态分析
    "hybrid_state": "HybridStateAnalyzer",          # 混合认知状态分析
    "neural_state": "NeuralStateAnalyzer",          # 基于神经网络的隐状态分析
}


def get_cognition_config(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """获取认知分析模块的完整配置

    将用户提供的自定义配置与默认配置合并。
    如果自定义配置中的字段在默认配置中不存在，则忽略（不抛出异常）。
    如果自定义配置中的字段为空，则使用默认值。

    Args:
        custom_config: 用户自定义配置字典
            格式参照 config.yaml 中的 cognition 段

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


def get_cognition_analyzer(
    config_or_engine: Dict[str, Any],
    custom_config: Optional[Dict[str, Any]] = None,
) -> BaseCognitionAnalyzer:
    """根据配置自动创建并初始化认知分析引擎实例

    这是工厂方法函数，根据配置中的 engine 字段动态创建对应的引擎实例。
    支持两种调用方式：
    1. 直接传入完整配置字典
    2. 分两次传入（先传引擎配置，再传自定义配置）

    Args:
        config_or_engine: 完整配置字典或引擎名称字符串
            如果为字典，应包含 engine 字段和其他配置
            如果为字符串，将其视为引擎名称
        custom_config: 可选的自定义配置
            仅在第一个参数为引擎名称时使用

    Returns:
        BaseCognitionAnalyzer: 初始化后的认知分析引擎实例

    Raises:
        ValueError: 如果引擎名称无效或导入失败
    """
    # 解析引擎名称
    if isinstance(config_or_engine, dict):
        engine_name = config_or_engine.get("engine", "knowledge_state")
        merged_config = get_cognition_config(config_or_engine)
    elif isinstance(config_or_engine, str):
        engine_name = config_or_engine
        merged_config = get_cognition_config(custom_config)
    else:
        raise ValueError(f"不支持的参数类型: {type(config_or_engine)}")

    # 从映射表中获取引擎类名
    if engine_name not in ENGINE_MAP:
        raise ValueError(
            f"不支持的认知分析引擎 '{engine_name}'。"
            f"可选引擎: {list(ENGINE_MAP.keys())}"
        )

    class_name = ENGINE_MAP[engine_name]

    # 动态导入并实例化
    try:
        import importlib

        module = importlib.import_module("tender.cognition")
        engine_cls = getattr(module, class_name)
        engine = engine_cls(merged_config)
        return engine

    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"无法导入引擎 {class_name}: {e}。"
            f"请确保 tender.cognition.{engine_name}.py 文件存在且包含 {class_name} 类。"
        )
    except Exception as e:
        raise RuntimeError(f"初始化引擎 {class_name} 失败: {e}")


def _validate_config(config: Dict[str, Any]) -> None:
    """验证配置参数的合法性和一致性

    在运行时对关键配置参数进行检查，避免因无效配置导致的运行时错误。

    Args:
        config: 待验证的配置字典

    Raises:
        ValueError: 如果配置参数无效
    """
    # 检查维度参数
    feature_dim = config.get("feature_dim", 16)
    output_dim = config.get("output_dim", 16)
    kg_embedding_dim = config.get("kg_embedding_dim", 16)

    if feature_dim < 1:
        raise ValueError(f"特征维度必须大于 0: {feature_dim}")
    if output_dim < 1:
        raise ValueError(f"输出维度必须大于 0: {output_dim}")

    # 检查知识图谱嵌入方法
    if config.get("engine") in ["knowledge_state", "hybrid_state"]:
        kg_method = config.get("kg_embedding_method", "node2vec")
        if kg_method not in ["node2vec", "graphsage", "onehot"]:
            raise ValueError(
                f"不支持的知识图谱嵌入方法 '{kg_method}'。"
                f"可选: ['node2vec', 'graphsage', 'onehot']"
            )

        if kg_method == "graphsage" and not config.get("kg_embedding_dim", 0) > 0:
            raise ValueError("使用 GraphSAGE 时需要指定有效的 kg_embedding_dim")

    # 检查窗口参数
    window_size = config.get("window_size_seconds", 300)
    if window_size < 10:
        raise ValueError(f"分析窗口大小不能小于 10 秒: {window_size}")

    min_messages = config.get("min_messages_for_analysis", 3)
    if min_messages < 1:
        raise ValueError(f"最少消息数量不能小于 1: {min_messages}")

    # 检查神经网络参数
    if config.get("engine") == "neural_state":
        neural_hidden = config.get("neural_hidden_dim", 64)
        if neural_hidden < 4:
            raise ValueError(f"神经网络隐藏层维度不能小于 4: {neural_hidden}")

        dropout = config.get("neural_dropout", 0.2)
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"Dropout 比率必须在 [0, 1) 范围内: {dropout}")

        batch_size = config.get("batch_size", 32)
        if batch_size < 1:
            raise ValueError(f"批处理大小不能小于 1: {batch_size}")

    # 检查聚合参数
    aggregation_strategy = config.get("aggregation_strategy", "weighted")
    if aggregation_strategy not in ["weighted", "average", "min", "max"]:
        raise ValueError(
            f"不支持的聚合策略 '{aggregation_strategy}'。"
            f"可选: ['weighted', 'average', 'min', 'max']"
        )
