""" 认知状态分析模块 - 抽象基类与数据结构

该模块定义了所有认知状态分析策略实现的统一接口和核心数据结构。
认知状态分析模块的核心功能是：接收成员的交互数据（文本消息、行为日志等），
输出结构化的认知状态描述（理解水平、认知负荷、认知阶段等）。

设计动机：
在"情绪-认知协同"的场景中，认知状态分析是情绪分析的基础。
只有清楚地知道"这个人现在在学什么、学得怎么样"，
才能准确判断"他的情绪状态是否适合当前的学习阶段"。
本模块旨在提供一个统一、可替换的认知状态分析接口。

核心输出指标：
- cognitive_load: 认知负荷水平 (0-1)，0=轻松，1=极重
- understanding_level: 理解水平 (0-1)，0=完全不懂，1=完全掌握
- cognitive_phase: 认知阶段（如"前期探索"、"核心理解"、"应用巩固"等）
- confidence: 分析的置信度 (0-1)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np


class CognitivePhase(Enum):
    """认知阶段枚举

    描述在知识学习过程中，群体所处的典型认知阶段。
    这些阶段对应不同的教学和干预策略。
    """
    EXPLORATION = "前期探索"          # 成员正在接触新知识，理解水平低
    CORE_UNDERSTANDING = "核心理解"    # 成员正在消化核心概念，认知负荷高
    APPLICATION_PRACTICE = "应用巩固"  # 成员将知识付诸实践，建立连接
    FATIGUE_PERIOD = "疲劳期"          # 认知疲劳，效率下降
    MASTERY = "精通期"                # 成员已掌握当前内容，可以进入下一阶段


class EngagementType(Enum):
    """参与类型枚举

    描述成员在当前认知活动中的参与方式。
    """
    PASSIVE = "passive"               # 被动接收信息（如阅读、观看）
    MODERATE = "moderate"             # 中度参与（如点赞、简单回应）
    FOCUSED = "focused"              # 专注参与（如回答问题、完成任务）
    INTENSE = "intense"              # 高强度参与（如主动提问、深入讨论）


@dataclass
class KnowledgeNode:
    """知识节点数据结构

    表示知识图谱中的一个知识点节点。

    Attributes:
        node_id: 知识点唯一标识
        name: 知识点名称（可读）
        difficulty: 难度等级 (0-1)，0=极简单，1=极难
        prerequisites: 前置知识点 ID 列表
        children: 子知识点 ID 列表
        embedding: 可选的知识点嵌入向量（用于计算语义距离）
        metadata: 附加元数据（如所属学科、教学资源等）
    """
    node_id: str
    name: str
    difficulty: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BehaviorProfile:
    """行为档案数据结构

    描述单个成员在当前时间窗口中的交互行为特征。

    Attributes:
        member_id: 成员唯一标识
        message_count: 发言数量
        question_count: 提问数量
        answer_count: 回答/回应数量
        avg_message_length: 平均消息长度（字符数）
        response_time: 平均响应时间（秒）
        emoji_count: 表情符号使用数量
        interaction_partners: 交互对象列表
        metadata: 附加元数据
    """
    member_id: str
    message_count: int = 0
    question_count: int = 0
    answer_count: int = 0
    avg_message_length: float = 0.0
    response_time: float = 0.0
    emoji_count: int = 0
    interaction_partners: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitionState:
    """认知状态数据结构

    描述单个成员或整个群体在某个时间窗口中的认知状态。

    这是认知分析模块的最终输出，将直接传递给协同模块进行融合。

    Attributes:
        member_id: 成员唯一标识（如果为群体分析，设为 "__group__"）
        cognitive_load: 认知负荷水平 (0-1)
        understanding_level: 理解水平 (0-1)
        cognitive_phase: 认知阶段
        engagement_type: 参与类型
        attention_score: 注意力集中程度 (0-1)
        confusion_level: 困惑水平 (0-1)
        cognitive_flexibility: 认知灵活性 (0-1)
        phase_confidence: 阶段判断的置信度 (0-1)
        source_engine: 生成该状态的分析引擎名称
        timestamp: 时间戳
        knowledge_nodes: 涉及的知识点 ID 列表
        raw_embedding: 原始认知状态嵌入向量（供下游使用）
        metadata: 附加元数据
    """
    member_id: str = "__group__"
    cognitive_load: float = 0.5
    understanding_level: float = 0.5
    cognitive_phase: CognitivePhase = CognitivePhase.EXPLORATION
    engagement_type: EngagementType = EngagementType.PASSIVE
    attention_score: float = 0.5
    confusion_level: float = 0.3
    cognitive_flexibility: float = 0.5
    phase_confidence: float = 0.5
    source_engine: str = "unknown"
    timestamp: float = 0.0
    knowledge_nodes: List[str] = field(default_factory=list)
    raw_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitionTrace:
    """认知轨迹数据结构

    记录认知状态随时间的变化轨迹。
    用于时间序列分析和因果分析。

    Attributes:
        member_id: 成员唯一标识
        states: 认知状态时间序列
        timestamps: 对应的时间戳列表
        meta_info: 附加信息
    """
    member_id: str
    states: List[CognitionState] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    meta_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphConfig:
    """知识图谱配置

    描述预设的知识图谱结构和配置参数。
    用于初始化知识图谱分析引擎。

    Attributes:
        nodes: 知识节点列表
        edges: 边列表（前置关系、父子关系等）
        default_difficulty: 未指定难度时的默认值
        embedding_dim: 节点嵌入向量的维度（如果使用嵌入）
        version: 知识图谱版本号
    """
    nodes: List[KnowledgeNode] = field(default_factory=list)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)
    default_difficulty: float = 0.5
    embedding_dim: int = 16
    version: str = "1.0.0"


class BaseCognitionAnalyzer(ABC):
    """认知状态分析抽象基类

    定义所有认知状态分析策略的统一接口。
    所有具体实现必须：
    1. 继承此类
    2. 实现所有抽象方法
    3. 严格保持方法签名一致

    设计原则：
    - 插件式架构：通过配置文件选择具体实现
    - 统一接口：所有实现对外暴露相同的方法签名
    - 可替换性：在不影响上下游模块的前提下替换实现

    核心输入：
    - member_messages: 成员的文本消息（来自数据采集模块）
    - knowledge_graph_config: 知识图谱的配置信息（如果使用知识图谱引擎）
    - behavior_profiles: 成员行为档案（如果使用行为分析引擎）

    核心输出：
    1. CognitionState: 包含认知负荷、理解水平、认知阶段等信息
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """初始化认知状态分析器

        Args:
            config: 配置字典，包含以下字段：
                - feature_dim: 认知特征维度（默认 16）
                - output_dim: 输出维度（默认 16）
                - use_knowledge_graph: 是否使用知识图谱（默认 False）
                - knowledge_graph_config: 知识图谱配置（如果使用）
        """
        pass

    @abstractmethod
    def analyze(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
    ) -> Dict[str, CognitionState]:
        """分析认知状态

        这是核心方法，将成员的交互数据转换为结构化的认知状态。

        Args:
            member_messages: 成员消息字典
                格式: {member_id: [{text: str, timestamp: float, ...}]}
            knowledge_graph_config: 知识图谱配置（可选，部分引擎需要使用）
            behavior_profiles: 成员行为档案字典（可选，部分引擎需要使用）
                格式: {member_id: BehaviorProfile}

        Returns:
            Dict[str, CognitionState]: 成员 ID 到认知状态的映射
                如果进行群体分析，返回 {"__group__": CognitionState}
        """
        pass

    @abstractmethod
    def analyze_single(
        self,
        member_id: str,
        messages: List[Dict[str, Any]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> CognitionState:
        """分析单个成员的认知状态

        用于对特定成员进行精细化的认知状态分析。

        Args:
            member_id: 成员唯一标识
            messages: 该成员的消息列表
            knowledge_graph_config: 知识图谱配置（可选）
            behavior_profile: 该成员的行为档案（可选）

        Returns:
            CognitionState: 该成员的认知状态
        """
        pass

    @abstractmethod
    def compute_group_state(
        self,
        member_states: Dict[str, CognitionState],
    ) -> CognitionState:
        """从成员状态计算群体认知状态

        将多个成员的认知状态聚合为群体级别的认知状态。
        聚合策略可以是平均、加权或基于特定规则。

        Args:
            member_states: 成员状态字典 {member_id: CognitionState}

        Returns:
            CognitionState: 群体的认知状态（member_id 为 "__group__"）
        """
        pass

    def validate_inputs(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
    ) -> bool:
        """验证输入数据的有效性

        Args:
            member_messages: 成员消息字典

        Returns:
            bool: 输入是否有效

        Raises:
            ValueError: 如果输入无效
        """
        if not member_messages:
            raise ValueError("成员消息字典不能为空")

        for member_id, messages in member_messages.items():
            if not isinstance(member_id, str):
                raise TypeError(f"成员 ID 必须为字符串: {member_id}")

            if not messages:
                raise ValueError(f"成员 {member_id} 的消息列表不能为空")

            for msg in messages:
                if "text" not in msg:
                    raise ValueError(f"成员 {member_id} 的消息缺少 'text' 字段")

        return True

    def _categorize_phase(
        self,
        cognitive_load: float,
        understanding_level: float,
    ) -> CognitivePhase:
        """根据认知负荷和理解水平分类认知阶段

        这是一个通用的工具方法，所有引擎都可以复用。

        Args:
            cognitive_load: 认知负荷 (0-1)
            understanding_level: 理解水平 (0-1)

        Returns:
            CognitivePhase: 对应的认知阶段
        """
        if cognitive_load > 0.6:
            if understanding_level < 0.4:
                return CognitivePhase.FATIGUE_PERIOD
            else:
                return CognitivePhase.CORE_UNDERSTANDING
        else:
            if understanding_level < 0.3:
                return CognitivePhase.EXPLORATION
            elif understanding_level > 0.7:
                return CognitivePhase.MASTERY
            else:
                return CognitivePhase.APPLICATION_PRACTICE

    def _compute_engagement_type(
        self,
        message_count: int,
        question_ratio: float,
        avg_length: float,
    ) -> EngagementType:
        """根据行为特征计算参与类型

        Args:
            message_count: 消息数量
            question_ratio: 提问占比
            avg_length: 平均消息长度

        Returns:
            EngagementType: 参与类型
        """
        engagement_score = (
            min(1.0, message_count / 50) * 0.3
            + min(1.0, question_ratio) * 0.3
            + min(1.0, avg_length / 100) * 0.4
        )

        if engagement_score > 0.7:
            return EngagementType.INTENSE
        elif engagement_score > 0.5:
            return EngagementType.FOCUSED
        elif engagement_score > 0.3:
            return EngagementType.MODERATE
        else:
            return EngagementType.PASSIVE

    def _aggregate_states(
        self,
        member_states: Dict[str, CognitionState],
        strategy: str = "weighted",
    ) -> CognitionState:
        """聚合多个成员的认知状态到群体状态

        支持多种聚合策略：
        - "weighted": 按参与度加权平均（默认）
        - "average": 简单平均
        - "min": 取最小值（保守估计）
        - "max": 取最大值（乐观估计）

        Args:
            member_states: 成员状态字典
            strategy: 聚合策略

        Returns:
            CognitionState: 聚合后的群体状态
        """
        if not member_states:
            raise ValueError("成员状态字典不能为空")

        if strategy == "average":
            # 简单平均
            load = np.mean([s.cognitive_load for s in member_states.values()])
            understanding = np.mean([s.understanding_level for s in member_states.values()])
            attention = np.mean([s.attention_score for s in member_states.values()])
            confusion = np.mean([s.confusion_level for s in member_states.values()])
            flexibility = np.mean([s.cognitive_flexibility for s in member_states.values()])
            confidence = np.mean([s.phase_confidence for s in member_states.values()])

        elif strategy == "min":
            # 保守估计：取最小值
            load = np.min([s.cognitive_load for s in member_states.values()])
            understanding = np.min([s.understanding_level for s in member_states.values()])
            attention = np.min([s.attention_score for s in member_states.values()])
            confusion = np.max([s.confusion_level for s in member_states.values()])
            flexibility = np.min([s.cognitive_flexibility for s in member_states.values()])
            confidence = np.min([s.phase_confidence for s in member_states.values()])

        elif strategy == "max":
            # 乐观估计：取最大值
            load = np.max([s.cognitive_load for s in member_states.values()])
            understanding = np.max([s.understanding_level for s in member_states.values()])
            attention = np.max([s.attention_score for s in member_states.values()])
            confusion = np.min([s.confusion_level for s in member_states.values()])
            flexibility = np.max([s.cognitive_flexibility for s in member_states.values()])
            confidence = np.max([s.phase_confidence for s in member_states.values()])

        else:  # "weighted" (默认)
            # 按参与度加权平均
            total_engagement = sum(
                s.engagement_type.value if hasattr(s.engagement_type, 'value')
                else [0.3, 0.5, 0.7, 0.9][s.engagement_type.value if isinstance(s.engagement_type, int) else 1]
                for s in member_states.values()
            )
            if total_engagement == 0:
                total_engagement = len(member_states)

            load = np.average(
                [s.cognitive_load for s in member_states.values()],
                weights=[
                    s.engagement_type.value if hasattr(s.engagement_type, 'value')
                    else 0.5
                    for s in member_states.values()
                ]
            )
            understanding = np.average(
                [s.understanding_level for s in member_states.values()],
                weights=[
                    s.engagement_type.value if hasattr(s.engagement_type, 'value')
                    else 0.5
                    for s in member_states.values()
                ]
            )
            attention = np.average(
                [s.attention_score for s in member_states.values()],
                weights=[
                    s.engagement_type.value if hasattr(s.engagement_type, 'value')
                    else 0.5
                    for s in member_states.values()
                ]
            )
            confusion = np.average(
                [s.confusion_level for s in member_states.values()],
                weights=[
                    s.engagement_type.value if hasattr(s.engagement_type, 'value')
                    else 0.5
                    for s in member_states.values()
                ]
            )
            flexibility = np.average(
                [s.cognitive_flexibility for s in member_states.values()],
                weights=[
                    s.engagement_type.value if hasattr(s.engagement_type, 'value')
                    else 0.5
                    for s in member_states.values()
                ]
            )
            confidence = np.mean([s.phase_confidence for s in member_states.values()])

        # 确定群体认知阶段（使用聚合后的负载和理解水平）
        phase = self._categorize_phase(load, understanding)

        # 收集所有涉及的知识点
        all_nodes = set()
        for s in member_states.values():
            all_nodes.update(s.knowledge_nodes)

        # 构建群体状态
        group_state = CognitionState(
            member_id="__group__",
            cognitive_load=float(np.clip(load, 0.0, 1.0)),
            understanding_level=float(np.clip(understanding, 0.0, 1.0)),
            cognitive_phase=phase,
            engagement_type=EngagementType.FOCUSED,
            attention_score=float(np.clip(attention, 0.0, 1.0)),
            confusion_level=float(np.clip(confusion, 0.0, 1.0)),
            cognitive_flexibility=float(np.clip(flexibility, 0.0, 1.0)),
            phase_confidence=float(np.clip(confidence, 0.0, 1.0)),
            source_engine="aggregation",
            timestamp=max(s.timestamp for s in member_states.values()),
            knowledge_nodes=list(all_nodes),
        )

        return group_state
