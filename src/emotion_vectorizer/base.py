"""
情绪向量化模块 - 抽象基类

该模块定义了所有情绪向量化实现的统一接口。
任何替代方案（LLM、神经符号、多模态）都必须继承此类并实现所有抽象方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


@dataclass
class EmotionVector:
    """
    情绪向量数据结构

    表示一个群成员在特定时间窗口内的情绪状态，
    采用连续的三维维度模型：[愉悦度, 唤醒度, 专注度]。

    该模型继承自 Russell 的 Valence-Arousal 维度理论 (1980)，
    并将 Mehrabian PAD 模型 (1996) 中的 "支配度" 替换为与群聊场景
    更相关的 "专注度"。

    Attributes:
        valence: 愉悦度, 取值范围 (-1, 1), 表示情绪的积极/消极程度
        arousal: 唤醒度, 取值范围 (0, 1), 表示情绪的强烈/平静程度
        focus: 专注度, 取值范围 (0, 1), 表示对当前话题的投入程度
        confidence: 置信度, 取值范围 (0, 1), 表示向量化结果的可靠程度
        timestamp: 时间窗口的结束时间戳（Unix时间戳，秒级）
        metadata: 附加元数据（如原始文本摘要、模型版本等）
    """
    valence: float       # 愉悦度
    arousal: float       # 唤醒度
    focus: float         # 专注度
    confidence: float = 1.0   # 置信度
    timestamp: float = 0.0    # 时间戳
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据

    def to_array(self) -> np.ndarray:
        """返回三维情绪向量 [valence, arousal, focus]"""
        return np.array([self.valence, self.arousal, self.focus])

    def __repr__(self) -> str:
        return (
            f"EmotionVector(valence={self.valence:.3f}, "
            f"arousal={self.arousal:.3f}, "
            f"focus={self.focus:.3f}, "
            f"conf={self.confidence:.2f})"
        )


@dataclass
class VectorizationResult:
    """
    单次向量化批处理的结果

    包含该时间窗口内所有成员的向量化结果集合，
    以及处理统计信息。

    Attributes:
        vectors: 成员ID到情绪向量的映射字典
        window_start: 时间窗口起始时间
        window_end: 时间窗口结束时间
        processing_time_ms: 处理耗时（毫秒）
        member_count: 处理的成员数量
        success_count: 成功向量化的成员数量
        failures: 失败的处理记录列表
    """
    vectors: Dict[str, EmotionVector]         # 成员ID -> 情绪向量
    window_start: float                       # 窗口起始时间
    window_end: float                         # 窗口结束时间
    processing_time_ms: float = 0.0           # 处理耗时
    member_count: int = 0                     # 成员总数
    success_count: int = 0                    # 成功数
    failures: List[Dict[str, Any]] = field(default_factory=list)  # 失败记录


class BaseEmotionVectorizer(ABC):
    """
    情绪向量化抽象基类

    定义情绪向量化模块的统一接口。所有具体实现必须：
    1. 继承此类
    2. 实现所有抽象方法
    3. 严格保持方法签名一致

    设计原则：
    - 插件式架构：通过配置文件选择具体实现
    - 统一接口：所有实现对外暴露相同的方法签名
    - 可替换性：在不影响上下游模块的前提下替换实现
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量化器

        Args:
            config: 配置字典，包含模型路径、API密钥、批大小等参数
        """
        pass

    @abstractmethod
    def vectorize(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        window_start: float,
        window_end: float
    ) -> VectorizationResult:
        """
        对单个时间窗口内的所有成员消息进行情绪向量化

        这是核心方法，将原始文本转换为三维情绪向量。
        每个成员的输出必须严格保证维度为3且顺序为 [valence, arousal, focus]。

        Args:
            member_messages: 成员消息字典
                格式: {成员ID: [{"text": str, "timestamp": float}, ...]}
            window_start: 时间窗口起始时间戳
            window_end: 时间窗口结束时间戳

        Returns:
            VectorizationResult: 向量化结果，包含所有成员的情绪向量
        """
        pass

    @abstractmethod
    def vectorize_single(
        self,
        messages: List[Dict[str, Any]],
        member_id: str,
        window_start: float,
        window_end: float
    ) -> EmotionVector:
        """
        对单个成员在一个时间窗口内的消息进行情绪向量化

        Args:
            messages: 该成员在该窗口内的消息列表
                格式: [{"text": str, "timestamp": float}, ...]
            member_id: 成员唯一标识
            window_start: 时间窗口起始时间戳
            window_end: 时间窗口结束时间戳

        Returns:
            EmotionVector: 该成员在该窗口内的情绪向量
        """
        pass

    @abstractmethod
    def batch_vectorize(
        self,
        window_data: List[Dict[str, Any]],
        progress_bar: bool = False
    ) -> List[VectorizationResult]:
        """
        批量处理多个时间窗口

        高效处理连续时间窗口的数据，内部可做缓存优化。

        Args:
            window_data: 多个时间窗口的数据列表
                格式: [
                    {
                        "member_messages": Dict[str, List],
                        "window_start": float,
                        "window_end": float
                    },
                    ...
                ]
            progress_bar: 是否显示进度条

        Returns:
            List[VectorizationResult]: 每个窗口的向量化结果
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取当前向量化器的信息

        返回模型名称、支持的语言、性能指标等元信息，
        用于日志记录和调试。

        Returns:
            Dict: 信息字典
        """
        pass

    def validate_vector(self, vector: EmotionVector) -> bool:
        """
        验证情绪向量的合法性

        检查维度是否在预期范围内：
        - valence: (-1, 1)
        - arousal: (0, 1)
        - focus: (0, 1)
        - confidence: (0, 1)

        Args:
            vector: 待验证的情绪向量

        Returns:
            bool: 是否合法
        """
        if not (-1.0 <= vector.valence <= 1.0):
            return False
        if not (0.0 <= vector.arousal <= 1.0):
            return False
        if not (0.0 <= vector.focus <= 1.0):
            return False
        if not (0.0 <= vector.confidence <= 1.0):
            return False
        return True
