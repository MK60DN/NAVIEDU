""" 情绪-认知协同模块 - 基础数据结构与抽象基类（已更新以适配认知模块）

该模块定义了情绪-认知协同的共享基础结构和约束。
修改目标：使 EmotionCognitionPair 能够直接引用新认知模块的 CognitionState 对象。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

# 导入新创建的认知状态分析模块
# 如果找不到，使用兼容模式（回退到旧有的认知状态）
try:
    from tender.cognition.base import CognitionState as NewCognitionState, CognitivePhase
    _COGNITION_MODULE_AVAILABLE = True
except ImportError:
    # 当认知模块尚未同步时，定义一个兼容的桩类型
    from typing import Any
    NewCognitionState = Any
    _COGNITION_MODULE_AVAILABLE = False
    print("警告：无法导入 tender.cognition 模块，使用旧有的认知状态兼容模式。")


class DominantDimension(Enum):
    """主导维度枚举"""
    EMOTION = "emotion"
    COGNITION = "cognition"
    BALANCED = "balanced"


class SynergyMode(Enum):
    """协同模式枚举"""
    HARMONIOUS = "harmonious"                  # 和谐
    EMOTIONAL_OVERWHELM = "emotional_overwhelm" # 情绪主导
    COGNITIVE_OVERLOAD = "cognitive_overload"   # 认知过载
    CONFLICTING = "conflicting"                 # 冲突
    DISENGAGED = "disengaged"                   # 脱离


@dataclass
class EmotionCognitionPair:
    """情绪-认知配对数据结构（已更新）

    这是协同分析的最小数据单元。

    Attributes:
        member_id: 成员唯一标识
        emotion_vector: 该成员的情绪向量 [valence, arousal, focus]
        cognition_state: 该成员的认知状态（现在支持两种格式）
            - 字符串/ID：旧有兼容模式
            - tender.cognition.base.CognitionState 对象：新认知模块
        cognition_confidence: 认知状态分析的置信度 (0-1)
        engagement: 参与度 (0-1)
        timestamp: 时间戳
        metadata: 附加元数据
    """
    member_id: str
    emotion_vector: np.ndarray               # [valence, arousal, focus]
    cognition_state: Any = None               # 兼容新老两种格式
    cognition_confidence: float = 0.5
    engagement: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后验证并规范化数据"""
        # 确保 emotion_vector 是 numpy 数组
        if self.emotion_vector is not None and not isinstance(self.emotion_vector, np.ndarray):
            self.emotion_vector = np.array(self.emotion_vector, dtype=np.float32)

        # 如果 cognition_state 是来自新认知模块的对象，提取关键数值
        if self.cognition_state is not None and hasattr(self.cognition_state, 'cognitive_load'):
            # 这是一个新风格的 CognitionState 对象
            # 保留原始对象，同时更新旧有的字段
            if self.cognition_confidence == 0.5 or self.cognition_confidence is None:
                self.cognition_confidence = getattr(
                    self.cognition_state, 'phase_confidence', 0.5
                )
            if self.engagement == 1.0 or self.engagement is None:
                self.engagement = 0.5  # 默认值，新模块中未定义


@dataclass
class SynergyResult:
    """单次协同操作的结果（已更新）

    包含情绪-认知协同分析后的综合结果。
    新增了对新认知模块元数据的传播支持。
    """
    combined_feature: np.ndarray                       # 融合后的特征
    synergy_score: float = 0.0                         # 协同度评分
    dominant_dimension: DominantDimension = DominantDimension.BALANCED  # 主导维度
    synergy_mode: SynergyMode = SynergyMode.HARMONIOUS  # 协同模式
    emotion_feature: Optional[np.ndarray] = None        # 原始情绪特征
    cognition_feature: Optional[np.ndarray] = None      # 原始认知特征
    recommendation_hint: str = ""                       # 建议提示
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据（传递认知模块信息）


class BaseSynergyEngine(ABC):
    """情绪-认知协同引擎抽象基类

    定义所有协同策略的统一接口。
    所有具体实现必须继承此类并实现抽象方法。
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """初始化协同引擎"""
        pass

    @abstractmethod
    def fuse(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> SynergyResult:
        """融合情绪和认知特征

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            member_pairs: 可选的成员级配对数据
                如果成员级数据中包含新认知模块的 CognitionState 对象，
                可以通过 metadata 传递相关信息

        Returns:
            SynergyResult: 协同分析结果
        """
        pass

    def validate_inputs(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> bool:
        """验证输入数据的有效性（增强版）

        现在支持从 member_pairs 中提取 cognition_state 对象。

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            bool: 输入是否有效

        Raises:
            ValueError: 如果维度不匹配或数据无效
        """
        if emotion_features is None or cognition_features is None:
            raise ValueError("情绪和认知特征不能为空")

        if isinstance(emotion_features, np.ndarray):
            if emotion_features.ndim not in [1, 2]:
                raise ValueError(
                    f"情绪特征维度不合法: {emotion_features.ndim}, 应为1或2"
                )
        else:
            raise TypeError(f"情绪特征类型不合法: {type(emotion_features)}")

        if isinstance(cognition_features, np.ndarray):
            if cognition_features.ndim not in [1, 2]:
                raise ValueError(
                    f"认知特征维度不合法: {cognition_features.ndim}, 应为1或2"
                )
        else:
            raise TypeError(f"认知特征类型不合法: {type(cognition_features)}")

        # 如果都是二维，检查第一个维度（成员数）是否一致
        if emotion_features.ndim == 2 and cognition_features.ndim == 2:
            if emotion_features.shape[0] != cognition_features.shape[0]:
                raise ValueError(
                    f"情绪和认知的成员数不匹配: "
                    f"{emotion_features.shape[0]} vs {cognition_features.shape[0]}"
                )

        return True

    def _extract_cognition_metadata(
        self,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> Dict[str, Any]:
        """从 member_pairs 中提取新认知模块的元数据

        这是新增的辅助方法，用于从 CognitionState 对象中提取信息。

        Args:
            member_pairs: 成员级配对数据

        Returns:
            Dict[str, Any]: 认知元数据字典
        """
        if not member_pairs:
            return {}

        metadata = {}
        total_members = len(member_pairs)

        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'source_engine'):
                # 新认知模块的 CognitionState 对象
                key = f"member_{pair.member_id}"
                metadata[key] = {
                    "cognitive_load": cs.cognitive_load,
                    "understanding_level": cs.understanding_level,
                    "cognitive_phase": cs.cognitive_phase.value,
                    "engagement_type": cs.engagement_type.value,
                    "phase_confidence": cs.phase_confidence,
                    "source_engine": cs.source_engine,
                }

        # 添加群体级别的摘要信息
        if _COGNITION_MODULE_AVAILABLE and metadata:
            avg_load = np.mean([v["cognitive_load"] for v in metadata.values()])
            avg_understanding = np.mean([v["understanding_level"] for v in metadata.values()])
            engines_used = list(set(v["source_engine"] for v in metadata.values()))

            metadata["_group_summary"] = {
                "avg_cognitive_load": float(avg_load),
                "avg_understanding_level": float(avg_understanding),
                "engines_used": engines_used,
                "n_members_with_cognition": len(metadata),
            }

        metadata["total_members"] = total_members
        return metadata

    def _quantize_synergy_score(
        self,
        score: float,
    ) -> Tuple[str, str]:
        """将连续协同度得分量化为可读的标签和分析文本

        Args:
            score: 协同度得分，范围 [-1, 1]

        Returns:
            Tuple[str, str]: (标签, 分析文本)
        """
        if score > 0.7:
            label = "高度协同"
            analysis = "情绪与认知状态高度协调，表现活跃且专注。"
        elif score > 0.3:
            label = "中度协同"
            analysis = "情绪与认知状态基本协调，但存在少量不匹配信号。"
        elif score > -0.3:
            label = "轻度冲突"
            analysis = "情绪与认知状态存在一定的冲突或分离。"
        elif score > -0.7:
            label = "中度冲突"
            analysis = "情绪与认知状态明显不匹配，需要关注。"
        else:
            label = "严重冲突"
            analysis = "情绪与认知状态严重背离，建议立即干预。"

        return label, analysis

    def compute_synergy_score(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> float:
        """计算情绪与认知的协同度得分（基础版本）

        使用余弦相似度作为基础协同度指标。

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            float: 协同度得分，范围 [-1, 1]
        """
        e_flat = emotion_features.flatten()
        c_flat = cognition_features.flatten()

        # 确保长度一致
        min_len = min(len(e_flat), len(c_flat))
        e_flat = e_flat[:min_len]
        c_flat = c_flat[:min_len]

        # 计算余弦相似度
        norm_e = np.linalg.norm(e_flat)
        norm_c = np.linalg.norm(c_flat)

        if norm_e < 1e-8 or norm_c < 1e-8:
            return 0.0

        cosine_sim = np.dot(e_flat, c_flat) / (norm_e * norm_c)

        # 映射到 [-1, 1]
        return float(np.clip(cosine_sim, -1.0, 1.0))

    def classify_synergy_mode(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        synergy_score: float,
    ) -> Tuple[DominantDimension, SynergyMode]:
        """分类协同模式（基础版本）

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            synergy_score: 协同度得分

        Returns:
            Tuple[DominantDimension, SynergyMode]: (主导维度, 协同模式)
        """
        # 计算特征幅度
        e_magnitude = np.linalg.norm(emotion_features.flatten())
        c_magnitude = np.linalg.norm(cognition_features.flatten())

        # 判断主导维度
        total = e_magnitude + c_magnitude + 1e-8
        e_ratio = e_magnitude / total

        if e_ratio > 0.6:
            dominant = DominantDimension.EMOTION
        elif e_ratio < 0.4:
            dominant = DominantDimension.COGNITION
        else:
            dominant = DominantDimension.BALANCED

        # 基于协同度判断协同模式
        if synergy_score > 0.5:
            mode = SynergyMode.HARMONIOUS
        elif synergy_score > 0.0:
            mode = SynergyMode.EMOTIONAL_OVERWHELM
        elif synergy_score > -0.5:
            mode = SynergyMode.CONFLICTING
        else:
            mode = SynergyMode.DISENGAGED

        # 如果认知维度明显占主导且协同度低，可能是认知过载
        if dominant == DominantDimension.COGNITION and synergy_score < 0.0:
            mode = SynergyMode.COGNITIVE_OVERLOAD

        return dominant, mode

    def _generate_hint(
        self,
        dominant: DominantDimension,
        mode: SynergyMode,
        synergy_score: float,
    ) -> str:
        """生成基础的建议提示文本

        Args:
            dominant: 主导维度
            mode: 协同模式
            synergy_score: 协同度得分

        Returns:
            str: 建议提示文本
        """
        if mode == SynergyMode.HARMONIOUS:
            hint = "当前情绪与认知状态和谐，建议维持当前节奏。"
        elif mode == SynergyMode.EMOTIONAL_OVERWHELM:
            hint = "情绪状态占主导，建议优先关注情绪调节。"
        elif mode == SynergyMode.COGNITIVE_OVERLOAD:
            hint = "认知负荷较高，建议适当降低难度或提供休息。"
        elif mode == SynergyMode.CONFLICTING:
            hint = "情绪与认知存在冲突，建议通过互动调节氛围。"
        else:  # DISENGAGED
            hint = "参与度较低，建议增加互动或挑战性任务。"

        return hint
