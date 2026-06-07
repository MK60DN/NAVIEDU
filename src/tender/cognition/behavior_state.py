""" 基于行为模式的认知状态分析引擎——认知状态分析模块

该模块实现了基于行为模式的认知状态分析策略（Strategy 2）。
与知识图谱策略不同，该策略不依赖预设的知识结构，
而是通过分析成员的交互行为模式来推断其认知状态。

核心思想：
- 成员的认知状态会在其行为中留下可观测的痕迹
- 例如：频繁提问可能意味着困惑，长篇发言可能意味着深入思考
- 通过分析这些行为模式，可以在没有知识图谱的情况下推断认知状态

工作流程：
1. 从成员消息中提取行为特征（发言频率、长度、问题比例等）
2. 结合时间维度的行为变化（响应时间、发言间隔等）
3. 使用启发式规则将行为特征映射到认知状态指标
4. 聚合为群体认知状态

适用场景：
- 开放域讨论，没有预设的知识图谱
- 知识图谱不完整的场景
- 作为知识图谱分析的回退方案或补充

学术基础：
- 行为认知模型 (Baker et al., 2010): 基于行为模式推断认知状态
- 学习分析 (Siemens & Long, 2011): 通过行为数据分析学习过程
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter

from tender.cognition.base import (
    BaseCognitionAnalyzer,
    CognitionState,
    KnowledgeNode,
    BehaviorProfile,
    KnowledgeGraphConfig,
    CognitivePhase,
    EngagementType,
)


class BehaviorStateAnalyzer(BaseCognitionAnalyzer):
    """基于行为模式的认知状态分析引擎

    通过分析成员的交互行为模式来推断认知状态。
    该引擎不需要预设知识图谱，因此适用于更广泛的场景。

    Args:
        config: 配置字典，包含以下字段：
            - feature_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 16）
            - window_size_seconds: 行为分析时间窗口（默认 300）
            - min_messages_for_analysis: 最少消息数量（默认 3）
            - question_keywords: 提问关键词列表
            - response_time_threshold: 响应时间阈值（默认 60）
            - aggregation_strategy: 群体聚合策略（默认 "weighted"）
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化行为模式分析引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.feature_dim = config.get("feature_dim", 16)
        self.output_dim = config.get("output_dim", 16)

        # 行为分析参数
        self.window_size_seconds = config.get("window_size_seconds", 300)
        self.min_messages_for_analysis = config.get("min_messages_for_analysis", 3)

        # 行为特征参数
        self.question_keywords = config.get("question_keywords", [
            "?", "？", "为什么", "怎么", "如何",
            "什么是", "能不能", "是什么意思",
        ])
        self.response_time_threshold = config.get("response_time_threshold", 60)

        # 聚合参数
        self.aggregation_strategy = config.get("aggregation_strategy", "weighted")

        # 行为特征统计的平滑参数
        self._smoothing_factor = 0.1  # 用于防止除零和过度敏感

        # 记录初始化信息
        self._init_info = (
            f"BehaviorStateAnalyzer initialized with "
            f"window={self.window_size_seconds}s, "
            f"min_msgs={self.min_messages_for_analysis}, "
            f"question_kws={len(self.question_keywords)} keywords"
        )

    def analyze(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
    ) -> Dict[str, CognitionState]:
        """分析所有成员的认知状态

        Args:
            member_messages: 成员消息字典
            knowledge_graph_config: 可选的知识图谱配置（此引擎不使用）
            behavior_profiles: 成员行为档案（可选）
                如果提供，将作为主要分析依据
                如果未提供，将从消息中提取

        Returns:
            Dict[str, CognitionState]: 成员 ID 到认知状态的映射
        """
        # 验证输入
        self.validate_inputs(member_messages)

        # 如果未提供行为档案，从消息中提取
        if behavior_profiles is None:
            behavior_profiles = self._extract_behavior_profiles(member_messages)

        # 对每个成员进行分析
        member_states = {}
        for member_id, messages in member_messages.items():
            behavior = behavior_profiles.get(member_id)
            state = self.analyze_single(
                member_id=member_id,
                messages=messages,
                behavior_profile=behavior,
            )
            member_states[member_id] = state

        # 计算群体状态（如果多个成员）
        if len(member_states) > 1:
            group_state = self.compute_group_state(member_states)
            member_states["__group__"] = group_state

        return member_states

    def analyze_single(
        self,
        member_id: str,
        messages: List[Dict[str, Any]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> CognitionState:
        """分析单个成员的认知状态

        Args:
            member_id: 成员唯一标识
            messages: 该成员的消息列表
            knowledge_graph_config: 可选的知识图谱配置（此引擎不使用）
            behavior_profile: 该成员的行为档案（可选）
                如果未提供，将从消息中提取

        Returns:
            CognitionState: 该成员的认知状态
        """
        # 步骤1：提取行为特征（从消息或行为档案）
        if behavior_profile is not None:
            profile = behavior_profile
        else:
            profile = self._extract_single_behavior(member_id, messages)

        # 步骤2：计算核心行为指标
        behavior_metrics = self._compute_behavior_metrics(messages, profile)

        # 步骤3：计算认知负荷
        cognitive_load = self._compute_cognitive_load(
            profile, behavior_metrics, messages
        )

        # 步骤4：计算理解水平
        understanding_level = self._compute_understanding(
            profile, behavior_metrics, messages
        )

        # 步骤5：计算注意力集中程度
        attention_score = self._compute_attention(
            profile, behavior_metrics, messages
        )

        # 步骤6：计算困惑水平
        confusion_level = self._compute_confusion(
            profile, behavior_metrics, messages
        )

        # 步骤7：计算认知灵活性
        cognitive_flexibility = self._compute_flexibility(
            profile, behavior_metrics, messages
        )

        # 步骤8：分类认知阶段
        phase = self._categorize_phase(cognitive_load, understanding_level)

        # 步骤9：计算参与类型
        engagement = self._compute_engagement_type(
            profile.message_count,
            profile.question_count / max(1, profile.message_count),
            profile.avg_message_length,
        )

        # 步骤10：计算置信度
        confidence = self._compute_confidence(profile, behavior_metrics)

        # 步骤11：获取时间戳
        timestamp = max(msg.get("timestamp", 0.0) for msg in messages) if messages else 0.0

        # 构建认知状态
        state = CognitionState(
            member_id=member_id,
            cognitive_load=float(np.clip(cognitive_load, 0.0, 1.0)),
            understanding_level=float(np.clip(understanding_level, 0.0, 1.0)),
            cognitive_phase=phase,
            engagement_type=engagement,
            attention_score=float(np.clip(attention_score, 0.0, 1.0)),
            confusion_level=float(np.clip(confusion_level, 0.0, 1.0)),
            cognitive_flexibility=float(np.clip(cognitive_flexibility, 0.0, 1.0)),
            phase_confidence=float(np.clip(confidence, 0.0, 1.0)),
            source_engine="behavior_state",
            timestamp=timestamp,
            raw_embedding=self._build_embedding(behavior_metrics),
            metadata={
                "message_count": profile.message_count,
                "question_ratio": profile.question_count / max(1, profile.message_count),
                "avg_message_length": profile.avg_message_length,
                "response_time": profile.response_time,
                "behavior_metrics": {k: float(v) for k, v in behavior_metrics.items()},
            },
        )

        return state

    def compute_group_state(
        self,
        member_states: Dict[str, CognitionState],
    ) -> CognitionState:
        """从成员状态计算群体认知状态

        使用配置中设定的聚合策略。

        Args:
            member_states: 成员状态字典

        Returns:
            CognitionState: 群体的认知状态
        """
        return self._aggregate_states(member_states, self.aggregation_strategy)

    def _extract_behavior_profiles(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, BehaviorProfile]:
        """从消息字典中提取所有成员的行为档案

        Args:
            member_messages: 成员消息字典

        Returns:
            Dict[str, BehaviorProfile]: 成员行为档案字典
        """
        profiles = {}
        for member_id, messages in member_messages.items():
            profile = self._extract_single_behavior(member_id, messages)
            profiles[member_id] = profile
        return profiles

    def _extract_single_behavior(
        self,
        member_id: str,
        messages: List[Dict[str, Any]],
    ) -> BehaviorProfile:
        """从单条消息列表中提取行为档案

        Args:
            member_id: 成员唯一标识
            messages: 该成员的消息列表

        Returns:
            BehaviorProfile: 行为档案
        """
        if not messages:
            return BehaviorProfile(member_id=member_id)

        # 基础统计
        message_count = len(messages)
        lengths = [len(msg.get("text", "")) for msg in messages]
        avg_message_length = np.mean(lengths) if lengths else 0.0
        timestamps = [msg.get("timestamp", 0.0) for msg in messages]

        # 提问检测
        question_count = 0
        for msg in messages:
            text = msg.get("text", "")
            for kw in self.question_keywords:
                if kw in text:
                    question_count += 1
                    break

        # 回答检测（回复他人消息）
        answer_count = 0
        interaction_partners = []
        for msg in messages:
            reply_to = msg.get("reply_to", None)
            if reply_to is not None:
                answer_count += 1
                interaction_partners.append(str(reply_to))

        # 响应时间计算（如果有回复关系）
        response_time = 0.0
        if len(timestamps) > 1:
            # 计算相邻消息的时间差的中位数
            intervals = np.diff(sorted(timestamps))
            if len(intervals) > 0:
                response_time = float(np.median(intervals))

        # 表情符号计数
        emoji_count = 0
        import re
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号与象形文字
            "\U0001F680-\U0001F6FF"  # 交通与地图符号
            "\U0001F1E0-\U0001F1FF"  # 国旗
            "]+", flags=re.UNICODE
        )
        for msg in messages:
            emoji_count += len(emoji_pattern.findall(msg.get("text", "")))

        return BehaviorProfile(
            member_id=member_id,
            message_count=message_count,
            question_count=question_count,
            answer_count=answer_count,
            avg_message_length=float(avg_message_length),
            response_time=float(response_time),
            emoji_count=emoji_count,
            interaction_partners=list(set(interaction_partners)),
        )

    def _compute_behavior_metrics(
        self,
        messages: List[Dict[str, Any]],
        profile: BehaviorProfile,
    ) -> Dict[str, float]:
        """计算精细化的行为指标

        从消息和行为档案中提取一系列数值化的行为指标。

        Args:
            messages: 成员的消息列表
            profile: 成员行为档案

        Returns:
            Dict[str, float]: 行为指标字典
        """
        n_msg = len(messages)

        if n_msg == 0:
            return {
                "message_rate": 0.0,
                "question_ratio": 0.0,
                "reply_ratio": 0.0,
                "length_variation": 0.0,
                "time_variation": 0.0,
                "emoji_ratio": 0.0,
                "keyword_diversity": 0.0,
                "response_speed": 0.0,
            }

        # 1. 发言率（相对于时间窗口）
        timestamps = [msg.get("timestamp", 0.0) for msg in messages]
        time_span = max(timestamps) - min(timestamps) if max(timestamps) > min(timestamps) else 1.0
        message_rate = min(1.0, n_msg / (time_span / self.window_size_seconds + self._smoothing_factor))

        # 2. 提问比例
        question_ratio = profile.question_count / max(1, n_msg)

        # 3. 回复比例
        reply_ratio = profile.answer_count / max(1, n_msg)

        # 4. 消息长度变异系数（反映了思考深度的变化）
        lengths = [len(msg.get("text", "")) for msg in messages]
        if lengths:
            length_mean = np.mean(lengths)
            length_variation = np.std(lengths) / max(1.0, length_mean)
        else:
            length_variation = 0.0

        # 5. 时间变异系数（反映了发言节奏的变化）
        if len(timestamps) > 1:
            intervals = np.diff(sorted(timestamps))
            time_variation = np.std(intervals) / max(1.0, np.mean(intervals))
        else:
            time_variation = 0.0

        # 6. 表情符号使用比例
        emoji_ratio = profile.emoji_count / max(1, n_msg)

        # 7. 关键词多样性（简单词频统计）
        all_text = " ".join(msg.get("text", "") for msg in messages)
        words = all_text.split()
        word_count = len(words)
        unique_words = len(set(words))
        keyword_diversity = unique_words / max(1, word_count) if word_count > 0 else 0.0

        # 8. 响应速度（响应时间归一化）
        response_speed = 1.0 - min(1.0, profile.response_time / self.response_time_threshold)

        return {
            "message_rate": float(min(1.0, message_rate)),
            "question_ratio": float(min(1.0, question_ratio)),
            "reply_ratio": float(min(1.0, reply_ratio)),
            "length_variation": float(min(1.0, length_variation)),
            "time_variation": float(min(1.0, time_variation)),
            "emoji_ratio": float(min(1.0, emoji_ratio)),
            "keyword_diversity": float(min(1.0, keyword_diversity)),
            "response_speed": float(np.clip(response_speed, 0.0, 1.0)),
        }

    def _compute_cognitive_load(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
        messages: List[Dict[str, Any]],
    ) -> float:
        """计算认知负荷水平

        基于行为指标推断认知负荷：
        - 高负荷特征：响应速度快、消息短促、提问比例高、长度变异大
        - 低负荷特征：响应速度慢、消息长度适中、提问比例低

        Args:
            profile: 行为档案
            metrics: 行为指标
            messages: 消息列表

        Returns:
            float: 认知负荷水平 (0-1)
        """
        # 因素1：响应速度（权重的 0.3）
        # 响应速度越快，可能意味着负荷越高（因为没时间深入思考）
        load_from_speed = metrics["response_speed"] * 0.3

        # 因素2：提问比例（权重的 0.3）
        # 提问越多，可能意味着负荷越高（处于理解困难）
        load_from_questions = metrics["question_ratio"] * 0.3

        # 因素3：消息长度变异（权重的 0.2）
        # 长度变异大，可能意味着在简单和困难内容之间切换
        load_from_variation = min(1.0, metrics["length_variation"]) * 0.2

        # 因素4：发言密度（权重的 0.2）
        # 发言越密集，可能意味着负荷越高
        load_from_rate = metrics["message_rate"] * 0.2

        total_load = (
            load_from_speed
            + load_from_questions
            + load_from_variation
            + load_from_rate
        )

        return float(np.clip(total_load, 0.0, 1.0))

    def _compute_understanding(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
        messages: List[Dict[str, Any]],
    ) -> float:
        """计算理解水平

        基于行为指标推断理解水平：
        - 高理解：能准确回答问题、消息长度适中、提问比例低
        - 低理解：频繁提问、消息内容碎片化、回复准确率低

        Args:
            profile: 行为档案
            metrics: 行为指标
            messages: 消息列表

        Returns:
            float: 理解水平 (0-1)
        """
        # 因素1：回答比例（权重的 0.4）
        # 能有效回答他人问题，说明理解水平高
        understand_from_reply = metrics["reply_ratio"] * 0.4

        # 因素2：提问比例的反面（权重的 0.3）
        # 提问越少，理解越好
        understand_from_not_questions = (1.0 - metrics["question_ratio"]) * 0.3

        # 因素3：关键词多样性（权重的 0.2）
        # 词汇多样性高，说明知识体系丰富
        understand_from_diversity = metrics["keyword_diversity"] * 0.2

        # 因素4：消息长度适宜性（权重的 0.1）
        avg_length = profile.avg_message_length
        # 长度在 50-200 字符之间可能是理想的
        if 50 <= avg_length <= 200:
            length_score = 0.8
        elif avg_length < 20:
            length_score = 0.3  # 太短，可能是碎片化参与
        elif avg_length > 500:
            length_score = 0.5  # 太长，可能是在发泄而非理解
        else:
            length_score = 0.6

        understand_from_length = length_score * 0.1

        understanding = (
            understand_from_reply
            + understand_from_not_questions
            + understand_from_diversity
            + understand_from_length
        )

        return float(np.clip(understanding, 0.0, 1.0))

    def _compute_attention(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
        messages: List[Dict[str, Any]],
    ) -> float:
        """计算注意力集中程度

        基于行为指标推断注意力：
        - 高注意力：发言节奏稳定、消息长度稳定、回复目标明确
        - 低注意力：发言节奏不稳定、频繁切换话题、回复杂乱

        Args:
            profile: 行为档案
            metrics: 行为指标
            messages: 消息列表

        Returns:
            float: 注意力集中程度 (0-1)
        """
        # 因素1：时间变异性的反面（权重的 0.4）
        # 时间变异性低（发言节奏稳定），注意力集中
        attention_from_stability = (1.0 - min(1.0, metrics["time_variation"])) * 0.4

        # 因素2：长度变异性的反面（权重的 0.3）
        # 长度变异性低，注意力稳定
        attention_from_length = (1.0 - min(1.0, metrics["length_variation"])) * 0.3

        # 因素3：交互专一性（权重的 0.2）
        # 交互对象越少，说明注意力越集中
        n_partners = len(profile.interaction_partners)
        if n_partners <= 1:
            focus_score = 0.8
        elif n_partners <= 3:
            focus_score = 0.6
        elif n_partners <= 5:
            focus_score = 0.4
        else:
            focus_score = 0.2

        attention_from_focus = focus_score * 0.2

        # 因素4：表情符号使用（权重的 0.1）——适度使用表明专注
        emoji_score = 1.0 - min(1.0, metrics["emoji_ratio"] * 5.0)

        attention = (
            attention_from_stability
            + attention_from_length
            + attention_from_focus
            + emoji_score * 0.1
        )

        return float(np.clip(attention, 0.0, 1.0))

    def _compute_confusion(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
        messages: List[Dict[str, Any]],
    ) -> float:
        """计算困惑水平

        基于行为指标推断困惑水平：
        - 高困惑：频繁提问、消息长度极端（极长或极短）、回复少
        - 低困惑：能流畅发言、有效参与讨论

        Args:
            profile: 行为档案
            metrics: 行为指标
            messages: 消息列表

        Returns:
            float: 困惑水平 (0-1)
        """
        # 因素1：提问比例（权重的 0.5）
        # 提问是困惑最直接的指标
        confusion_from_questions = metrics["question_ratio"] * 0.5

        # 因素2：消息长度偏移度（权重的 0.2）
        avg_length = profile.avg_message_length
        if avg_length > 0:
            # 距离理想长度 100 字符的距离
            length_deviation = abs(avg_length - 100) / 200
        else:
            length_deviation = 0.5
        confusion_from_length = min(1.0, length_deviation) * 0.2

        # 因素3：回答比例的反面（权重的 0.2）
        # 不能有效回答问题，说明困惑
        confusion_from_not_reply = (1.0 - metrics["reply_ratio"]) * 0.2

        # 因素4：关键词多样性的反面（权重的 0.1）
        # 词汇多样性低，可能意味着理解受限
        confusion_from_low_diversity = (1.0 - metrics["keyword_diversity"]) * 0.1

        confusion = (
            confusion_from_questions
            + confusion_from_length
            + confusion_from_not_reply
            + confusion_from_low_diversity
        )

        return float(np.clip(confusion, 0.0, 1.0))

    def _compute_flexibility(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
        messages: List[Dict[str, Any]],
    ) -> float:
        """计算认知灵活性

        基于行为指标推断认知灵活性：
        - 高灵活性：能与多人交互、词汇丰富、发言节奏灵活
        - 低灵活性：只与少数人交互、词汇单调、发言模式固定

        Args:
            profile: 行为档案
            metrics: 行为指标
            messages: 消息列表

        Returns:
            float: 认知灵活性 (0-1)
        """
        # 因素1：交互广度（权重的 0.4）
        n_partners = len(profile.interaction_partners)
        interaction_breadth = min(1.0, n_partners / 5.0)  # 5个以上视为广度充分
        flexibility_from_breadth = interaction_breadth * 0.4

        # 因素2：关键词多样性（权重的 0.3）
        flexibility_from_diversity = metrics["keyword_diversity"] * 0.3

        # 因素3：发言节奏灵活性（权重的 0.2）
        # 时间变异性和长度变异性的结合
        rhythm_flexibility = (
            min(1.0, metrics["time_variation"]) * 0.5
            + min(1.0, metrics["length_variation"]) * 0.5
        )
        flexibility_from_rhythm = rhythm_flexibility * 0.2

        # 因素4：参与类型（权重的 0.1）
        if profile.message_count > 10 and profile.answer_count > 3:
            engagement_score = 0.8
        elif profile.message_count > 5:
            engagement_score = 0.5
        else:
            engagement_score = 0.2
        flexibility_from_engagement = engagement_score * 0.1

        flexibility = (
            flexibility_from_breadth
            + flexibility_from_diversity
            + flexibility_from_rhythm
            + flexibility_from_engagement
        )

        return float(np.clip(flexibility, 0.0, 1.0))

    def _compute_confidence(
        self,
        profile: BehaviorProfile,
        metrics: Dict[str, float],
    ) -> float:
        """计算认知状态分析的置信度

        置信度基于行为数据的质量和数量。

        Args:
            profile: 行为档案
            metrics: 行为指标

        Returns:
            float: 置信度 (0-1)
        """
        # 因素1：数据量（权重的 0.5）
        n_msg = profile.message_count
        if n_msg >= self.min_messages_for_analysis:
            data_confidence = min(1.0, n_msg / 20.0)  # 20条以上视为充分
        else:
            data_confidence = 0.0  # 数据不足

        # 因素2：行为指标的丰富度（权重的 0.3）
        active_dims = sum(1 for v in metrics.values() if v > 0.1)
        richness = active_dims / len(metrics)

        # 因素3：行为一致性（权重的 0.2）
        # 如果各种指标都很极端，一致性低，置信度应该降低
        extreme_metrics = sum(1 for v in metrics.values() if v > 0.9 or v < 0.05)
        consistency = 1.0 - min(1.0, extreme_metrics / len(metrics))

        confidence = (
            data_confidence * 0.5
            + richness * 0.3
            + consistency * 0.2
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _build_embedding(
        self,
        behavior_metrics: Dict[str, float],
    ) -> np.ndarray:
        """构建基于行为指标的嵌入向量

        将行为指标编码为固定维度的数值向量。

        Args:
            behavior_metrics: 行为指标字典

        Returns:
            np.ndarray: 行为嵌入向量
        """
        # 将指标映射到向量
        metric_values = list(behavior_metrics.values())
        n_metrics = len(metric_values)

        if n_metrics >= self.output_dim:
            # 直接截断
            return np.array(metric_values[:self.output_dim])
        else:
            # 填充到目标维度
            embedding = np.zeros(self.output_dim)
            embedding[:n_metrics] = metric_values[:n_metrics]
            return embedding
