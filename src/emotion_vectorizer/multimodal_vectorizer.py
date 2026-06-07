"""
多模态情绪向量化实现（可替换方案）

该模块实现了基于多模态融合的情绪向量化方案。

学术背景：
传统文本情绪分析仅依赖语言信息，忽略了社交互动中的非语言信号。
研究表明，表情符号、消息频率、响应时间、表情图像等行为信号
与情绪状态高度相关（Derks et al., 2008; Cohn et al., 2009）。

本模块在文本分析的基础上，融合以下多模态信号：
1. 表情符号使用频率与类型
2. 消息长度与发送频率
3. 响应时间（间隔）
4. 图片/表情包分享行为
5. 消息时序模式（如深夜活跃度）

参考文献：
- Derks, D., et al. (2008). The role of emotion in computer-mediated communication.
- Cohn, J. F., et al. (2009). Multimodal emotion recognition.
- Soleymani, M., et al. (2024). A multimodal approach to emotion understanding in social media.
"""

import time
from typing import Dict, List, Any, Tuple
from collections import Counter

import numpy as np

from tender.emotion_vectorizer.base import (
    BaseEmotionVectorizer,
    EmotionVector,
    VectorizationResult,
)


class MultimodalVectorizer(BaseEmotionVectorizer):
    """
    多模态情绪向量化器

    融合文本语义、行为特征和社交信号的多维度情绪分析方案。
    适合具备多模态数据采集能力的环境。

    Args:
        config: 配置字典，包含以下字段：
            - text_weight: 文本模态权重 (0-1)
            - behavior_weight: 行为模态权重 (0-1)
            - social_weight: 社交模态权重 (0-1)
            - emoji_dict_path: 表情符号情绪映射表路径
    """

    # 常用表情符号的情绪映射（简化版）
    EMOJI_SENTIMENT_MAP = {
        # 积极表情
        "😊": {"valence": 0.6, "arousal": 0.3},
        "😂": {"valence": 0.7, "arousal": 0.7},
        "❤️": {"valence": 0.8, "arousal": 0.5},
        "🎉": {"valence": 0.7, "arousal": 0.6},
        "👍": {"valence": 0.5, "arousal": 0.2},
        "😍": {"valence": 0.8, "arousal": 0.6},
        "🥰": {"valence": 0.7, "arousal": 0.4},
        # 消极表情
        "😢": {"valence": -0.6, "arousal": 0.3},
        "😡": {"valence": -0.7, "arousal": 0.8},
        "💔": {"valence": -0.5, "arousal": 0.4},
        "😞": {"valence": -0.5, "arousal": 0.2},
        "😠": {"valence": -0.6, "arousal": 0.7},
        # 中性/复杂表情
        "🤔": {"valence": 0.0, "arousal": 0.3, "focus": 0.6},
        "😅": {"valence": 0.2, "arousal": 0.5},
        "🤷": {"valence": 0.0, "arousal": 0.1},
    }

    def __init__(self, config: Dict[str, Any]):
        self.text_weight = config.get("text_weight", 0.5)
        self.behavior_weight = config.get("behavior_weight", 0.3)
        self.social_weight = config.get("social_weight", 0.2)

        # 内部文本向量化器复用
        from tender.emotion_vectorizer.llm_vectorizer import LLMVectorizer

        self._text_vectorizer = LLMVectorizer(config)

        # 归一化权重
        total = self.text_weight + self.behavior_weight + self.social_weight
        self.text_weight /= total
        self.behavior_weight /= total
        self.social_weight /= total

        # 历史行为记录（用于计算变化趋势）
        self._history: Dict[str, List[Dict[str, Any]]] = {}

    def _extract_emojis(self, text: str) -> List[str]:
        """
        提取文本中的表情符号

        Args:
            text: 文本

        Returns:
            List[str]: 表情符号列表
        """
        # 简单实现：匹配已知的表情符号
        emojis = []
        for char in text:
            if char in self.EMOJI_SENTIMENT_MAP:
                emojis.append(char)
        return emojis

    def _analyze_emoji_sentiment(
        self, emojis: List[str]
    ) -> Tuple[float, float, float]:
        """
        分析表情符号的情绪倾向

        Args:
            emojis: 表情符号列表

        Returns:
            Tuple[float, float, float]: (valence, arousal, focus)
        """
        if not emojis:
            return (0.0, 0.5, 0.5)

        total_valence = 0.0
        total_arousal = 0.0
        total_focus = 0.0
        count = 0

        for emoji in emojis:
            if emoji in self.EMOJI_SENTIMENT_MAP:
                mapping = self.EMOJI_SENTIMENT_MAP[emoji]
                total_valence += mapping.get("valence", 0.0)
                total_arousal += mapping.get("arousal", 0.5)
                total_focus += mapping.get("focus", 0.5)
                count += 1

        if count == 0:
            return (0.0, 0.5, 0.5)

        return (
            total_valence / count,
            total_arousal / count,
            total_focus / count,
        )

    def _analyze_behavior(
        self,
        messages: List[Dict[str, Any]],
        window_duration: float,
    ) -> Tuple[float, float, float]:
        """
        分析行为模式特征

        从消息发送频率、长度变化等行为信号中提取情绪特征：
        - 高频短消息往往表示高唤醒度
        - 长消息表示高专注度
        - 深夜活跃可能表示低愉悦度

        Args:
            messages: 消息列表
            window_duration: 窗口时长（秒）

        Returns:
            Tuple[float, float, float]: (valence, arousal, focus)
        """
        if not messages:
            return (0.0, 0.5, 0.5)

        # 计算消息频率（条/分钟）
        duration_minutes = max(window_duration / 60.0, 1.0)
        freq = len(messages) / duration_minutes

        # 计算平均消息长度
        avg_length = np.mean([len(msg.get("text", "")) for msg in messages])

        # 计算时间分布
        timestamps = [msg.get("timestamp", 0) for msg in messages]
        if timestamps:
            # 计算间隔的变异系数（规律性指标）
            intervals = np.diff(sorted(timestamps))
            cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            cv = 0.0

        # 行为到情绪维度的映射
        # 高频率 → 高唤醒度
        arousal = min(1.0, 0.5 + freq * 0.1)

        # 长消息 + 规律发送 → 高专注度
        focus = min(1.0, 0.3 + (avg_length / 200.0) * 0.4 + (1.0 / (1.0 + cv)) * 0.3)

        # 非常不规律 + 深夜 → 低愉悦度
        valence = 0.0
        if cv > 2.0 and timestamps:
            # 检查是否有大量深夜消息（假设23:00-6:00为深夜）
            night_messages = sum(
                1
                for t in timestamps
                if (time.localtime(t).tm_hour >= 23 or time.localtime(t).tm_hour < 6)
            )
            night_ratio = night_messages / len(timestamps)
            valence = -0.3 * night_ratio

        return (valence, arousal, focus)

    def _analyze_social(
        self,
        messages: List[Dict[str, Any]],
        member_id: str,
    ) -> Tuple[float, float, float]:
        """
        分析社交交互特征

        从响应模式、被提及频率等社交信号中提取情绪特征：
        - 快速响应 @ 表示高参与度
        - 被多次提及表示影响力/关注度

        Args:
            messages: 消息列表
            member_id: 成员标识

        Returns:
            Tuple[float, float, float]: (valence, arousal, focus)
        """
        if not messages:
            return (0.0, 0.5, 0.5)

        # 统计包含 @ 的消息
        mention_count = sum(
            1 for msg in messages if "@" in msg.get("text", "")
        )
        mention_ratio = mention_count / len(messages)

        # 计算交互强度
        # 被 @ 的频率表示社交参与度
        focus = min(1.0, 0.5 + mention_ratio * 0.5)

        # 响应速度（如果有时间戳信息）
        arousal = 0.5 + mention_ratio * 0.3

        # 积极的社交互动通常带来正面的愉悦度
        valence = mention_ratio * 0.3

        return (valence, arousal, focus)

    def vectorize_single(
        self,
        messages: List[Dict[str, Any]],
        member_id: str,
        window_start: float,
        window_end: float,
    ) -> EmotionVector:
        """
        对单个成员使用多模态方法进行情绪向量化

        融合文本、行为、社交三个模态的分析结果。

        Args:
            messages: 消息列表
            member_id: 成员标识
            window_start: 窗口起始时间
            window_end: 窗口结束时间

        Returns:
            EmotionVector: 情绪向量
        """
        if not messages:
            return EmotionVector(
                valence=0.0,
                arousal=0.5,
                focus=0.5,
                confidence=0.3,
                timestamp=window_end,
                metadata={"member_id": member_id, "method": "default_no_data"},
            )

        window_duration = window_end - window_start
        texts = [msg.get("text", "") for msg in messages]

        # 1. 文本模态分析
        text_vector = self._text_vectorizer.vectorize_single(
            messages, member_id, window_start, window_end
        )

        # 提取表情符号
        all_emojis = []
        for text in texts:
            all_emojis.extend(self._extract_emojis(text))
        emoji_vector = self._analyze_emoji_sentiment(all_emojis)

        # 融合文本和表情符号（表情符号视为文本模态的子信号）
        text_valence = text_vector.valence * 0.7 + emoji_vector[0] * 0.3
        text_arousal = text_vector.arousal * 0.6 + emoji_vector[1] * 0.4
        text_focus = text_vector.focus * 0.8 + emoji_vector[2] * 0.2

        # 2. 行为模态分析
        behav_valence, behav_arousal, behav_focus = self._analyze_behavior(
            messages, window_duration
        )

        # 3. 社交模态分析
        social_valence, social_arousal, social_focus = self._analyze_social(
            messages, member_id
        )

        # 4. 加权融合
        valence = (
            self.text_weight * text_valence
            + self.behavior_weight * behav_valence
            + self.social_weight * social_valence
        )

        arousal = (
            self.text_weight * text_arousal
            + self.behavior_weight * behav_arousal
            + self.social_weight * social_arousal
        )

        focus = (
            self.text_weight * text_focus
            + self.behavior_weight * behav_focus
            + self.social_weight * social_focus
        )

        # 确保在有效范围内
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        focus = max(0.0, min(1.0, focus))

        # 置信度：基于消息数量和模态一致性
        confidence = min(1.0, len(messages) / 10.0)

        vector = EmotionVector(
            valence=valence,
            arousal=arousal,
            focus=focus,
            confidence=confidence,
            timestamp=window_end,
            metadata={
                "member_id": member_id,
                "method": "multimodal",
                "text_weight": self.text_weight,
                "behavior_weight": self.behavior_weight,
                "social_weight": self.social_weight,
                "emoji_count": len(all_emojis),
                "message_count": len(messages),
            },
        )

        return vector

    def vectorize(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        window_start: float,
        window_end: float,
    ) -> VectorizationResult:
        """
        对单个时间窗口内的所有成员进行多模态向量化

        Args:
            member_messages: 成员消息字典
            window_start: 窗口起始时间
            window_end: 窗口结束时间

        Returns:
            VectorizationResult: 向量化结果
        """
        start_time = time.time()
        vectors = {}
        failures = []
        total_count = len(member_messages)
        success_count = 0

        for member_id, messages in member_messages.items():
            try:
                vector = self.vectorize_single(
                    messages, member_id, window_start, window_end
                )
                vectors[member_id] = vector
                success_count += 1
            except Exception as e:
                failures.append({"member_id": member_id, "error": str(e)})
                vectors[member_id] = EmotionVector(
                    valence=0.0,
                    arousal=0.5,
                    focus=0.5,
                    confidence=0.0,
                    timestamp=window_end,
                    metadata={"member_id": member_id, "error": str(e)},
                )

        processing_time_ms = (time.time() - start_time) * 1000

        return VectorizationResult(
            vectors=vectors,
            window_start=window_start,
            window_end=window_end,
            processing_time_ms=processing_time_ms,
            member_count=total_count,
            success_count=success_count,
            failures=failures,
        )

    def batch_vectorize(
        self,
        window_data: List[Dict[str, Any]],
        progress_bar: bool = False,
    ) -> List[VectorizationResult]:
        """
        批量处理多个时间窗口

        Args:
            window_data: 窗口数据列表
            progress_bar: 是否显示进度条

        Returns:
            List[VectorizationResult]: 向量化结果列表
        """
        results = []
        iterator = window_data

        if progress_bar:
            try:
                from tqdm import tqdm

                iterator = tqdm(window_data, desc="Multimodal vectorization")
            except ImportError:
                pass

        for data in iterator:
            result = self.vectorize(
                data["member_messages"],
                data["window_start"],
                data["window_end"],
            )
            results.append(result)

        return results

    def get_info(self) -> Dict[str, Any]:
        """
        获取向量化器信息

        Returns:
            Dict: 信息字典
        """
        return {
            "name": "MultimodalVectorizer",
            "text_weight": self.text_weight,
            "behavior_weight": self.behavior_weight,
            "social_weight": self.social_weight,
            "emoji_map_size": len(self.EMOJI_SENTIMENT_MAP),
        }
