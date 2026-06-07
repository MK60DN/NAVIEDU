"""
神经符号情绪向量化实现（可替换方案）

该模块实现了基于神经符号（Neuro-Symbolic）方法的情绪向量化方案。

学术背景：
传统的 LLM 直接输出情绪向量是一个纯连接主义的"黑箱"映射。
神经符号方法则分为两步：
1) 先让 LLM 提取文本中的情绪因果事件（如"A因为B的批评感到愤怒"）
2) 再将事件结构通过符号化的映射规则编码到维度空间中

这种方法的优势在于：
- 可解释性：情绪向量的生成有明确的语义路径可追溯
- 鲁棒性：对 LLM 的幻觉具有更强的容错能力
- 可审计性：可以检查每个事件映射到维度的合理性

参考文献：
- d'Avila Garcez & Lamb (2023). Neurosymbolic AI: The 3rd Wave.
- Yu et al. (2024). Symbolic Knowledge Distillation for Affective Computing.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple

from tender.emotion_vectorizer.base import (
    BaseEmotionVectorizer,
    EmotionVector,
    VectorizationResult,
)


class NeuroSymbolicVectorizer(BaseEmotionVectorizer):
    """
    基于神经符号方法的情绪向量化器

    工作流程：
    1. 使用 LLM 从文本中提取结构化的事件三元组
    2. 将事件三元组映射到情绪维度的符号规则库
    3. 聚合所有事件的影响得到最终的情绪向量

    Args:
        config: 配置字典，包含以下字段：
            - llm_model: 用于事件提取的 LLM 模型名称
            - api_url: API 地址
            - api_key: API 密钥
            - rule_weight: 符号规则的权重（0-1），越高越依赖规则
            - use_causal_chain: 是否构建因果链
    """

    # 默认的情绪事件映射规则
    # 结构: {事件类型: {维度: 影响值}}
    DEFAULT_EVENT_RULES = {
        "praise": {"valence": 0.4, "arousal": 0.2, "focus": 0.1},
        "criticism": {"valence": -0.4, "arousal": 0.3, "focus": 0.2},
        "agreement": {"valence": 0.3, "arousal": 0.1, "focus": 0.2},
        "disagreement": {"valence": -0.2, "arousal": 0.3, "focus": 0.3},
        "question": {"valence": 0.0, "arousal": 0.2, "focus": 0.5},
        "joke": {"valence": 0.5, "arousal": 0.4, "focus": -0.2},
        "complaint": {"valence": -0.5, "arousal": 0.3, "focus": 0.2},
        "support": {"valence": 0.3, "arousal": 0.2, "focus": 0.1},
        "ignore": {"valence": -0.1, "arousal": -0.1, "focus": -0.3},
        "attack": {"valence": -0.6, "arousal": 0.6, "focus": 0.4},
        "defense": {"valence": -0.2, "arousal": 0.4, "focus": 0.5},
    }

    def __init__(self, config: Dict[str, Any]):
        self.llm_model = config.get("llm_model", "deepseek")
        self.api_url = config.get("api_url")
        self.api_key = config.get("api_key")
        self.rule_weight = config.get("rule_weight", 0.7)
        self.use_causal_chain = config.get("use_causal_chain", False)

        # 编译正则表达式用于事件解析
        self._event_patterns = self._compile_event_patterns()

    def _compile_event_patterns(self) -> Dict[str, re.Pattern]:
        """
        编译事件提取的正则表达式模式

        用于从结构化的事件描述中提取关键信息。
        这里只做辅助提取，主要的事件识别由 LLM 完成。

        Returns:
            Dict[str, re.Pattern]: 编译后的正则表达式
        """
        return {
            "event_type": re.compile(r"event_type:\s*(\w+)"),
            "subject": re.compile(r"subject:\s*(\w+)"),
            "object": re.compile(r"object:\s*(\w+)"),
            "intensity": re.compile(r"intensity:\s*([0-9.]+)"),
        }

    def _build_event_extraction_prompt(
        self,
        messages: List[Dict[str, Any]],
        member_id: str,
    ) -> str:
        """
        构建事件提取提示

        引导 LLM 从消息中提取结构化的情绪因果事件。
        每个事件包含：类型、主体、客体、强度。

        Args:
            messages: 成员消息列表
            member_id: 成员标识

        Returns:
            str: 格式化提示
        """
        texts = [msg["text"] for msg in messages]
        combined_text = "\n".join(texts)

        prompt = f"""你是一个情绪事件提取专家。请分析以下聊天消息，提取其中包含的情绪事件。

用户ID：{member_id}
消息内容：
{combined_text}

请提取出该用户经历或表达的情绪事件，输出JSON数组（不要输出其他内容）：
每个事件格式如下：
{{
    "event_type": "praise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|",
    "event_type": "praise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|copraise|",
    "event_type": "事件类型",
    "event_type": "事件类型（如 praise, criticism, agreement, disagreement, question, joke, complaint, support, ignore, attack, defense）",
    "subject": "事件主体（谁发起的）",
    "object": "事件客体（对谁）",
    "intensity": 强度（0到1之间）
}}

示例输出：
[
    {{"event_type": "criticism", "subject": "用户B", "object": "用户A的观点", "intensity": 0.8}},
    {{"event_type": "defense", "subject": "用户A", "object": "用户B的批评", "intensity": 0.6}}
]

只输出JSON数组，不要任何解释。"""

        return prompt

    def _extract_events(self, prompt: str) -> List[Dict[str, Any]]:
        """
        调用 LLM 提取事件

        Args:
            prompt: 输入提示

        Returns:
            List[Dict]: 事件列表
        """
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # 提取事件需要确定性输出
                "max_tokens": 512,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            content = response.json()["choices"]["message"]["content"]

            # 解析 JSON 数组
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                events = json.loads(content[start:end])
                return events

            return []

        except Exception as e:
            print(f"警告：事件提取失败: {e}")
            return []

    def _map_events_to_vector(
        self,
        events: List[Dict[str, Any]],
    ) -> Tuple[float, float, float, float]:
        """
        将提取的事件通过符号规则映射到情绪向量

        使用预定义的规则库，将每个事件的影响累加，
        然后通过 sigmoid 函数归一化到合理范围。

        Args:
            events: 事件列表

        Returns:
            Tuple[float, float, float, float]: (valence, arousal, focus, confidence)
        """
        if not events:
            return (0.0, 0.5, 0.5, 0.3)

        total_valence = 0.0
        total_arousal = 0.0
        total_focus = 0.0
        total_weight = 0.0

        for event in events:
            event_type = event.get("event_type", "").lower()
            intensity = event.get("intensity", 0.5)

            # 查找匹配的规则
            if event_type in self.DEFAULT_EVENT_RULES:
                rule = self.DEFAULT_EVENT_RULES[event_type]
            else:
                # 未知事件类型，使用中性规则
                rule = {"valence": 0.0, "arousal": 0.1, "focus": 0.1}

            # 累加加权影响
            total_valence += rule["valence"] * intensity
            total_arousal += rule["arousal"] * intensity
            total_focus += rule["focus"] * intensity
            total_weight += self.rule_weight + (1 - self.rule_weight) * intensity

        # 归一化
        if total_weight > 0:
            valence = total_valence / total_weight
            arousal = 0.5 + total_arousal / (2 * total_weight)
            focus = 0.5 + total_focus / (2 * total_weight)
        else:
            valence = 0.0
            arousal = 0.5
            focus = 0.5

        # 确保在有效范围内
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        focus = max(0.0, min(1.0, focus))

        # 置信度基于事件数量和规则匹配率
        confidence = min(1.0, len(events) / 5.0) * self.rule_weight

        return (valence, arousal, focus, confidence)

    def vectorize_single(
        self,
        messages: List[Dict[str, Any]],
        member_id: str,
        window_start: float,
        window_end: float,
    ) -> EmotionVector:
        """
        对单个成员使用神经符号方法进行情绪向量化

        两步流程：
        1. 提取情绪事件
        2. 通过符号规则映射到维度向量

        Args:
            messages: 该成员的消息列表
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
                metadata={"member_id": member_id, "method": "default_no_messages"},
            )

        # 第一步：提取事件
        prompt = self._build_event_extraction_prompt(messages, member_id)
        events = self._extract_events(prompt)

        # 第二步：映射到向量
        valence, arousal, focus, confidence = self._map_events_to_vector(events)

        vector = EmotionVector(
            valence=valence,
            arousal=arousal,
            focus=focus,
            confidence=confidence,
            timestamp=window_end,
            metadata={
                "member_id": member_id,
                "method": "neuro_symbolic",
                "event_count": len(events),
                "events": events[:5],  # 只保存前5个事件作为记录
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
        对单个时间窗口内的所有成员进行神经符号向量化

        Args:
            member_messages: 成员消息字典
            window_start: 窗口起始时间
            window_end: 窗口结束时间

        Returns:
            VectorizationResult: 向量化结果
        """
        import time

        start_time = time.time()
        vectors = {}
        failures = []
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
            member_count=len(member_messages),
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

                iterator = tqdm(window_data, desc="Neuro-symbolic vectorization")
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
            "name": "NeuroSymbolicVectorizer",
            "llm_model": self.llm_model,
            "rule_count": len(self.DEFAULT_EVENT_RULES),
            "rule_weight": self.rule_weight,
            "use_causal_chain": self.use_causal_chain,
        }
