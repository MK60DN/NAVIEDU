"""情绪向量化模块

该模块负责将群聊成员的文本消息转换为三维连续情绪向量 [valence, arousal, focus]，
为后续的拓扑分析和因果分析提供输入数据。

可替换引擎：
- neuro_symbolic（默认）：神经符号向量化器
  结合符号规则与 LLM 解析，兼顾效率与准确性
- multimodal：多模态向量化器
  融合文本、表情、发言模式等多源信号，适用于数据丰富的场景
"""

from tender.emotion_vectorizer.base import (
    BaseEmotionVectorizer,
    EmotionVector,
)
from tender.emotion_vectorizer.neuro_symbolic_vectorizer import (
    NeuroSymbolicVectorizer,
)
from tender.emotion_vectorizer.multimodal_vectorizer import (
    MultimodalVectorizer,
)
from tender.emotion_vectorizer.config import (
    DEFAULT_CONFIG,
    ENGINE_MAP,
    get_emotion_vectorizer_config,
)

__all__ = [
    "BaseEmotionVectorizer",
    "EmotionVector",
    "NeuroSymbolicVectorizer",
    "MultimodalVectorizer",
    "DEFAULT_CONFIG",
    "ENGINE_MAP",
    "get_emotion_vectorizer_config",
]
