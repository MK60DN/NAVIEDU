"""
Tender v2.0 — Embracing Heterogeneity
情绪向量化模块测试

覆盖范围：
  - EmotionVector 数据类的创建和序列化
  - NeuroSymbolicVectorizer 的向量化逻辑
  - MultimodalVectorizer 的向量化逻辑
  - 边界情况：空消息列表、无文本消息等
"""

import pytest
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_messages() -> Dict[str, List[Dict[str, Any]]]:
    """提供一个标准消息集用于测试。"""
    return {
        "alice": [
            {"content": "太棒了！我完全理解了！", "timestamp": 100.0, "user_id": "alice"},
            {"content": "这个例子太清楚了，谢谢老师。", "timestamp": 105.0, "user_id": "alice"},
        ],
        "bob": [
            {"content": "有没有人能解释一下这个公式？", "timestamp": 102.0, "user_id": "bob"},
            {"content": "我还是不太懂，感觉很沮丧...", "timestamp": 108.0, "user_id": "bob"},
        ],
        "charlie": [],  # 空消息列表测试
    }


@pytest.fixture
def vectorizer_config() -> Dict[str, Any]:
    """标准的向量化配置。"""
    return {
        "engine": "neuro_symbolic",
        "model_name": "gpt-4o-mini",
        "temperature": 0.1,
        "batch_size": 16,
        "symbol_rule_weights": {
            "default": 0.3,
            "praise": 0.8,
            "complaint": 0.7,
        },
    }


# ============================================================================
# EmotionVector 数据类测试
# ============================================================================

class TestEmotionVector:
    def test_creation(self):
        """测试标准创建"""
        from tender.emotion_vectorizer.emotion_vector import EmotionVector
        
        vector = EmotionVector(
            valence=0.5,
            arousal=0.3,
            focus=0.8,
            confidence=0.9,
            timestamp=100.0,
        )
        
        assert vector.valence == 0.5
        assert vector.arousal == 0.3
        assert vector.focus == 0.8
        assert vector.confidence == 0.9
        assert vector.timestamp == 100.0
        assert vector.source == "unknown"
        assert vector.metadata == {}
    
    def test_to_array(self):
        """测试转换为 numpy 数组"""
        from tender.emotion_vectorizer.emotion_vector import EmotionVector
        
        vector = EmotionVector(valence=0.5, arousal=0.3, focus=0.8, confidence=0.9, timestamp=100.0)
        arr = vector.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert np.allclose(arr, [0.5, 0.3, 0.8])
    
    def test_to_array_with_normalize(self):
        """测试归一化转换"""
        from tender.emotion_vectorizer.emotion_vector import EmotionVector
        
        vector = EmotionVector(valence=0.5, arousal=0.3, focus=0.8, confidence=0.9, timestamp=100.0)
        arr = vector.to_array(normalize=True)
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert np.allclose(np.linalg.norm(arr), 1.0)
    
    def test_invalid_valence(self):
        """测试无效的愉悦度值"""
        from tender.emotion_vectorizer.emotion_vector import EmotionVector
        
        with pytest.raises(ValueError):
            EmotionVector(valence=1.5, arousal=0.5, focus=0.5, confidence=0.9, timestamp=100.0)


# ============================================================================
# NeuroSymbolicVectorizer 测试
# ============================================================================

class TestNeuroSymbolicVectorizer:
    def test_create_vectorizer(self, vectorizer_config):
        """测试向量化器的创建"""
        from tender.emotion_vectorizer.neuro_symbolic import NeuroSymbolicVectorizer
        
        vectorizer = NeuroSymbolicVectorizer(vectorizer_config)
        assert vectorizer is not None
        assert vectorizer.model_name == "gpt-4o-mini"
    
    def test_vectorize_empty_messages(self, vectorizer_config):
        """测试空消息列表"""
        from tender.emotion_vectorizer.neuro_symbolic import NeuroSymbolicVectorizer
        
        vectorizer = NeuroSymbolicVectorizer(vectorizer_config)
        result = vectorizer.vectorize({})
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    @pytest.mark.slow
    def test_vectorize_basic(self, vectorizer_config, sample_messages):
        """测试基本的向量化功能（需要 LLM API 调用）"""
        from tender.emotion_vectorizer.neuro_symbolic import NeuroSymbolicVectorizer
        
        vectorizer = NeuroSymbolicVectorizer(vectorizer_config)
        result = vectorizer.vectorize(sample_messages)
        
        # Alice 有 2 条积极消息，应该有对应的情绪向量
        assert "alice" in result
        assert len(result["alice"]) == 2
        assert result["alice"][0].valence > 0  # 积极消息，valence 应该为正
        
        # Bob 有 2 条消极/困惑消息，应该有对应的情绪向量
        assert "bob" in result
        assert len(result["bob"]) == 2
        
        # Charlie 的消息列表为空，不应出现
        assert "charlie" not in result or len(result["charlie"]) == 0


# ============================================================================
# MultimodalVectorizer 测试
# ============================================================================

class TestMultimodalVectorizer:
    def test_create_vectorizer(self):
        """测试多模态向量化器的创建"""
        from tender.emotion_vectorizer.multimodal import MultimodalVectorizer
        
        config = {
            "engine": "multimodal",
            "multimodal": {
                "text_weight": 0.5,
                "behavior_weight": 0.3,
                "social_weight": 0.2,
            },
        }
        vectorizer = MultimodalVectorizer(config)
        assert vectorizer is not None
    
    def test_vectorize_with_behavior(self):
        """测试包含行为特征的向量化"""
        from tender.emotion_vectorizer.multimodal import MultimodalVectorizer
        
        config = {
            "engine": "multimodal",
            "multimodal": {
                "text_weight": 0.5,
                "behavior_weight": 0.3,
                "social_weight": 0.2,
            },
        }
        vectorizer = MultimodalVectorizer(config)
        messages = {
            "member_1": [
                {
                    "content": "我完全不懂...",
                    "timestamp": 100.0,
                    "user_id": "member_1",
                    "message_length": 12,
                    "response_time": 5.0,
                }
            ]
        }
        result = vectorizer.vectorize(messages)
        assert "member_1" in result


# ============================================================================
# 异常测试
# ============================================================================

class TestEmotionVectorizerExceptions:
    def test_missing_config(self):
        """测试缺少必需配置项"""
        from tender.emotion_vectorizer.neuro_symbolic import NeuroSymbolicVectorizer
        
        with pytest.raises(KeyError):
            NeuroSymbolicVectorizer({})  # 缺少 model_name
    
    def test_invalid_messages_format(self):
        """测试无效的消息格式"""
        from tender.emotion_vectorizer.neuro_symbolic import NeuroSymbolicVectorizer
        
        vectorizer = NeuroSymbolicVectorizer({"model_name": "test"})
        with pytest.raises(TypeError):
            vectorizer.vectorize("invalid input")  # 应该是字典而不是字符串
