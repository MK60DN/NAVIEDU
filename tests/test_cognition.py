"""
Tender v2.0 — Embracing Heterogeneity
认知状态分析模块测试

覆盖范围：
  - CognitionState 数据类的创建和序列化
  - HybridStateAnalyzer 的认知状态推断流程
  - KnowledgeStateAnalyzer 的认知状态推断流程
  - BehaviorStateAnalyzer 的认知状态推断流程
  - NeuralStateAnalyzer 的认知状态推断流程
  - 边界情况：消息过少、无文本消息、所有成员认知相同
"""

import pytest
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_messages() -> Dict[str, List[Dict[str, Any]]]:
    """提供一个标准的消息集用于认知分析测试。"""
    return {
        "alice": [
            {"content": "我完全理解了！这个概念太清晰了！", "timestamp": 100.0, "user_id": "alice"},
            {"content": "我来举个例子说明这个定理的应用...", "timestamp": 105.0, "user_id": "alice"},
            {"content": "@bob 你看第三步的推导，这里用到了前面的公式。", "timestamp": 110.0, "user_id": "alice"},
        ],
        "bob": [
            {"content": "我还是不懂... 有没有人能从头讲一遍？", "timestamp": 102.0, "user_id": "bob"},
            {"content": "这些符号代表什么意思？我完全看不懂。", "timestamp": 107.0, "user_id": "bob"},
            {"content": "好焦虑，感觉进度太快了。", "timestamp": 112.0, "user_id": "bob"},
        ],
        "charlie": [
            {"content": "我查了一下，这个理论和之前学的拉普拉斯变换有关系。", "timestamp": 103.0, "user_id": "charlie"},
        ],
    }


@pytest.fixture
def sample_emotion_vectors():
    """模拟情绪向量。"""
    from tender.emotion_vectorizer.emotion_vector import EmotionVector

    return {
        "alice": [
            EmotionVector(valence=0.8, arousal=0.3, focus=0.9, confidence=0.9, timestamp=100.0),
            EmotionVector(valence=0.7, arousal=0.4, focus=0.8, confidence=0.8, timestamp=105.0),
        ],
        "bob": [
            EmotionVector(valence=-0.3, arousal=0.8, focus=0.4, confidence=0.3, timestamp=102.0),
            EmotionVector(valence=-0.5, arousal=0.9, focus=0.3, confidence=0.2, timestamp=107.0),
        ],
        "charlie": [
            EmotionVector(valence=0.2, arousal=0.5, focus=0.6, confidence=0.6, timestamp=103.0),
        ],
    }


@pytest.fixture
def cognition_config() -> Dict[str, Any]:
    """标准的认知分析配置。"""
    return {
        "engine": "hybrid_state",
        "n_components": 3,
        "knowledge_graph_path": "data/knowledge_graph.yaml",
        "min_messages_for_analysis": 1,
    }


# ============================================================================
# CognitionState 数据类测试
# ============================================================================

class TestCognitionState:
    def test_creation(self):
        """测试标准创建。"""
        from tender.cognition.base import CognitionState, CognitivePhase

        state = CognitionState(
            cognitive_load=0.7,
            understanding_level=0.3,
            confusion_level=0.6,
            attention_score=0.5,
            cognitive_flexibility=0.4,
            phase_confidence=0.8,
            cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
        )

        assert state.cognitive_load == pytest.approx(0.7)
        assert state.understanding_level == pytest.approx(0.3)
        assert state.cognitive_phase == CognitivePhase.CORE_UNDERSTANDING

    def test_invalid_load(self):
        """测试无效的认知负荷值。"""
        from tender.cognition.base import CognitionState, CognitivePhase

        with pytest.raises(ValueError):
            CognitionState(
                cognitive_load=-0.1,  # 不能为负
                understanding_level=0.5,
                confusion_level=0.5,
                attention_score=0.5,
                cognitive_flexibility=0.5,
                phase_confidence=0.5,
                cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
            )

    def test_invalid_understanding(self):
        """测试无效的理解水平。"""
        from tender.cognition.base import CognitionState, CognitivePhase

        with pytest.raises(ValueError):
            CognitionState(
                cognitive_load=0.5,
                understanding_level=1.5,  # 不能超过1
                confusion_level=0.5,
                attention_score=0.5,
                cognitive_flexibility=0.5,
                phase_confidence=0.5,
                cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
            )


# ============================================================================
# 认知引擎通用测试
# ============================================================================

class TestCognitionEngines:
    """测试所有认知引擎的公共接口。"""

    @pytest.fixture(params=[
        "hybrid_state",
        "knowledge_state",
        "behavior_state",
        "neural_state",
    ])
    def engine_name(self, request):
        return request.param

    def test_create_analyzer(self, engine_name, cognition_config):
        """测试各种引擎的创建。"""
        from tender.cognition.engine_mapping import get_cognition_analyzer

        config = cognition_config.copy()
        config["engine"] = engine_name
        analyzer = get_cognition_analyzer(config)
        assert analyzer is not None

    def test_analyze_basic(self, engine_name, cognition_config, sample_messages, sample_emotion_vectors):
        """测试各种引擎的基本分析流程。"""
        from tender.cognition.engine_mapping import get_cognition_analyzer

        # 跳过需要外部资源的引擎
        if engine_name in ["knowledge_state", "neural_state"]:
            pytest.skip(f"{engine_name} 需要外部资源，跳过基础测试")

        config = cognition_config.copy()
        config["engine"] = engine_name
        analyzer = get_cognition_analyzer(config)
        result = analyzer.analyze(sample_messages, sample_emotion_vectors)

        assert result is not None
        assert isinstance(result, dict)
        assert "alice" in result
        assert "bob" in result


# ============================================================================
# HybridStateAnalyzer 详细测试
# ============================================================================

class TestHybridStateAnalyzer:
    def test_create_analyzer(self, cognition_config):
        """测试分析器的创建。"""
        from tender.cognition.hybrid_state import HybridStateAnalyzer

        analyzer = HybridStateAnalyzer(cognition_config)
        assert analyzer is not None

    @pytest.mark.slow
    def test_analyze_basic(self, cognition_config, sample_messages, sample_emotion_vectors):
        """测试基本的认知分析流程。"""
        from tender.cognition.hybrid_state import HybridStateAnalyzer

        analyzer = HybridStateAnalyzer(cognition_config)
        result = analyzer.analyze(sample_messages, sample_emotion_vectors)

        assert len(result) == 3  # 3个成员

        # Alice 理解水平应该较高
        alice = result["alice"]
        assert alice.understanding_level > 0.5

        # Bob 应该困惑水平较高
        bob = result["bob"]
        assert bob.confusion_level > 0.3

    def test_empty_messages(self, cognition_config):
        """测试空消息的边界情况。"""
        from tender.cognition.hybrid_state import HybridStateAnalyzer

        analyzer = HybridStateAnalyzer(cognition_config)
        result = analyzer.analyze({}, {})

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_single_message(self, cognition_config):
        """测试只有单条消息的边界情况。"""
        from tender.cognition.hybrid_state import HybridStateAnalyzer

        messages = {
            "user_1": [{"content": "好的", "timestamp": 100.0, "user_id": "user_1"}],
        }
        analyzer = HybridStateAnalyzer(cognition_config)
        result = analyzer.analyze(messages, {})

        # 即使只有单条消息，也应该返回一个合理的结果
        assert "user_1" in result
        user = result["user_1"]
        assert 0 <= user.cognitive_load <= 1
        assert 0 <= user.understanding_level <= 1


# ============================================================================
# BehaviorStateAnalyzer 测试
# ============================================================================

class TestBehaviorStateAnalyzer:
    @pytest.fixture
    def behavior_config(self) -> Dict[str, Any]:
        return {
            "engine": "behavior_state",
            "message_length_weight": 0.3,
            "response_time_weight": 0.3,
            "participation_frequency_weight": 0.4,
        }

    def test_create_analyzer(self, behavior_config):
        """测试分析器的创建。"""
        from tender.cognition.behavior_state import BehaviorStateAnalyzer

        analyzer = BehaviorStateAnalyzer(behavior_config)
        assert analyzer is not None

    def test_analyze_with_behavior_data(self, behavior_config):
        """测试包含行为数据的认知分析。"""
        from tender.cognition.behavior_state import BehaviorStateAnalyzer

        messages = {
            "active_user": [
                {"content": "这是一个很长的消息，用于测试行为分析。认知负荷可能较高。", "timestamp": 100.0, "user_id": "active_user", "message_length": 30, "response_time": 2.0},
                {"content": "继续讨论，保持活跃。", "timestamp": 105.0, "user_id": "active_user", "message_length": 15, "response_time": 3.0},
            ],
            "passive_user": [
                {"content": "好的", "timestamp": 110.0, "user_id": "passive_user", "message_length": 2, "response_time": 30.0},
            ],
        }

        analyzer = BehaviorStateAnalyzer(behavior_config)
        result = analyzer.analyze(messages, {})

        # 活跃用户应该有更高的认知负荷
        assert "active_user" in result
        assert "passive_user" in result

        # 活跃用户的消息更长、回复更快，认知负荷可能更高
        assert result["active_user"].cognitive_load >= result["passive_user"].cognitive_load


# ============================================================================
# 边界情况与异常测试
# ============================================================================

class TestCognitionEdgeCases:
    def test_invalid_engine(self, cognition_config):
        """测试无效的引擎类型。"""
        from tender.cognition.engine_mapping import get_cognition_analyzer

        config = cognition_config.copy()
        config["engine"] = "nonexistent_engine"

        with pytest.raises(ValueError):
            get_cognition_analyzer(config)

    def test_all_members_same_cognition(self, cognition_config):
        """测试所有成员认知相同的极端情况。"""
        from tender.cognition.hybrid_state import HybridStateAnalyzer

        messages = {}
        for i in range(5):
            messages[f"user_{i}"] = [
                {"content": "收到。", "timestamp": 100.0, "user_id": f"user_{i}"},
            ]

        analyzer = HybridStateAnalyzer(cognition_config)
        result = analyzer.analyze(messages, {})

        # 所有成员的认知状态应该相似（但不一定完全相同，因为可能会有随机性）
        understanding_levels = [result[uid].understanding_level for uid in result]
        assert np.std(understanding_levels) < 0.3  # 标准差应该较小
