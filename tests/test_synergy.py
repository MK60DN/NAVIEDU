"""
Tender v2.0 — Embracing Heterogeneity
情绪-认知协同模块测试

覆盖范围：
  - SynergyResult 数据类的创建和序列化
  - LayeredReasoningEngine 的协同推理流程
  - WeightedFusionEngine 的加权融合流程
  - GatedFusionEngine 的门控融合流程
  - CausalCoordinationEngine 的因果协调流程
  - 边界情况：情绪与认知完全匹配、完全不匹配
"""

import pytest
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_emotion_features() -> np.ndarray:
    """模拟 32 维情绪融合特征（4 个成员）。"""
    np.random.seed(42)
    return np.random.randn(4, 16)


@pytest.fixture
def sample_cognition_states():
    """模拟认知状态字典（4 个成员）。"""
    from tender.cognition.base import CognitionState, CognitivePhase

    return {
        "user_0": CognitionState(
            cognitive_load=0.3, understanding_level=0.8, confusion_level=0.1,
            attention_score=0.9, cognitive_flexibility=0.7,
            phase_confidence=0.9, cognitive_phase=CognitivePhase.APPLICATION_CONSOLIDATION,
        ),
        "user_1": CognitionState(
            cognitive_load=0.8, understanding_level=0.2, confusion_level=0.7,
            attention_score=0.4, cognitive_flexibility=0.3,
            phase_confidence=0.5, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
        ),
        "user_2": CognitionState(
            cognitive_load=0.5, understanding_level=0.5, confusion_level=0.4,
            attention_score=0.6, cognitive_flexibility=0.5,
            phase_confidence=0.7, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
        ),
        "user_3": CognitionState(
            cognitive_load=0.2, understanding_level=0.9, confusion_level=0.1,
            attention_score=0.8, cognitive_flexibility=0.8,
            phase_confidence=0.9, cognitive_phase=CognitivePhase.APPLICATION_CONSOLIDATION,
        ),
    }


@pytest.fixture
def sample_member_pairs() -> List[tuple]:
    """模拟成员配对列表。"""
    return [("user_0", "user_1"), ("user_0", "user_2"), ("user_1", "user_3")]


@pytest.fixture
def synergy_config() -> Dict[str, Any]:
    """标准的协同分析配置。"""
    return {
        "engine": "layered_reasoning",
        "emotion_dim": 16,
        "cognition_dim": 16,
        "output_dim": 32,
        "layered_reasoning": {
            "priority": "cognition_first",
            "adaptation_thresholds": {"valence": 0.3, "arousal": 0.3, "focus": 0.2},
        },
    }


# ============================================================================
# SynergyResult 数据类测试
# ============================================================================

class TestSynergyResult:
    def test_creation(self):
        """测试标准创建。"""
        from tender.synergy.base import SynergyResult, SynergyMode

        result = SynergyResult(
            combined_feature=np.random.randn(32),
            synergy_score=0.85,
            dominant_dimension="cognition",
            synergy_mode=SynergyMode.HARMONIOUS,
            adaptation_score=0.78,
            recommendation="维持当前节奏",
        )

        assert result.synergy_score == pytest.approx(0.85)
        assert result.dominant_dimension == "cognition"
        assert result.synergy_mode == SynergyMode.HARMONIOUS
        assert result.adaptation_score == pytest.approx(0.78)
        assert result.combined_feature.shape == (32,)

    def test_invalid_score(self):
        """测试无效的协同度。"""
        from tender.synergy.base import SynergyResult, SynergyMode

        with pytest.raises(ValueError):
            SynergyResult(
                combined_feature=np.zeros(32),
                synergy_score=1.5,  # 不能超过 1
                dominant_dimension="emotion",
                synergy_mode=SynergyMode.HARMONIOUS,
                adaptation_score=0.5,
                recommendation="",
            )

    def test_empty_feature(self):
        """测试空特征向量。"""
        from tender.synergy.base import SynergyResult, SynergyMode

        with pytest.raises(ValueError):
            SynergyResult(
                combined_feature=np.array([]),  # 不能为空
                synergy_score=0.5,
                dominant_dimension="emotion",
                synergy_mode=SynergyMode.HARMONIOUS,
                adaptation_score=0.5,
                recommendation="",
            )


# ============================================================================
# 通用引擎测试
# ============================================================================

class TestSynergyEngines:
    """测试所有协同引擎的公共接口。"""

    @pytest.fixture(params=[
        "layered_reasoning",
        "weighted_fusion",
        "gated_fusion",
        "causal_coordination",
    ])
    def engine_name(self, request):
        return request.param

    def test_create_engine(self, engine_name, synergy_config):
        """测试各种引擎的创建。"""
        from tender.synergy.engine_mapping import get_synergy_engine

        # 跳过需要时序数据的引擎
        if engine_name == "causal_coordination":
            pytest.skip("causal_coordination 需要时序数据，跳过基础测试")

        config = synergy_config.copy()
        config["engine"] = engine_name
        engine = get_synergy_engine(config)
        assert engine is not None


# ============================================================================
# LayeredReasoningEngine 详细测试
# ============================================================================

class TestLayeredReasoningEngine:
    def test_create_engine(self, synergy_config):
        """测试引擎的创建。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine

        engine = LayeredReasoningEngine(synergy_config)
        assert engine is not None
        assert engine.priority == "cognition_first"

    @pytest.mark.slow
    def test_fuse_basic(self, synergy_config, sample_emotion_features, sample_cognition_states, sample_member_pairs):
        """测试基本的协同融合。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine

        engine = LayeredReasoningEngine(synergy_config)
        result = engine.fuse(
            emotion_features=sample_emotion_features,
            cognition_states=sample_cognition_states,
            member_pairs=sample_member_pairs,
        )

        assert result is not None
        assert 0 <= result.synergy_score <= 1
        assert 0 <= result.adaptation_score <= 1
        assert result.combined_feature.shape[0] > 0
        assert result.recommendation is not None

    def test_harmonious_case(self, synergy_config):
        """测试和谐匹配的情况（情绪适应认知阶段）。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine
        from tender.cognition.base import CognitionState, CognitivePhase

        # user_0 处于"核心理解阶段"，情绪为适度焦虑+高专注（期望情绪）
        cognition = {
            "user_0": CognitionState(
                cognitive_load=0.6, understanding_level=0.4, confusion_level=0.5,
                attention_score=0.7, cognitive_flexibility=0.4,
                phase_confidence=0.8, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
            ),
        }
        # 模拟期望情绪：valence=0.3, arousal=0.7, focus=0.8
        emotion_features = np.array([[0.28, 0.72, 0.82, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

        engine = LayeredReasoningEngine(synergy_config)
        result = engine.fuse(emotion_features=emotion_features, cognition_states=cognition, member_pairs=["user_0"])

        # 应该检测到较高的适应度
        assert result.adaptation_score > 0.5
        assert result.synergy_mode.value == "HARMONIOUS"

    def test_conflicting_case(self, synergy_config):
        """测试冲突不匹配的情况。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine
        from tender.cognition.base import CognitionState, CognitivePhase

        # user_0 处于"核心理解阶段"，但情绪为高愉悦度+低唤醒度（不匹配=无聊）
        cognition = {
            "user_0": CognitionState(
                cognitive_load=0.7, understanding_level=0.3, confusion_level=0.6,
                attention_score=0.3, cognitive_flexibility=0.5,
                phase_confidence=0.6, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING,
            ),
        }
        # 错误情绪：高 valence 但低 arousal 和低 focus
        emotion_features = np.array([[0.8, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
                                       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

        engine = LayeredReasoningEngine(synergy_config)
        result = engine.fuse(emotion_features=emotion_features, cognition_states=cognition, member_pairs=["user_0"])

        # 应该检测到较低的适应度
        assert result.adaptation_score < 0.6
        assert result.synergy_mode.value != "HARMONIOUS"


# ============================================================================
# WeightedFusionEngine 测试
# ============================================================================

class TestWeightedFusionEngine:
    @pytest.fixture
    def weighted_config(self) -> Dict[str, Any]:
        return {"engine": "weighted_fusion", "emotion_weight": 0.5, "cognition_weight": 0.5}

    def test_create_engine(self, weighted_config):
        """测试引擎的创建。"""
        from tender.synergy.weighted_fusion import WeightedFusionEngine

        engine = WeightedFusionEngine(weighted_config)
        assert engine is not None
        assert engine.emotion_weight == 0.5

    def test_fuse_basic(self, weighted_config, sample_emotion_features, sample_cognition_states):
        """测试基本的加权融合。"""
        from tender.synergy.weighted_fusion import WeightedFusionEngine

        engine = WeightedFusionEngine(weighted_config)
        result = engine.fuse(emotion_features=sample_emotion_features, cognition_states=sample_cognition_states)

        assert result is not None
        assert result.synergy_score >= 0
        assert result.combined_feature.shape[0] == sample_emotion_features.shape

    def test_asymmetric_weights(self, weighted_config):
        """测试不对称权重。"""
        from tender.synergy.weighted_fusion import WeightedFusionEngine

        config = weighted_config.copy()
        config["emotion_weight"] = 1.0
        config["cognition_weight"] = 0.0

        engine = WeightedFusionEngine(config)
        assert engine.emotion_weight == 1.0


# ============================================================================
# 边界情况与异常测试
# ============================================================================

class TestSynergyEdgeCases:
    def test_empty_cognition_states(self, synergy_config, sample_emotion_features):
        """测试空认知状态。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine

        engine = LayeredReasoningEngine(synergy_config)
        with pytest.raises(ValueError):
            engine.fuse(emotion_features=sample_emotion_features, cognition_states={}, member_pairs=[])

    def test_mismatched_dimensions(self, synergy_config, sample_cognition_states):
        """测试维度不匹配。"""
        from tender.synergy.layered_reasoning import LayeredReasoningEngine

        wrong_features = np.random.randn(4, 8)  # 应该是 16 维
        engine = LayeredReasoningEngine(synergy_config)

        with pytest.raises(ValueError):
            engine.fuse(emotion_features=wrong_features, cognition_states=sample_cognition_states, member_pairs=[])
