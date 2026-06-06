"""
Tender v2.0 — Embracing Heterogeneity
时空融合模块测试

覆盖范围：
  - FusionResult 数据类的创建和序列化
  - DCTGNNEngine 的融合流程（确定性一致性变换 + 图神经网络）
  - AttentionFusionEngine 的注意力融合流程
  - 边界情况：缺失时间维度、单一成员、异常输入
"""

import pytest
import numpy as np
from typing import Dict, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_emotion_features() -> np.ndarray:
    """模拟 32 维情绪融合特征（4 个成员）。"""
    np.random.seed(42)
    return np.random.randn(4, 16)


@pytest.fixture
def sample_synergy_feature() -> np.ndarray:
    """模拟 32 维协同特征。"""
    np.random.seed(43)
    return np.random.randn(32)


@pytest.fixture
def sample_temporal_data() -> Dict[str, np.ndarray]:
    """模拟时间序列数据（4 个成员，20 个时间步，3 维）。"""
    np.random.seed(44)
    return {
        "user_0": np.random.randn(20, 3),
        "user_1": np.random.randn(20, 3),
        "user_2": np.random.randn(20, 3),
        "user_3": np.random.randn(20, 3),
    }


@pytest.fixture
def fusion_config() -> Dict[str, Any]:
    """标准的融合配置。"""
    return {
        "engine": "dct_gnn",
        "input_dim": 16,
        "hidden_dim": 64,
        "output_dim": 32,
        "num_heads": 4,
        "dropout": 0.1,
        "dct_config": {
            "coefficient_count": 10,
            "normalize": True,
        },
        "gnn_config": {
            "num_layers": 2,
            "aggregation": "mean",
        },
    }


# ============================================================================
# FusionResult 数据类测试
# ============================================================================

class TestFusionResult:
    def test_creation(self):
        """测试标准创建。"""
        from tender.fusion.base import FusionResult

        result = FusionResult(
            fused_features=np.random.randn(32),
            health_index=0.75,
            forecast=np.array([0.6, 0.7, 0.8]),
            dynamic_graph=np.random.randn(4, 4),
        )

        assert result.fused_features.shape == (32,)
        assert result.health_index == pytest.approx(0.75)
        assert result.forecast.shape == (3,)

    def test_invalid_health_index(self):
        """测试无效的健康指数。"""
        from tender.fusion.base import FusionResult

        with pytest.raises(ValueError):
            FusionResult(
                fused_features=np.zeros(32),
                health_index=1.5,  # 不能超过 1
                forecast=np.zeros(3),
                dynamic_graph=None,
            )

    def test_negative_health_index(self):
        """测试负的健康指数。"""
        from tender.fusion.base import FusionResult

        with pytest.raises(ValueError):
            FusionResult(
                fused_features=np.zeros(32),
                health_index=-0.1,  # 不能为负
                forecast=np.zeros(3),
                dynamic_graph=None,
            )


# ============================================================================
# DCTGNNEngine 测试
# ============================================================================

class TestDCTGNNEngine:
    def test_create_engine(self, fusion_config):
        """测试引擎的创建。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        engine = DCTGNNEngine(fusion_config)
        assert engine is not None
        assert engine.input_dim == 16
        assert engine.hidden_dim == 64

    def test_fuse_basic(self, fusion_config, sample_emotion_features, sample_synergy_feature):
        """测试基本的融合流程。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        engine = DCTGNNEngine(fusion_config)
        result = engine.fuse(
            emotion_features=sample_emotion_features,
            synergy_feature=sample_synergy_feature,
        )

        assert result is not None
        assert isinstance(result, object)
        assert result.fused_features.shape == (32,)
        assert 0 <= result.health_index <= 1
        assert result.forecast.shape == (3,)

    def test_fuse_single_member(self, fusion_config):
        """测试单一成员的融合（边界情况）。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        single_features = np.random.randn(1, 16)
        synergy_feature = np.random.randn(32)

        engine = DCTGNNEngine(fusion_config)
        result = engine.fuse(
            emotion_features=single_features,
            synergy_feature=synergy_feature,
        )

        # 即使只有一个成员，也应该能正常输出
        assert result.fused_features.shape == (32,)
        assert 0 <= result.health_index <= 1

    def test_fuse_many_members(self, fusion_config):
        """测试大量成员的融合。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        many_features = np.random.randn(100, 16)  # 100 个成员
        synergy_feature = np.random.randn(32)

        engine = DCTGNNEngine(fusion_config)
        result = engine.fuse(
            emotion_features=many_features,
            synergy_feature=synergy_feature,
        )

        assert result.fused_features.shape == (32,)
        assert 0 <= result.health_index <= 1

    def test_zero_features(self, fusion_config):
        """测试零特征输入。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        zero_features = np.zeros((4, 16))
        synergy_feature = np.zeros(32)

        engine = DCTGNNEngine(fusion_config)
        result = engine.fuse(
            emotion_features=zero_features,
            synergy_feature=synergy_feature,
        )

        # 零特征应返回默认/保守的输出
        assert result.fused_features.shape == (32,)
        assert result.health_index < 0.5  # 零特征导致低健康度


# ============================================================================
# AttentionFusionEngine 测试
# ============================================================================

class TestAttentionFusionEngine:
    @pytest.fixture
    def attention_config(self) -> Dict[str, Any]:
        return {
            "engine": "attention_fusion",
            "input_dim": 16,
            "hidden_dim": 32,
            "output_dim": 32,
            "num_heads": 4,
        }

    def test_create_engine(self, attention_config):
        """测试引擎的创建。"""
        from tender.fusion.attention_fusion import AttentionFusionEngine

        engine = AttentionFusionEngine(attention_config)
        assert engine is not None
        assert engine.num_heads == 4

    def test_fuse_basic(self, attention_config, sample_emotion_features, sample_synergy_feature):
        """测试基本的注意力融合。"""
        from tender.fusion.attention_fusion import AttentionFusionEngine

        engine = AttentionFusionEngine(attention_config)
        result = engine.fuse(
            emotion_features=sample_emotion_features,
            synergy_feature=sample_synergy_feature,
        )

        assert result is not None
        assert result.fused_features.shape == (32,)
        assert 0 <= result.health_index <= 1

    def test_attention_weights(self, attention_config):
        """测试注意力权重的输出（如果引擎支持）。"""
        from tender.fusion.attention_fusion import AttentionFusionEngine

        features = np.random.randn(5, 16)
        synergy = np.random.randn(32)

        engine = AttentionFusionEngine(attention_config)
        result = engine.fuse(
            emotion_features=features,
            synergy_feature=synergy,
        )

        # 检查是否输出了注意力权重
        # 注意：具体实现可能不同，这里仅测试基本功能
        assert result is not None


# ============================================================================
# 异常与维度检查测试
# ============================================================================

class TestFusionExceptions:
    def test_wrong_input_dim(self, fusion_config):
        """测试输入维度与配置不匹配。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        wrong_features = np.random.randn(4, 8)  # 配置要求 16 维
        synergy_feature = np.random.randn(32)

        engine = DCTGNNEngine(fusion_config)
        with pytest.raises(ValueError):
            engine.fuse(
                emotion_features=wrong_features,
                synergy_feature=synergy_feature,
            )

    def test_wrong_synergy_dim(self, fusion_config):
        """测试协同特征维度不匹配。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        features = np.random.randn(4, 16)
        wrong_synergy = np.random.randn(16)  # 应该是 32 维

        engine = DCTGNNEngine(fusion_config)
        with pytest.raises(ValueError):
            engine.fuse(
                emotion_features=features,
                synergy_feature=wrong_synergy,
            )

    def test_empty_feature_list(self, fusion_config):
        """测试空特征列表。"""
        from tender.fusion.dct_gnn import DCTGNNEngine

        empty_features = np.array([]).reshape(0, 0)
        synergy_feature = np.random.randn(32)

        engine = DCTGNNEngine(fusion_config)
        with pytest.raises(ValueError):
            engine.fuse(
                emotion_features=empty_features,
                synergy_feature=synergy_feature,
            )
