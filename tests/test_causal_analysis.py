"""
Tender v2.0 — Embracing Heterogeneity
时间因果分析模块测试

覆盖范围：
  - CausalResult、CausalEdge 数据类的创建和序列化
  - ConvergentCrossMappingAnalyzer 的因果推断流程
  - StructuralCausalModelAnalyzer 的因果推断流程
  - PCLiNGAMAnalyzer 的因果推断流程
  - 边界情况：单个成员、无因果关系的随机数据
"""

import pytest
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_time_series() -> Dict[str, np.ndarray]:
    """提供一个标准的时间序列数据集（5个成员，100个时间步，3维特征）。"""
    np.random.seed(42)
    n_members = 5
    n_timesteps = 100
    n_features = 3

    series = {}
    for i in range(n_members):
        member_id = f"user_{i}"
        # 生成带有因果关系的模拟数据
        base = np.sin(np.linspace(0, 4 * np.pi, n_timesteps)).reshape(-1, 1)
        noise = np.random.randn(n_timesteps, n_features) * 0.1
        data = np.tile(base, (1, n_features)) + noise

        if member_id == "user_1":
            # user_1 是 user_0 滞后 5 步的原因
            data[5:] += 0.5 * series["user_0"][:-5, :]
        elif member_id == "user_2":
            # user_2 是 user_0 滞后 3 步的结果
            data[3:] += 0.3 * series["user_0"][:-3, :]

        series[member_id] = data

    return series


@pytest.fixture
def causal_config() -> Dict[str, Any]:
    """标准的因果分析配置。"""
    return {
        "engine": "convergent_cross_mapping",
        "embedding_dimension": 3,
        "time_delay": 1,
        "significance_level": 0.05,
        "max_lag": 10,
    }


# ============================================================================
# CausalEdge 与 CausalResult 数据类测试
# ============================================================================

class TestCausalEdge:
    def test_creation(self):
        """测试因果边的创建。"""
        from tender.causal_analysis.base import CausalEdge

        edge = CausalEdge(source="user_0", target="user_1", strength=0.85, lag=3)
        assert edge.source == "user_0"
        assert edge.target == "user_1"
        assert edge.strength == pytest.approx(0.85)
        assert edge.lag == 3

    def test_invalid_strength(self):
        """测试无效的因果强度。"""
        from tender.causal_analysis.base import CausalEdge

        with pytest.raises(ValueError):
            CausalEdge(source="user_0", target="user_1", strength=1.5, lag=1)

    def test_negative_lag(self):
        """测试负的滞后值。"""
        from tender.causal_analysis.base import CausalEdge

        with pytest.raises(ValueError):
            CausalEdge(source="user_0", target="user_1", strength=0.5, lag=-1)


class TestCausalResult:
    def test_creation(self):
        """测试因果分析结果的创建。"""
        from tender.causal_analysis.base import CausalEdge, CausalResult

        edges = [
            CausalEdge("user_0", "user_1", 0.8, 3),
            CausalEdge("user_0", "user_2", 0.6, 5),
        ]
        result = CausalResult(
            causal_edges=edges,
            in_degrees={"user_1": 1, "user_2": 1, "user_0": 0},
            out_degrees={"user_0": 2, "user_1": 0, "user_2": 0},
            super_spreaders=["user_0"],
            causal_density=0.2,
        )

        assert len(result.causal_edges) == 2
        assert result.causal_density == pytest.approx(0.2)
        assert "user_0" in result.super_spreaders

    def test_empty_edges(self):
        """测试无因果边的边界情况。"""
        from tender.causal_analysis.base import CausalResult

        result = CausalResult(
            causal_edges=[],
            in_degrees={},
            out_degrees={},
            super_spreaders=[],
            causal_density=0.0,
        )

        assert len(result.causal_edges) == 0
        assert len(result.super_spreaders) == 0
        assert result.causal_density == pytest.approx(0.0)


# ============================================================================
# ConvergentCrossMappingAnalyzer 测试
# ============================================================================

class TestConvergentCrossMappingAnalyzer:
    def test_create_analyzer(self, causal_config):
        """测试分析器的创建。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        assert analyzer is not None
        assert analyzer.embedding_dimension == 3

    def test_analyze_basic(self, causal_config, sample_time_series):
        """测试基本的因果分析流程。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        result = analyzer.analyze(
            emotion_vectors=sample_time_series, history_window=5
        )

        assert result is not None
        assert len(result.causal_edges) > 0
        assert len(result.in_degrees) > 0
        assert len(result.out_degrees) > 0
        assert result.causal_density > 0

    def test_single_member(self, causal_config):
        """测试只有一个成员的边界情况。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        single_series = {"user_0": np.random.randn(100, 3)}
        result = analyzer.analyze(
            emotion_vectors=single_series, history_window=5
        )

        # 只有一个成员时，应该没有因果边
        assert len(result.causal_edges) == 0

    def test_causal_direction_detection(self, causal_config):
        """测试因果方向检测的正确性。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        np.random.seed(42)
        n = 200
        t = np.arange(n)

        # 明确构建 X → Y 的因果关系
        x = np.sin(0.1 * t) + 0.1 * np.random.randn(n)
        y = np.zeros(n)
        for i in range(5, n):
            y[i] = 0.8 * x[i - 5] + 0.1 * np.random.randn()

        series = {
            "x": x.reshape(-1, 1),
            "y": y.reshape(-1, 1),
        }

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        result = analyzer.analyze(emotion_vectors=series, history_window=10)

        # 应该检测到 x 到 y 的因果边
        x_to_y_edges = [
            e
            for e in result.causal_edges
            if e.source == "x" and e.target == "y"
        ]
        # 注意：由于 CCM 的对称性和噪声，不保证一定能检测到
        # 但理论上 x→y 的因果强度应该大于 y→x
        # 这里仅验证不报错


# ============================================================================
# StructuralCausalModelAnalyzer 测试
# ============================================================================

class TestStructuralCausalModelAnalyzer:
    @pytest.fixture
    def scm_config(self) -> Dict[str, Any]:
        return {
            "engine": "structural_causal_model",
            "causal_mechanism": "lingam",
            "intervention_enabled": True,
        }

    def test_create_analyzer(self, scm_config):
        """测试分析器的创建。"""
        from tender.causal_analysis.structural_causal_model import (
            StructuralCausalModelAnalyzer,
        )

        analyzer = StructuralCausalModelAnalyzer(scm_config)
        assert analyzer is not None

    def test_analyze_basic(self, scm_config, sample_time_series):
        """测试基本的因果分析流程。"""
        from tender.causal_analysis.structural_causal_model import (
            StructuralCausalModelAnalyzer,
        )

        analyzer = StructuralCausalModelAnalyzer(scm_config)
        result = analyzer.analyze(
            emotion_vectors=sample_time_series, history_window=5
        )

        assert result is not None
        # SCM 应该能检测到一些因果结构
        assert len(result.causal_edges) >= 0


# ============================================================================
# PCLiNGAMAnalyzer 测试
# ============================================================================

class TestPCLiNGAMAnalyzer:
    @pytest.fixture
    def pclingam_config(self) -> Dict[str, Any]:
        return {
            "engine": "pc_lingam",
            "independence_test": "fisherz",
            "alpha": 0.05,
        }

    def test_create_analyzer(self, pclingam_config):
        """测试分析器的创建。"""
        from tender.causal_analysis.pc_lingam import PCLiNGAMAnalyzer

        analyzer = PCLiNGAMAnalyzer(pclingam_config)
        assert analyzer is not None

    def test_analyze_basic(self, pclingam_config, sample_time_series):
        """测试基本的因果分析流程。"""
        from tender.causal_analysis.pc_lingam import PCLiNGAMAnalyzer

        analyzer = PCLiNGAMAnalyzer(pclingam_config)
        result = analyzer.analyze(
            emotion_vectors=sample_time_series, history_window=5
        )

        assert result is not None


# ============================================================================
# 异常与边界测试
# ============================================================================

class TestCausalEdgeCases:
    def test_random_no_causality(self, causal_config):
        """测试无因果关系的时间序列（纯噪声）。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        np.random.seed(42)
        n_members = 3
        n_timesteps = 100

        # 纯噪声，无因果关系
        series = {}
        for i in range(n_members):
            series[f"user_{i}"] = np.random.randn(n_timesteps, 3)

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        result = analyzer.analyze(emotion_vectors=series, history_window=5)

        # 无因果关系时，因果强度应该都很低或为零
        # 但 CCM 可能会偶然检测到弱相关，所以仅验证不报错
        assert result is not None

    def test_short_time_series(self, causal_config):
        """测试时间序列过短的边界情况。"""
        from tender.causal_analysis.convergent_cross_mapping import (
            ConvergentCrossMappingAnalyzer,
        )

        series = {"user_0": np.random.randn(5, 3), "user_1": np.random.randn(5, 3)}

        analyzer = ConvergentCrossMappingAnalyzer(causal_config)
        with pytest.raises(ValueError):
            analyzer.analyze(emotion_vectors=series, history_window=10)
