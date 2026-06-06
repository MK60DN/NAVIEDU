"""
Tender v2.0 — Embracing Heterogeneity
个人-群体不匹配检测模块测试

覆盖范围：
  - MismatchMetrics 数据类的创建和序列化
  - TopologicalMismatchDetector 的拓扑不匹配距离计算
  - DynamicMismatchDetector 的动态不匹配距离计算
  - PersonalIndependenceModel 的自洽性计算
  - 边界情况：个人与群体完全一致、完全不一致、单一成员
"""

import pytest
import numpy as np
from typing import Dict, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_personal_point_cloud() -> np.ndarray:
    """模拟个人情绪点云（20个点，3维）。"""
    np.random.seed(42)
    return np.random.randn(20, 3)


@pytest.fixture
def sample_group_point_cloud() -> np.ndarray:
    """模拟群体情绪点云（100个点，3维）。"""
    np.random.seed(43)
    return np.random.randn(100, 3)


@pytest.fixture
def sample_personal_time_series() -> np.ndarray:
    """模拟个人时间序列（50个时间步，3维）。"""
    np.random.seed(42)
    n = 50
    t = np.linspace(0, 10 * np.pi, n)
    # 个人轨迹：正弦波 + 噪声
    return np.column_stack([
        np.sin(t),
        0.5 * np.cos(t / 2),
        0.3 * np.sin(t / 3),
    ]) + 0.1 * np.random.randn(n, 3)


@pytest.fixture
def sample_group_time_series() -> np.ndarray:
    """模拟群体时间序列（50个时间步，3维）。"""
    np.random.seed(43)
    n = 50
    t = np.linspace(0, 10 * np.pi, n)
    # 群体轨迹：余弦波 + 噪声
    return np.column_stack([
        np.cos(t),
        0.6 * np.sin(t / 2 + 0.5),
        0.2 * np.cos(t / 3),
    ]) + 0.1 * np.random.randn(n, 3)


@pytest.fixture
def mismatch_config() -> Dict[str, Any]:
    """标准的不匹配检测配置。"""
    return {
        "enabled": True,
        "topological_mismatch": {
            "method": "wasserstein",
            "wasserstein_regularization": 0.01,
        },
        "dynamic_mismatch": {
            "method": "dtw",
            "dtw_window": None,
        },
        "personal_independence": {
            "method": "autocorrelation",
            "window_size": 10,
        },
    }


# ============================================================================
# MismatchMetrics 数据类测试
# ============================================================================

class TestMismatchMetrics:
    def test_creation(self):
        """测试标准创建。"""
        from tender.mismatch.base import MismatchMetrics

        metrics = MismatchMetrics(
            structural_distance=0.7,
            dynamic_distance=0.8,
            personal_self_consistency=0.9,
        )

        assert metrics.structural_distance == pytest.approx(0.7)
        assert metrics.dynamic_distance == pytest.approx(0.8)
        assert metrics.personal_self_consistency == pytest.approx(0.9)

    def test_invalid_distance(self):
        """测试无效的距离值。"""
        from tender.mismatch.base import MismatchMetrics

        with pytest.raises(ValueError):
            MismatchMetrics(
                structural_distance=-0.1,  # 不能为负
                dynamic_distance=0.5,
                personal_self_consistency=0.5,
            )

    def test_invalid_consistency(self):
        """测试无效的自洽性。"""
        from tender.mismatch.base import MismatchMetrics

        with pytest.raises(ValueError):
            MismatchMetrics(
                structural_distance=0.5,
                dynamic_distance=0.5,
                personal_self_consistency=1.5,  # 不能超过 1
            )

    def test_healthy_independent_pattern(self):
        """测试健康独立的模式（高不匹配 + 高自洽性）。"""
        from tender.mismatch.base import MismatchMetrics

        metrics = MismatchMetrics(
            structural_distance=0.8,
            dynamic_distance=0.7,
            personal_self_consistency=0.9,
        )

        # 自洽性高，不匹配度高 → 健康独立
        needs_intervention = (metrics.structural_distance > 0.6 or metrics.dynamic_distance > 0.6) and metrics.personal_self_consistency < 0.5
        assert not needs_intervention

    def test_needs_intervention_pattern(self):
        """测试需要干预的模式（高不匹配 + 低自洽性）。"""
        from tender.mismatch.base import MismatchMetrics

        metrics = MismatchMetrics(
            structural_distance=0.8,
            dynamic_distance=0.7,
            personal_self_consistency=0.3,
        )

        # 自洽性低，不匹配度高 → 需要干预
        needs_intervention = (metrics.structural_distance > 0.6 or metrics.dynamic_distance > 0.6) and metrics.personal_self_consistency < 0.5
        assert needs_intervention


# ============================================================================
# TopologicalMismatchDetector 测试
# ============================================================================

class TestTopologicalMismatchDetector:
    def test_create_detector(self, mismatch_config):
        """测试检测器的创建。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        detector = TopologicalMismatchDetector(mismatch_config)
        assert detector is not None
        assert detector.method == "wasserstein"

    def test_compute_distance_basic(self, mismatch_config, sample_personal_point_cloud, sample_group_point_cloud):
        """测试基本的拓扑不匹配距离计算。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        detector = TopologicalMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_point_cloud=sample_personal_point_cloud,
            group_point_cloud=sample_group_point_cloud,
        )

        assert isinstance(distance, float)
        assert distance >= 0
        # 不同的随机数据，距离应该大于 0
        assert distance > 0

    def test_perfect_match(self, mismatch_config):
        """测试个人与群体完全一致的情况。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        cloud = np.random.randn(10, 3)
        detector = TopologicalMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_point_cloud=cloud,
            group_point_cloud=cloud,  # 完全一致
        )

        # 完全一致，距离应该非常接近 0
        assert distance < 1e-6

    def test_completely_different(self, mismatch_config):
        """测试个人与群体完全不同的情况。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        # 个人点云集中在 (0,0,0) 附近
        personal = np.random.randn(10, 3) * 0.1
        # 群体点云集中在 (100,100,100) 附近
        group = np.random.randn(100, 3) * 0.1 + 100

        detector = TopologicalMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_point_cloud=personal,
            group_point_cloud=group,
        )

        # 完全不同，距离应该很大
        assert distance > 0.5

    def test_single_point_cloud(self, mismatch_config):
        """测试单点云（只有一个点）的边界情况。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        personal = np.array([[0.5, 0.5, 0.5]])
        group = np.random.randn(100, 3)

        detector = TopologicalMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_point_cloud=personal,
            group_point_cloud=group,
        )

        assert distance >= 0
        assert isinstance(distance, float)


# ============================================================================
# DynamicMismatchDetector 测试
# ============================================================================

class TestDynamicMismatchDetector:
    def test_create_detector(self, mismatch_config):
        """测试检测器的创建。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        detector = DynamicMismatchDetector(mismatch_config)
        assert detector is not None
        assert detector.method == "dtw"

    def test_compute_distance_basic(self, mismatch_config, sample_personal_time_series, sample_group_time_series):
        """测试基本的动态不匹配距离计算。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        detector = DynamicMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_ts=sample_personal_time_series,
            group_ts=sample_group_time_series,
        )

        assert isinstance(distance, float)
        assert distance >= 0
        # 不同的时间序列，距离应该大于 0
        assert distance > 0

    def test_same_trajectory(self, mismatch_config):
        """测试相同轨迹的情况。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        n = 50
        t = np.linspace(0, 10 * np.pi, n)
        trajectory = np.column_stack([np.sin(t), np.cos(t), np.sin(t / 2)])

        detector = DynamicMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_ts=trajectory,
            group_ts=trajectory,  # 完全相同的轨迹
        )

        # 相同轨迹，距离应为 0
        assert distance < 1e-6

    def test_offset_trajectory(self, mismatch_config):
        """测试偏移但形状相同的轨迹。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        n = 50
        t = np.linspace(0, 10 * np.pi, n)
        base = np.column_stack([np.sin(t), np.cos(t), np.sin(t / 2)])

        # 个人轨迹是群体轨迹的平移（时间偏移）
        personal = np.roll(base, shift=5, axis=0)

        detector = DynamicMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_ts=personal,
            group_ts=base,
        )

        # DTW 应该能处理时间偏移，距离应该较小
        assert distance < 0.5

    def test_different_lengths(self, mismatch_config):
        """测试不同长度的序列。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        personal = np.random.randn(30, 3)  # 30 个时间步
        group = np.random.randn(100, 3)    # 100 个时间步

        detector = DynamicMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_ts=personal,
            group_ts=group,
        )

        assert distance >= 0


# ============================================================================
# PersonalIndependenceModel 测试
# ============================================================================

class TestPersonalIndependenceModel:
    def test_create_model(self, mismatch_config):
        """测试模型的创建。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        model = PersonalIndependenceModel(mismatch_config)
        assert model is not None
        assert model.method == "autocorrelation"

    def test_compute_self_consistency_basic(self, mismatch_config, sample_personal_time_series):
        """测试基本的自洽性计算。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(
            time_series=sample_personal_time_series,
        )

        assert isinstance(consistency, float)
        assert 0 <= consistency <= 1

    def test_perfectly_consistent(self, mismatch_config):
        """测试完全自洽（恒定信号）的情况。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        # 恒定信号：自洽性应该很高
        ts = np.ones((100, 3)) * 0.5

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(time_series=ts)

        # 恒定信号的自洽性应该接近 1
        assert consistency > 0.9

    def test_completely_random(self, mismatch_config):
        """测试完全随机信号的情况。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        # 纯白噪声：自洽性应该很低
        np.random.seed(42)
        ts = np.random.randn(100, 3)

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(time_series=ts)

        # 完全随机信号的自洽性应该较低
        assert consistency < 0.5

    def test_single_time_step(self, mismatch_config):
        """测试单时间步的边界情况。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        # 只有一个时间步
        ts = np.array([[0.5, 0.5, 0.5]])

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(time_series=ts)

        # 单时间步，自洽性应为 1（完全自洽）
        assert consistency == 1.0

    def test_periodic_signal(self, mismatch_config):
        """测试周期性信号（高自洽性）的情况。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        n = 200
        t = np.linspace(0, 20 * np.pi, n)
        # 完美周期信号
        ts = np.column_stack([
            np.sin(t),
            np.cos(t + 0.1),
            np.sin(t / 2 + 0.2),
        ])

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(time_series=ts)

        # 周期信号的自洽性应该较高
        assert consistency > 0.5

    def test_sudden_shift(self, mismatch_config):
        """测试包含突变信号（低自洽性）的情况。"""
        from tender.mismatch.personal_independence import PersonalIndependenceModel

        n = 100
        # 前半段是一致信号，后半段完全不同的信号
        ts = np.zeros((n, 3))
        ts[:50] = 0.2
        ts[50:] = 0.8

        model = PersonalIndependenceModel(mismatch_config)
        consistency = model.compute_self_consistency(time_series=ts)

        # 存在突变的信号，自洽性应该较低
        assert consistency < 0.6


# ============================================================================
# 边界与异常测试
# ============================================================================

class TestMismatchEdgeCases:
    def test_empty_point_cloud(self, mismatch_config):
        """测试空点云的边界情况。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        detector = TopologicalMismatchDetector(mismatch_config)
        with pytest.raises(ValueError):
            detector.compute_distance(
                personal_point_cloud=np.array([]),
                group_point_cloud=np.random.randn(10, 3),
            )

    def test_empty_time_series(self, mismatch_config):
        """测试空时间序列的边界情况。"""
        from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector

        detector = DynamicMismatchDetector(mismatch_config)
        with pytest.raises(ValueError):
            detector.compute_distance(
                personal_ts=np.array([]),
                group_ts=np.random.randn(10, 3),
            )

    def test_missing_config(self):
        """测试缺少配置的边界情况。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        with pytest.raises(KeyError):
            TopologicalMismatchDetector({})

    def test_one_dimensional_data(self, mismatch_config):
        """测试一维数据的兼容性。"""
        from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector

        personal = np.random.randn(10, 1)
        group = np.random.randn(100, 1)

        detector = TopologicalMismatchDetector(mismatch_config)
        distance = detector.compute_distance(
            personal_point_cloud=personal,
            group_point_cloud=group,
        )

        assert distance >= 0
        assert isinstance(distance, float)
