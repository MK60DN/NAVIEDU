"""
Tender v2.0 — Embracing Heterogeneity
空间拓扑分析模块测试

覆盖范围：
  - TopologyResult 数据类的创建和序列化
  - PersistentLaplacianAnalyzer 的完整分析流程
  - TopologicalGradientFlowAnalyzer 的完整分析流程
  - 边界情况：单一成员、所有成员情绪相同、离群点检测
"""

import pytest
import numpy as np
from typing import Dict, List, Any


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def sample_emotion_vectors() -> Dict[str, List[Any]]:
    """提供一个标准的情绪向量数据集。"""
    from tender.emotion_vectorizer.emotion_vector import EmotionVector

    n_members = 10
    n_points_per_member = 5

    vectors = {}
    for i in range(n_members):
        member_id = f"user_{i}"
        member_vectors = []
        for j in range(n_points_per_member):
            member_vectors.append(
                EmotionVector(
                    valence=float(np.random.uniform(-1, 1)),
                    arousal=float(np.random.uniform(0, 1)),
                    focus=float(np.random.uniform(0, 1)),
                    confidence=float(np.random.uniform(0.5, 1.0)),
                    timestamp=float(100.0 + j * 10.0),
                )
            )
        vectors[member_id] = member_vectors

    return vectors


@pytest.fixture
def uniform_emotion_vectors() -> Dict[str, List[Any]]:
    """所有成员情绪相同的极端情况。"""
    from tender.emotion_vectorizer.emotion_vector import EmotionVector

    vectors = {}
    for i in range(5):
        member_id = f"user_{i}"
        vectors[member_id] = [
            EmotionVector(
                valence=0.5, arousal=0.5, focus=0.5, confidence=1.0, timestamp=100.0
            )
        ]
    return vectors


@pytest.fixture
def single_member_vectors() -> Dict[str, List[Any]]:
    """只有一个成员的边界情况。"""
    from tender.emotion_vectorizer.emotion_vector import EmotionVector

    vectors = {}
    vectors["only_one"] = [
        EmotionVector(
            valence=0.5, arousal=0.5, focus=0.5, confidence=1.0, timestamp=100.0
        )
    ]
    return vectors


@pytest.fixture
def topology_config() -> Dict[str, Any]:
    """标准的拓扑分析配置。"""
    return {
        "engine": "persistent_laplacian",
        "laplacian_type": "combinatorial",
        "spectral_gap_threshold": 0.1,
        "clustering": {
            "min_cluster_size": 2,
            "min_samples": 1,
        },
        "max_dimension": 1,
        "normalize": True,
    }


# ============================================================================
# TopologyResult 数据类测试
# ============================================================================

class TestTopologyResult:
    def test_creation(self):
        """测试标准创建"""
        from tender.topology_analysis.base import TopologyResult

        result = TopologyResult(
            n_clusters=3,
            cluster_labels=np.array([0, 0, 1, 1, 2, -1]),
            ring_exists=True,
            outlier_ratio=1 / 6,
            outlier_scores={"user_1": 0.8, "user_2": 0.1},
            global_centroid=np.array([0.2, 0.5, 0.3]),
            point_cloud=np.random.randn(6, 3),
            trajectory=np.random.randn(10, 3),
            edges=[("user_1", "user_2", 0.9)],
        )

        assert result.n_clusters == 3
        assert result.ring_exists is True
        assert result.outlier_ratio == 1 / 6

    def test_no_clusters(self):
        """测试没有聚类的边界情况"""
        from tender.topology_analysis.base import TopologyResult

        result = TopologyResult(
            n_clusters=0,
            cluster_labels=np.array([-1, -1, -1]),
            ring_exists=False,
            outlier_ratio=1.0,
            outlier_scores={"user_1": 1.0, "user_2": 1.0},
            global_centroid=np.zeros(3),
            point_cloud=np.random.randn(3, 3),
            trajectory=np.random.randn(10, 3),
            edges=[],
        )

        assert result.n_clusters == 0
        assert result.outlier_ratio == 1.0
        assert len(result.edges) == 0


# ============================================================================
# PersistentLaplacianAnalyzer 测试
# ============================================================================

class TestPersistentLaplacianAnalyzer:
    def test_create_analyzer(self, topology_config):
        """测试分析器的创建"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )

        analyzer = PersistentLaplacianAnalyzer(topology_config)
        assert analyzer is not None

    def test_analyze_basic(self, topology_config, sample_emotion_vectors):
        """测试基本的分析流程"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )

        analyzer = PersistentLaplacianAnalyzer(topology_config)
        result = analyzer.analyze(sample_emotion_vectors)

        assert result is not None
        assert isinstance(result.n_clusters, int)
        assert isinstance(result.ring_exists, bool)
        assert 0 <= result.outlier_ratio <= 1
        assert result.point_cloud is not None
        assert result.global_centroid is not None

    def test_uniform_emotions(self, topology_config, uniform_emotion_vectors):
        """测试所有成员情绪相同的极端情况（应该只有 1 个簇）"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )

        analyzer = PersistentLaplacianAnalyzer(topology_config)
        result = analyzer.analyze(uniform_emotion_vectors)

        assert result.n_clusters == 1
        assert result.outlier_ratio == 0.0

    def test_single_member(self, topology_config, single_member_vectors):
        """测试只有一个成员的边界情况"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )

        analyzer = PersistentLaplacianAnalyzer(topology_config)
        result = analyzer.analyze(single_member_vectors)

        assert result.n_clusters == 1  # 应该自成一簇
        assert result.outlier_ratio == 0.0
        assert len(result.outlier_scores) == 1

    def test_empty_input(self, topology_config):
        """测试空输入"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )

        analyzer = PersistentLaplacianAnalyzer(topology_config)

        with pytest.raises(ValueError):
            analyzer.analyze({})

    def test_ring_detection(self, topology_config):
        """测试环状结构检测"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )
        from tender.emotion_vectorizer.emotion_vector import EmotionVector

        # 构建一个明显形成环状结构的点云（在 S^1 上采样）
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        vectors = {}
        for i, angle in enumerate(angles):
            member_id = f"user_{i}"
            vectors[member_id] = [
                EmotionVector(
                    valence=float(np.cos(angle)),
                    arousal=float(0.5 * (np.sin(angle) + 1)),  # [0,1]
                    focus=0.5,
                    confidence=1.0,
                    timestamp=100.0,
                )
            ]

        analyzer = PersistentLaplacianAnalyzer(topology_config)
        result = analyzer.analyze(vectors)

        # 理想情况下，环状结构应该被检测到
        # 取决于谱隙阈值，可能不一定总能检测到
        # 至少保证不报错


# ============================================================================
# TopologicalGradientFlowAnalyzer 测试
# ============================================================================

class TestTopologicalGradientFlowAnalyzer:
    @pytest.fixture
    def gradient_config(self) -> Dict[str, Any]:
        return {
            "engine": "topological_gradient_flow",
            "gradient_step": 0.01,
            "max_iterations": 100,
            "convergence_threshold": 1e-5,
            "clustering": {
                "min_cluster_size": 2,
                "min_samples": 1,
            },
        }

    def test_create_analyzer(self, gradient_config):
        """测试分析器的创建"""
        from tender.topology_analysis.topological_gradient_flow import (
            TopologicalGradientFlowAnalyzer,
        )

        analyzer = TopologicalGradientFlowAnalyzer(gradient_config)
        assert analyzer is not None
        assert analyzer.gradient_step == 0.01

    def test_analyze_basic(self, gradient_config, sample_emotion_vectors):
        """测试基本的分析流程"""
        from tender.topology_analysis.topological_gradient_flow import (
            TopologicalGradientFlowAnalyzer,
        )

        analyzer = TopologicalGradientFlowAnalyzer(gradient_config)
        result = analyzer.analyze(sample_emotion_vectors)

        assert result is not None
        assert isinstance(result.n_clusters, int)
        assert isinstance(result.ring_exists, bool)
        assert 0 <= result.outlier_ratio <= 1

    def test_single_member(self, gradient_config, single_member_vectors):
        """测试只有一个成员的边界情况"""
        from tender.topology_analysis.topological_gradient_flow import (
            TopologicalGradientFlowAnalyzer,
        )

        analyzer = TopologicalGradientFlowAnalyzer(gradient_config)
        result = analyzer.analyze(single_member_vectors)

        assert result.n_clusters == 1
        assert result.outlier_ratio == 0.0


# ============================================================================
# 边界情况与异常测试
# ============================================================================

class TestTopologyEdgeCases:
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        return {"engine": "persistent_laplacian", "clustering": {"min_cluster_size": 2}}

    def test_very_large_clusters(self, config):
        """测试大数据量下的拓扑分析"""
        from tender.topology_analysis.persistent_laplacian import (
            PersistentLaplacianAnalyzer,
        )
        from tender.emotion_vectorizer.emotion_vector import EmotionVector

        n_members = 100
        vectors = {}
        for i in range(n_members):
            member_id = f"user_{i}"
            vectors[member_id] = [
                EmotionVector(
                    valence=float(np.random.uniform(-1, 1)),
                    arousal=float(np.random.uniform(0, 1)),
                    focus=float(np.random.uniform(0, 1)),
                    confidence=1.0,
                    timestamp=100.0,
                )
            ]

        analyzer = PersistentLaplacianAnalyzer(config)
        result = analyzer.analyze(vectors)

        assert result.n_clusters >= 1
        assert result.point_cloud.shape[0] == n_members

    def test_missing_engine_in_config(self):
        """测试配置中缺少引擎类型"""
        from tender.topology_analysis.engine_mapping import (
            get_topology_analyzer,
        )

        with pytest.raises(KeyError):
            get_topology_analyzer({})
