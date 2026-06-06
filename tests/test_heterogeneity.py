"""
Tender v2.0 — Embracing Heterogeneity
异质性分析模块测试

覆盖范围：
  - HeterogeneityMetrics 数据类的创建和序列化
  - TopologicalDisconnectAnalyzer 的脱离度计算
  - LoopDetector 的环状结构检测
  - CausalFragmentationAnalyzer 的碎片化分析
  - PowerCentralityAnalyzer 的影响力集中度分析
  - ParticipationGiniAnalyzer 的基尼系数计算
  - IsolateAnalyzer 的离群者类型分类
  - 边界情况：所有成员完全一致、单一成员、环状结构不存在
"""

import pytest
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def mock_topology_result():
    """模拟拓扑分析结果。"""
    from tender.topology_analysis.base import TopologyResult

    return TopologyResult(
        n_clusters=3,
        cluster_labels=np.array([0, 0, 1, 1, 2, 2, -1]),
        ring_exists=True,
        outlier_ratio=1 / 7,
        outlier_scores={"user_0": 0.1, "user_1": 0.1, "user_2": 0.2,
                        "user_3": 0.2, "user_4": 0.3, "user_5": 0.3,
                        "user_6": 0.9},
        global_centroid=np.array([0.2, 0.5, 0.3]),
        point_cloud=np.random.randn(7, 3),
        trajectory=np.random.randn(10, 3),
        edges=[("user_0", "user_1", 0.9), ("user_2", "user_3", 0.8)],
    )


@pytest.fixture
def mock_causal_result():
    """模拟因果分析结果。"""
    from tender.causal_analysis.base import CausalEdge, CausalResult

    edges = [
        CausalEdge("user_0", "user_1", 0.8, 3),
        CausalEdge("user_0", "user_2", 0.6, 5),
        CausalEdge("user_1", "user_3", 0.7, 2),
        CausalEdge("user_4", "user_5", 0.5, 4),
    ]

    return CausalResult(
        causal_edges=edges,
        in_degrees={"user_1": 1, "user_2": 1, "user_3": 1, "user_5": 1, "user_0": 0, "user_4": 0},
        out_degrees={"user_0": 2, "user_1": 1, "user_4": 1, "user_2": 0, "user_3": 0, "user_5": 0},
        super_spreaders=["user_0"],
        causal_density=0.15,
    )


@pytest.fixture
def mock_cognition_states():
    """模拟认知状态。"""
    from tender.cognition.base import CognitionState, CognitivePhase

    return {
        "user_0": CognitionState(cognitive_load=0.3, understanding_level=0.8, confusion_level=0.1,
                                  attention_score=0.9, cognitive_flexibility=0.7,
                                  phase_confidence=0.9, cognitive_phase=CognitivePhase.APPLICATION_CONSOLIDATION),
        "user_1": CognitionState(cognitive_load=0.8, understanding_level=0.2, confusion_level=0.7,
                                  attention_score=0.4, cognitive_flexibility=0.3,
                                  phase_confidence=0.5, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING),
        "user_2": CognitionState(cognitive_load=0.5, understanding_level=0.5, confusion_level=0.4,
                                  attention_score=0.6, cognitive_flexibility=0.5,
                                  phase_confidence=0.7, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING),
        "user_3": CognitionState(cognitive_load=0.2, understanding_level=0.9, confusion_level=0.1,
                                  attention_score=0.8, cognitive_flexibility=0.8,
                                  phase_confidence=0.9, cognitive_phase=CognitivePhase.APPLICATION_CONSOLIDATION),
        "user_4": CognitionState(cognitive_load=0.6, understanding_level=0.4, confusion_level=0.5,
                                  attention_score=0.5, cognitive_flexibility=0.4,
                                  phase_confidence=0.6, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING),
        "user_5": CognitionState(cognitive_load=0.4, understanding_level=0.6, confusion_level=0.3,
                                  attention_score=0.7, cognitive_flexibility=0.6,
                                  phase_confidence=0.8, cognitive_phase=CognitivePhase.EXPLORATION),
        "user_6": CognitionState(cognitive_load=0.1, understanding_level=0.1, confusion_level=0.1,
                                  attention_score=0.1, cognitive_flexibility=0.1,
                                  phase_confidence=0.1, cognitive_phase=CognitivePhase.CORE_UNDERSTANDING),
    }


@pytest.fixture
def mock_messages() -> Dict[str, List[Dict]]:
    """模拟消息数据。"""
    return {
        "user_0": [{"content": "讲得好！", "timestamp": 100.0}, {"content": "再来一个例子", "timestamp": 105.0}],
        "user_1": [{"content": "不懂...", "timestamp": 102.0}, {"content": "谁能解释？", "timestamp": 108.0}],
        "user_2": [{"content": "明白了，谢谢！", "timestamp": 103.0}],
        "user_3": [{"content": "我补充一点...", "timestamp": 101.0}],
        "user_4": [{"content": "作业完成了", "timestamp": 110.0}],
        "user_5": [{"content": "好的", "timestamp": 100.0}],
        "user_6": [],  # 无消息的成员
    }


@pytest.fixture
def heterogeneity_config() -> Dict[str, Any]:
    """标准的异质性分析配置。"""
    return {
        "enabled": True,
        "topology_disconnect": {
            "space_weight": 0.4,
            "causal_weight": 0.3,
            "cognition_weight": 0.3,
            "outlier_threshold": 0.6,
        },
        "loop_detection": {
            "min_persistence": 0.1,
            "max_loops": 5,
        },
        "causal_fragmentation": {
            "min_component_size": 1,
        },
        "participation_gini": {
            "min_messages": 0,
        },
    }


# ============================================================================
# HeterogeneityMetrics 数据类测试
# ============================================================================

class TestHeterogeneityMetrics:
    def test_creation(self):
        """测试标准创建。"""
        from tender.heterogeneity.base import HeterogeneityMetrics

        metrics = HeterogeneityMetrics(
            topological_richness=0.7,
            loop_strength=0.3,
            causal_fragmentation=0.5,
            component_separation=0.8,
            temporal_asynchrony=0.2,
            linguistic_divergence=0.4,
            participation_gini=0.6,
            cluster_ids=[0, 1, 2],
            cluster_members={0: ["user_0", "user_1"], 1: ["user_2", "user_3"]},
            outlier_types={"user_6": "VOLUNTARY_ISOLATE"},
        )

        assert metrics.topological_richness == 0.7
        assert metrics.loop_strength == 0.3
        assert metrics.participation_gini == 0.6
        assert "user_6" in metrics.outlier_types
        assert metrics.outlier_types["user_6"] == "VOLUNTARY_ISOLATE"

    def test_invalid_metrics(self):
        """测试无效的指标值。"""
        from tender.heterogeneity.base import HeterogeneityMetrics

        with pytest.raises(ValueError):
            HeterogeneityMetrics(
                topological_richness=-0.1,  # 不能为负
                loop_strength=0.0,
                causal_fragmentation=0.0,
                component_separation=0.0,
                temporal_asynchrony=0.0,
                linguistic_divergence=0.0,
                participation_gini=0.0,
                cluster_ids=[],
                cluster_members={},
                outlier_types={},
            )

    def test_from_dict(self):
        """测试从字典创建。"""
        from tender.heterogeneity.base import HeterogeneityMetrics

        data = {
            "topological_richness": 0.8,
            "loop_strength": 0.2,
            "causal_fragmentation": 0.4,
            "component_separation": 0.7,
            "temporal_asynchrony": 0.3,
            "linguistic_divergence": 0.5,
            "participation_gini": 0.6,
            "cluster_ids": [0, 1],
            "cluster_members": {0: ["a"], 1: ["b"]},
            "outlier_types": {"c": "PURE_DIVERSE"},
        }
        metrics = HeterogeneityMetrics.from_dict(data)
        assert metrics.topological_richness == 0.8
        assert metrics.participation_gini == 0.6


# ============================================================================
# TopologicalDisconnectAnalyzer 测试
# ============================================================================

class TestTopologicalDisconnectAnalyzer:
    def test_create_analyzer(self, heterogeneity_config):
        """测试分析器的创建。"""
        from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer

        analyzer = TopologicalDisconnectAnalyzer(heterogeneity_config)
        assert analyzer is not None
        assert analyzer.space_weight == 0.4
        assert analyzer.causal_weight == 0.3
        assert analyzer.cognition_weight == 0.3

    def test_compute_disconnect_basic(self, heterogeneity_config, mock_topology_result, mock_causal_result, mock_cognition_states):
        """测试基本的脱离度计算。"""
        from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer

        analyzer = TopologicalDisconnectAnalyzer(heterogeneity_config)

        # 测试一个正常融入的成员
        score = analyzer.compute_disconnect(
            member_id="user_0",
            topology_result=mock_topology_result,
            causal_result=mock_causal_result,
            cognition_states=mock_cognition_states,
        )

        assert score is not None
        assert 0 <= score.total <= 1
        assert 0 <= score.space_disconnect <= 1
        assert 0 <= score.causal_disconnect <= 1
        assert 0 <= score.cognition_disconnect <= 1

    def test_outlier_member(self, heterogeneity_config, mock_topology_result, mock_causal_result, mock_cognition_states):
        """测试离群成员的脱离度计算。"""
        from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer

        analyzer = TopologicalDisconnectAnalyzer(heterogeneity_config)

        # User 6 是离群者（标签为 -1）
        score = analyzer.compute_disconnect(
            member_id="user_6",
            topology_result=mock_topology_result,
            causal_result=mock_causal_result,
            cognition_states=mock_cognition_states,
        )

        # 离群者的脱离度应该较高
        assert score.total > 0.5

    def test_uniform_disconnect(self, heterogeneity_config, mock_cognition_states):
        """测试所有成员完全一致时的脱离度。"""
        from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer
        from tender.topology_analysis.base import TopologyResult

        # 所有成员都完全一致
        uniform_topology = TopologyResult(
            n_clusters=1,
            cluster_labels=np.array([0, 0, 0]),
            ring_exists=False,
            outlier_ratio=0.0,
            outlier_scores={"a": 0.0, "b": 0.0, "c": 0.0},
            global_centroid=np.zeros(3),
            point_cloud=np.zeros((3, 3)),
            trajectory=np.zeros((10, 3)),
            edges=[],
        )
        from tender.causal_analysis.base import CausalResult
        uniform_causal = CausalResult(
            causal_edges=[], in_degrees={}, out_degrees={}, super_spreaders=[], causal_density=0.0,
        )

        analyzer = TopologicalDisconnectAnalyzer(heterogeneity_config)
        scores = []
        for mid in ["a", "b", "c"]:
            score = analyzer.compute_disconnect(mid, uniform_topology, uniform_causal, mock_cognition_states)
            scores.append(score.total)

        # 脱离度差异应该很小
        assert np.std(scores) < 0.2


# ============================================================================
# LoopDetector 测试
# ============================================================================

class TestLoopDetector:
    def test_detect_loops(self, heterogeneity_config, mock_topology_result):
        """测试环状结构检测。"""
        from tender.heterogeneity.topology_analysis import LoopDetector

        # 只有当拓扑结果中有环时才能检测到
        if mock_topology_result.ring_exists:
            detector = LoopDetector(heterogeneity_config)
            loops = detector.detect_loops(mock_topology_result)

            assert isinstance(loops, list)
            if loops:
                loop = loops
                assert hasattr(loop, 'persistence')
                assert hasattr(loop, 'member_ids')
                assert loop.persistence >= 0

    def test_no_loop(self, heterogeneity_config):
        """测试无环状结构的情况。"""
        from tender.heterogeneity.topology_analysis import LoopDetector
        from tender.topology_analysis.base import TopologyResult

        no_ring_result = TopologyResult(
            n_clusters=3,
            cluster_labels=np.array([0, 0, 1, 1, 2, 2]),
            ring_exists=False,
            outlier_ratio=0.0,
            outlier_scores={},
            global_centroid=np.zeros(3),
            point_cloud=np.random.randn(6, 3),
            trajectory=np.random.randn(10, 3),
            edges=[],
        )

        detector = LoopDetector(heterogeneity_config)
        loops = detector.detect_loops(no_ring_result)

        assert isinstance(loops, list)
        assert len(loops) == 0

    def test_compute_richness(self, heterogeneity_config, mock_topology_result):
        """测试拓扑丰富度计算。"""
        from tender.heterogeneity.topology_analysis import LoopDetector

        detector = LoopDetector(heterogeneity_config)
        richness = detector.compute_richness(mock_topology_result)

        assert isinstance(richness, float)
        assert richness >= 0


# ============================================================================
# CausalFragmentationAnalyzer 测试
# ============================================================================

class TestCausalFragmentationAnalyzer:
    def test_compute_fragmentation(self, heterogeneity_config, mock_causal_result):
        """测试因果碎片化计算。"""
        from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer

        analyzer = CausalFragmentationAnalyzer(heterogeneity_config)
        metrics = analyzer.compute_fragmentation(mock_causal_result)

        assert metrics is not None
        assert 0 <= metrics.fragmentation_index <= 1
        assert metrics.n_components >= 1
        assert 0 <= metrics.largest_component_ratio <= 1

    def test_fully_connected(self, heterogeneity_config):
        """测试完全连通的因果网络。"""
        from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer
        from tender.causal_analysis.base import CausalEdge, CausalResult

        # 全连通网络
        edges = [
            CausalEdge("a", "b", 0.8, 1),
            CausalEdge("a", "c", 0.7, 1),
            CausalEdge("b", "c", 0.6, 1),
        ]
        result = CausalResult(
            causal_edges=edges,
            in_degrees={"b": 1, "c": 2, "a": 0},
            out_degrees={"a": 2, "b": 1, "c": 0},
            super_spreaders=["a"],
            causal_density=0.5,
        )

        analyzer = CausalFragmentationAnalyzer(heterogeneity_config)
        metrics = analyzer.compute_fragmentation(result)

        # 全连通网络，碎片化指数应该接近 0
        assert metrics.fragmentation_index < 0.3

    def test_no_edges(self, heterogeneity_config):
        """测试无边的因果网络。"""
        from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer
        from tender.causal_analysis.base import CausalResult

        result = CausalResult(
            causal_edges=[], in_degrees={}, out_degrees={}, super_spreaders=[], causal_density=0.0,
        )

        analyzer = CausalFragmentationAnalyzer(heterogeneity_config)
        metrics = analyzer.compute_fragmentation(result)

        # 无边，碎片化指数应为最高
        assert metrics.fragmentation_index == 1.0
        assert metrics.n_components == 0


# ============================================================================
# ParticipationGiniAnalyzer 测试
# ============================================================================

class TestParticipationGiniAnalyzer:
    def test_compute_gini(self, heterogeneity_config):
        """测试基尼系数的计算。"""
        from tender.heterogeneity.behavior_analysis import ParticipationGiniAnalyzer

        analyzer = ParticipationGiniAnalyzer(heterogeneity_config)

        # 完全平均分布
        gini = analyzer.compute_gini({"a": 1, "b": 1, "c": 1})
        assert gini == 0.0  # 完全平等

        # 完全不平等分布
        gini = analyzer.compute_gini({"a": 10, "b": 0, "c": 0})
        assert gini == 1.0  # 完全不平等

        # 一般情况
        gini = analyzer.compute_gini({"a": 5, "b": 3, "c": 2})
        assert 0 < gini < 1

    def test_single_member(self, heterogeneity_config):
        """测试单一成员的情况。"""
        from tender.heterogeneity.behavior_analysis import ParticipationGiniAnalyzer

        analyzer = ParticipationGiniAnalyzer(heterogeneity_config)
        gini = analyzer.compute_gini({"a": 5})

        # 单一成员，基尼系数应为 0
        assert gini == 0.0


# ============================================================================
# IsolateAnalyzer 测试
# ============================================================================

class TestIsolateAnalyzer:
    def test_classify_voluntary_isolate(self, heterogeneity_config):
        """测试分类为自愿隔离者。"""
        from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer
        from tender.heterogeneity.topology_analysis import DisconnectScore

        # 高脱离度 + 高自洽性
        score = DisconnectScore(total=0.85, space_disconnect=0.9, causal_disconnect=0.8, cognition_disconnect=0.7)
        personal_profile = {"self_consistency": 0.9, "wellbeing": 0.8}

        analyzer = IsolateAnalyzer(heterogeneity_config)
        result = analyzer.classify(disconnect_score=score, personal_profile=personal_profile, group_profile=None)

        assert result is not None
        assert result.value == "VOLUNTARY_ISOLATE"

    def test_classify_involuntary_outcast(self, heterogeneity_config):
        """测试分类为被迫排斥者。"""
        from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer
        from tender.heterogeneity.topology_analysis import DisconnectScore

        # 高脱离度 + 低自洽性
        score = DisconnectScore(total=0.85, space_disconnect=0.9, causal_disconnect=0.8, cognition_disconnect=0.7)
        personal_profile = {"self_consistency": 0.2, "wellbeing": 0.3}

        analyzer = IsolateAnalyzer(heterogeneity_config)
        result = analyzer.classify(disconnect_score=score, personal_profile=personal_profile, group_profile=None)

        assert result is not None
        assert result.value == "INVOLUNTARY_OUTCAST"

    def test_no_classification_for_low_disconnect(self, heterogeneity_config):
        """测试低脱离度时不进行分类。"""
        from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer
        from tender.heterogeneity.topology_analysis import DisconnectScore

        score = DisconnectScore(total=0.2, space_disconnect=0.1, causal_disconnect=0.1, cognition_disconnect=0.1)
        personal_profile = {"self_consistency": 0.8, "wellbeing": 0.9}

        analyzer = IsolateAnalyzer(heterogeneity_config)
        result = analyzer.classify(disconnect_score=score, personal_profile=personal_profile, group_profile=None)

        # 低脱离度，不应该是离群者
        assert result is None


# ============================================================================
# 边界与异常测试
# ============================================================================

class TestHeterogeneityEdgeCases:
    def test_single_member_group(self, heterogeneity_config):
        """测试只有一个成员的群体。"""
        from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer
        from tender.heterogeneity.behavior_analysis import ParticipationGiniAnalyzer
        from tender.topology_analysis.base import TopologyResult
        from tender.causal_analysis.base import CausalResult

        single_topology = TopologyResult(
            n_clusters=1,
            cluster_labels=np.array(,
            ring_exists=False,
            outlier_ratio=0.0,
            outlier_scores={"a": 0.0},
            global_centroid=np.zeros(3),
            point_cloud=np.zeros((1, 3)),
            trajectory=np.zeros((10, 3)),
            edges=[],
        )
        single_causal = CausalResult(
            causal_edges=[], in_degrees={}, out_degrees={}, super_spreaders=[], causal_density=0.0,
        )

        disconnect_analyzer = TopologicalDisconnectAnalyzer(heterogeneity_config)
        score = disconnect_analyzer.compute_disconnect("a", single_topology, single_causal, {})

        # 唯一成员的脱离度应较低
        assert score.total < 0.3

        # 基尼系数应为 0
        gini_analyzer = ParticipationGiniAnalyzer(heterogeneity_config)
        gini = gini_analyzer.compute_gini({"a": 1})
        assert gini == 0.0
