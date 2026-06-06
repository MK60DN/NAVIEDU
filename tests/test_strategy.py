"""
Tender v2.0 — Embracing Heterogeneity
策略推理模块测试

覆盖范围：
  - StrategyResult 数据类的创建和序列化
  - CausalRLEngine 的策略推理流程
  - RuleBasedEngine 的规则推理流程
  - HeterogeneityCoordinationLayer 的异质性协调流程
  - 边界情况：所有成员状态良好、群体处于危机状态
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass


# ============================================================================
# Mock 与 Fixture
# ============================================================================

@pytest.fixture
def mock_fusion_result():
    """模拟融合结果。"""
    from tender.fusion.base import FusionResult

    return FusionResult(
        fused_features=np.random.randn(32),
        health_index=0.75,
        forecast=np.random.randn(3),
        dynamic_graph=None,  # 简化测试
    )


@pytest.fixture
def mock_synergy_result():
    """模拟协同结果。"""
    from tender.synergy.base import SynergyResult, SynergyMode

    return SynergyResult(
        combined_feature=np.random.randn(32),
        synergy_score=0.8,
        dominant_dimension="cognition",
        synergy_mode=SynergyMode.HARMONIOUS,
        adaptation_score=0.7,
        recommendation=None,
    )


@pytest.fixture
def mock_mismatch_metrics():
    """模拟不匹配度量。"""
    return {
        "user_0": {"structural_distance": 0.2, "dynamic_distance": 0.3, "personal_self_consistency": 0.9},
        "user_1": {"structural_distance": 0.8, "dynamic_distance": 0.9, "personal_self_consistency": 0.8},
        "user_2": {"structural_distance": 0.7, "dynamic_distance": 0.6, "personal_self_consistency": 0.3},
    }


@pytest.fixture
def mock_heterogeneity_metrics():
    """模拟异质性指标。"""
    from tender.heterogeneity.base import HeterogeneityMetrics

    return HeterogeneityMetrics(
        topological_richness=0.6,
        loop_strength=0.3,
        causal_fragmentation=0.5,
        component_separation=0.7,
        temporal_asynchrony=0.4,
        linguistic_divergence=0.5,
        participation_gini=0.6,
        cluster_ids=[0, 1, 2],
        cluster_members={0: ["user_0"], 1: ["user_1"], 2: ["user_2"]},
        outlier_types={"user_1": "VOLUNTARY_ISOLATE", "user_2": "INVOLUNTARY_OUTCAST"},
    )


@pytest.fixture
def strategy_config() -> Dict[str, Any]:
    """标准的策略推理配置。"""
    return {
        "engine": "causal_rl",
        "learning_rate": 1e-4,
        "gamma": 0.95,
        "epsilon": 0.1,
        "heterogeneity_coordination": {
            "enabled": True,
            "strategy_count": 3,
        },
    }


# ============================================================================
# StrategyResult 与 RiskLevel 数据类测试
# ============================================================================

class TestStrategyResult:
    def test_creation(self):
        """测试标准创建。"""
        from tender.strategy.base import StrategyResult, RiskLevel

        result = StrategyResult(
            risk_level=RiskLevel.LOW,
            selected_action=0,
            target_members=["alice", "bob"],
            confidence=0.85,
            rationale=None,
            specific_actions=["继续观察", "保持支持"],
        )

        assert result.risk_level == RiskLevel.LOW
        assert result.selected_action == 0
        assert len(result.target_members) == 2
        assert result.confidence == pytest.approx(0.85)

    def test_risk_level_ordering(self):
        """测试风险等级的排序。"""
        from tender.strategy.base import RiskLevel

        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL
        assert RiskLevel.LOW < RiskLevel.CRITICAL

    def test_invalid_confidence(self):
        """测试无效的置信度。"""
        from tender.strategy.base import StrategyResult, RiskLevel

        with pytest.raises(ValueError):
            StrategyResult(
                risk_level=RiskLevel.LOW,
                selected_action=0,
                target_members=[],
                confidence=1.5,  # 不能超过 1
                rationale=None,
                specific_actions=[],
            )

    def test_empty_targets(self):
        """测试空目标成员列表。"""
        from tender.strategy.base import StrategyResult, RiskLevel

        result = StrategyResult(
            risk_level=RiskLevel.LOW,
            selected_action=0,
            target_members=[],  # 空列表也是允许的（不需要干预时）
            confidence=0.0,
            rationale=None,
            specific_actions=[],
        )

        assert len(result.target_members) == 0
        assert len(result.specific_actions) == 0


# ============================================================================
# CausalRLEngine 测试
# ============================================================================

class TestCausalRLEngine:
    def test_create_engine(self, strategy_config):
        """测试引擎的创建。"""
        from tender.strategy.causal_rl import CausalRLEngine

        engine = CausalRLEngine(strategy_config)
        assert engine is not None
        assert engine.learning_rate == 1e-4
        assert engine.gamma == 0.95

    def test_reason_basic(self, strategy_config, mock_fusion_result, mock_synergy_result, mock_mismatch_metrics):
        """测试基本的推理流程。"""
        from tender.strategy.causal_rl import CausalRLEngine

        engine = CausalRLEngine(strategy_config)
        result = engine.reason(
            fusion_result=mock_fusion_result,
            synergy_result=mock_synergy_result,
            mismatch_metrics=mock_mismatch_metrics,
        )

        assert result is not None
        assert isinstance(result.risk_level, Any)
        assert result.confidence >= 0
        assert isinstance(result.target_members, list)

    def test_healthy_group(self, strategy_config):
        """测试群体状态良好时应该输出低风险策略。"""
        from tender.strategy.causal_rl import CausalRLEngine
        from tender.fusion.base import FusionResult
        from tender.synergy.base import SynergyResult, SynergyMode

        # 所有指标都很好
        fusion = FusionResult(
            fused_features=np.random.randn(32),
            health_index=0.95,  # 高健康度
            forecast=np.array([0.9, 0.8, 0.7]),
            dynamic_graph=None,
        )
        synergy = SynergyResult(
            combined_feature=np.random.randn(32),
            synergy_score=0.9,  # 高协同度
            dominant_dimension="cognition",
            synergy_mode=SynergyMode.HARMONIOUS,
            adaptation_score=0.85,  # 高适应度
            recommendation=None,
        )
        mismatch = {
            "a": {"structural_distance": 0.2, "dynamic_distance": 0.1, "personal_self_consistency": 0.9},
            "b": {"structural_distance": 0.3, "dynamic_distance": 0.2, "personal_self_consistency": 0.8},
        }

        engine = CausalRLEngine(strategy_config)
        result = engine.reason(fusion_result=fusion, synergy_result=synergy, mismatch_metrics=mismatch)

        # 群体状态良好，风险等级应较低
        assert result.risk_level in ["LOW", "MEDIUM"] or 0 <= result.risk_level.value <= 1

    def test_crisis_group(self, strategy_config):
        """测试群体处于危机状态时应该输出高风险策略。"""
        from tender.strategy.causal_rl import CausalRLEngine
        from tender.fusion.base import FusionResult
        from tender.synergy.base import SynergyResult, SynergyMode

        # 所有指标都很差
        fusion = FusionResult(
            fused_features=np.random.randn(32),
            health_index=0.2,  # 低健康度
            forecast=np.array([0.1, 0.2, 0.3]),
            dynamic_graph=None,
        )
        synergy = SynergyResult(
            combined_feature=np.random.randn(32),
            synergy_score=0.2,  # 低协同度
            dominant_dimension="emotion",
            synergy_mode=SynergyMode.CONFLICTING,  # 冲突模式
            adaptation_score=0.15,  # 低适应度
            recommendation="需要紧急干预",
        )
        mismatch = {
            "a": {"structural_distance": 0.9, "dynamic_distance": 0.8, "personal_self_consistency": 0.2},
            "b": {"structural_distance": 0.8, "dynamic_distance": 0.9, "personal_self_consistency": 0.3},
        }

        engine = CausalRLEngine(strategy_config)
        result = engine.reason(fusion_result=fusion, synergy_result=synergy, mismatch_metrics=mismatch)

        # 危机状态，风险等级应较高
        assert uint8(result.risk_level.value) >= 2  # 应该至少是 HIGH 或 CRITICAL

    def test_train_method(self, strategy_config):
        """测试 DQN 训练方法。"""
        from tender.strategy.causal_rl import CausalRLEngine

        engine = CausalRLEngine(strategy_config)

        # 模拟一个简单的训练批次
        batch = {
            "states": np.random.randn(10, 32),
            "actions": np.random.choice(4, 10),
            "rewards": np.random.randn(10),
            "next_states": np.random.randn(10, 32),
            "dones": np.zeros(10, dtype=bool),
        }

        # 确保训练不报错
        loss = engine.train(batch)
        assert loss is None or loss >= 0


# ============================================================================
# RuleBasedEngine 测试
# ============================================================================

class TestRuleBasedEngine:
    @pytest.fixture
    def rule_config(self) -> Dict[str, Any]:
        return {
            "engine": "rule_based",
            "rules": {
                "high_risk_threshold": 0.3,
                "medium_risk_threshold": 0.6,
                "action_mapping": {
                    "maintain": 0,
                    "monitor": 1,
                    "support_individual": 2,
                    "intervene": 3,
                },
            },
        }

    def test_create_engine(self, rule_config):
        """测试引擎的创建。"""
        from tender.strategy.rule_based import RuleBasedEngine

        engine = RuleBasedEngine(rule_config)
        assert engine is not None
        assert engine.high_risk_threshold == 0.3

    def test_reason_healthy(self, rule_config):
        """测试健康状态的推理。"""
        from tender.strategy.rule_based import RuleBasedEngine
        from tender.fusion.base import FusionResult
        from tender.synergy.base import SynergyResult, SynergyMode

        fusion = FusionResult(fused_features=np.zeros(32), health_index=0.9, forecast=np.zeros(3), dynamic_graph=None)
        synergy = SynergyResult(combined_feature=np.zeros(32), synergy_score=0.85, dominant_dimension="", synergy_mode=SynergyMode.HARMONIOUS, adaptation_score=0.8, recommendation="")
        mismatch = {"a": {}, "b": {}}

        engine = RuleBasedEngine(rule_config)
        result = engine.reason(fusion_result=fusion, synergy_result=synergy, mismatch_metrics=mismatch)

        # 健康状态，应选择 maintain 动作
        assert result.selected_action == 0  # maintain

    def test_reason_crisis(self, rule_config):
        """测试危机状态的推理。"""
        from tender.strategy.rule_based import RuleBasedEngine
        from tender.fusion.base import FusionResult
        from tender.synergy.base import SynergyResult, SynergyMode

        fusion = FusionResult(fused_features=np.zeros(32), health_index=0.1, forecast=np.zeros(3), dynamic_graph=None)
        synergy = SynergyResult(combined_feature=np.zeros(32), synergy_score=0.2, dominant_dimension="", synergy_mode=SynergyMode.CONFLICTING, adaptation_score=0.1, recommendation="")
        mismatch = {"a": {"needs_intervention": True}, "b": {}}

        engine = RuleBasedEngine(rule_config)
        result = engine.reason(fusion_result=fusion, synergy_result=synergy, mismatch_metrics=mismatch)

        # 危机状态，应选择 intervene 动作
        assert result.selected_action == 3  # intervene


# ============================================================================
# HeterogeneityCoordinationLayer 测试
# ============================================================================

class TestHeterogeneityCoordinationLayer:
    def test_create_layer(self, strategy_config):
        """测试协调层的创建。"""
        from tender.strategy.heterogeneity_coordination import HeterogeneityCoordinationLayer

        layer = HeterogeneityCoordinationLayer(strategy_config)
        assert layer is not None
        assert layer.strategy_count == 3

    def test_coordinate_basic(self, strategy_config, mock_heterogeneity_metrics, mock_mismatch_metrics):
        """测试基本的协调流程。"""
        from tender.strategy.heterogeneity_coordination import HeterogeneityCoordinationLayer
        from tender.strategy.base import StrategyResult, RiskLevel

        # 创建几个基础策略
        base_strategies = [
            StrategyResult(risk_level=RiskLevel.MEDIUM, selected_action=0, target_members=["user_0", "user_1"], confidence=0.7, rationale=None, specific_actions=[]),
            StrategyResult(risk_level=RiskLevel.HIGH, selected_action=1, target_members=["user_2"], confidence=0.8, rationale=None, specific_actions=[]),
            StrategyResult(risk_level=RiskLevel.LOW, selected_action=0, target_members=[], confidence=0.9, rationale=None, specific_actions=[]),
        ]

        layer = HeterogeneityCoordinationLayer(strategy_config)
        final_strategies = layer.coordinate(
            base_strategies=base_strategies,
            heterogeneity_metrics=mock_heterogeneity_metrics,
            mismatch_metrics=mock_mismatch_metrics,
        )

        assert isinstance(final_strategies, list)
        assert len(final_strategies) <= strategy_config["heterogeneity_coordination"]["strategy_count"]

        if final_strategies:
            strategy = final_strategies
            assert isinstance(strategy, StrategyResult)

    def test_no_heterogeneity(self, strategy_config):
        """测试无异质性时的协调。"""
        from tender.strategy.heterogeneity_coordination import HeterogeneityCoordinationLayer
        from tender.strategy.base import StrategyResult, RiskLevel
        from tender.heterogeneity.base import HeterogeneityMetrics

        # 无异质性（所有指标为 0）
        no_hetero = HeterogeneityMetrics(
            topological_richness=0.0, loop_strength=0.0, causal_fragmentation=0.0,
            component_separation=0.0, temporal_asynchrony=0.0, linguistic_divergence=0.0,
            participation_gini=0.0, cluster_ids=[0], cluster_members={0: []}, outlier_types={},
        )

        base_strategies = [
            StrategyResult(risk_level=RiskLevel.LOW, selected_action=0, target_members=[], confidence=0.9, rationale=None, specific_actions=[]),
        ]

        layer = HeterogeneityCoordinationLayer(strategy_config)
        final = layer.coordinate(base_strategies=base_strategies, heterogeneity_metrics=no_hetero, mismatch_metrics={})

        # 无异质性时，策略数量应保持不变
        assert len(final) == len(base_strategies)

    def test_with_outliers(self, strategy_config, mock_heterogeneity_metrics, mock_mismatch_metrics):
        """测试存在离群者时的协调。"""
        from tender.strategy.heterogeneity_coordination import HeterogeneityCoordinationLayer
        from tender.strategy.base import StrategyResult, RiskLevel

        # 创建一个不考虑离群者的基础策略
        base = [
            StrategyResult(risk_level=RiskLevel.LOW, selected_action=0, target_members=["user_0"], confidence=0.9, rationale=None, specific_actions=[]),
        ]

        layer = HeterogeneityCoordinationLayer(strategy_config)
        final = layer.coordinate(base_strategies=base, heterogeneity_metrics=mock_heterogeneity_metrics, mismatch_metrics=mock_mismatch_metrics)

        # 存在离群者时，可能会生成额外的策略来处理他们
        assert len(final) >= 1


# ============================================================================
# 异常与边界测试
# ============================================================================

class TestStrategyEdgeCases:
    def test_missing_fusion_result(self, strategy_config):
        """测试缺少融合结果的边界情况。"""
        from tender.strategy.causal_rl import CausalRLEngine
        from tender.synergy.base import SynergyResult, SynergyMode

        synergy = SynergyResult(
            combined_feature=np.zeros(32), synergy_score=0.5, dominant_dimension="",
            synergy_mode=SynergyMode.HARMONIOUS, adaptation_score=0.5, recommendation="",
        )

        engine = CausalRLEngine(strategy_config)
        with pytest.raises(ValueError):
            engine.reason(fusion_result=None, synergy_result=synergy, mismatch_metrics={})

    def test_all_members_inactive(self, strategy_config):
        """测试所有成员都不活跃的边界情况。"""
        from tender.strategy.causal_rl import CausalRLEngine

        # 所有成员都不活跃，策略推理应能正常处理
        pass  # 依赖具体实现，这里仅作标记
