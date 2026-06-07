"""
共识化过滤层（Reconciliation Layer）

功能：在策略推理结果出来之后，根据共识度进一步优化策略推荐。
它不修改上游的数学模型，而是在策略选择时加入“促进共同点”的考量。

核心思想：
- 当检测到低共识 + 低互惠时，优先推荐能促进共识的策略
- 当检测到高共识 + 高互惠时，保持原有策略不动
"""

import numpy as np
from typing import Dict, List, Any

from tender.strategy.base import StrategyDecision, InterventionStrategy, RiskLevel


class ReconciliationLayer:
    """
    共识化过滤层

    在策略决策之后运行，用于优化策略推荐。
    它只读取融合结果中的原始数据，并添加共识相关的过滤逻辑。
    """

    def __init__(self, config: Dict[str, Any]):
        # 共识相关阈值
        self.low_consensus_threshold = config.get("low_consensus_threshold", 0.3)
        self.low_reciprocity_threshold = config.get("low_reciprocity_threshold", 0.2)
        self.toxic_ring_detection = config.get("toxic_ring_detection", True)
        self.prioritize_reconciliation = config.get("prioritize_reconciliation", True)

    def compute_consensus_metrics(
        self, fusion_result
    ) -> Dict[str, float]:
        """
        从融合结果中计算共识相关指标

        这只是一个后处理计算，使用的是融合结果中已有的
        12维特征向量 + 融合图拓扑属性，不修改任何原始数据结构。
        """
        feat = fusion_result.feature_vector
        # 从12维向量中提取已有特征
        outlier_ratio = feat[1]         # 第2个特征：离群比例
        ring_exists = feat[2]          # 第3个特征：环标志
        valence = feat[3]              # 第4个特征：全局重心愉悦度

        # 计算共识分数（直接从情绪点云的离散程度推导）
        # 我们无法直接获得所有成员的情绪向量，但可以从已有指标反推
        # 离群比例高 → 共识低
        # 环存在 → 可能存在对抗性共识
        # 情绪重心极端 → 共识可能被少数人绑架
        consensus_score = 1.0 - outlier_ratio * 0.5
        if ring_exists:
            consensus_score *= 0.7  # 存在环意味着情绪循环，共识度打折扣

        # 避免极端值
        consensus_score = max(0.0, min(1.0, consensus_score))

        # 计算互惠指数（从因果网络图推导）
        # 我们可以从融合图中提取边属性，判断是否存在双向边
        fusion_graph = fusion_result.fusion_graph
        n_nodes = fusion_graph.number_of_nodes()
        n_edges = fusion_graph.number_of_edges()
        n_mutual_edges = 0

        for u, v in fusion_graph.edges():
            if fusion_graph.has_edge(v, u):
                n_mutual_edges += 0.5  # 每条双向边算0.5条互惠边

        reciprocity_index = (2 * n_mutual_edges) / max(n_edges, 1)

        # 计算最小共识方向偏差（从已有指标近似）
        # 这里使用情绪重心与离群比例的比值作为近似
        min_consensus_deviation = outlier_ratio / max(abs(valence + 1) / 2, 0.1)

        return {
            "consensus_score": consensus_score,
            "reciprocity_index": reciprocity_index,
            "min_consensus_deviation": min_consensus_deviation,
        }

    def detect_reconciliation_pattern(
        self, consensus_metrics: Dict[str, float]
    ) -> str:
        """
        根据共识指标检测需要调和的模式

        返回值:
            - "no_action": 无需调和
            - "low_consensus": 低共识
            - "low_reciprocity": 低互惠
            - "toxic_ring": 存在对抗性环
            - "combo": 复合模式（低共识+低互惠+环）
        """
        patterns = []

        if consensus_metrics["consensus_score"] < self.low_consensus_threshold:
            patterns.append("low_consensus")

        if consensus_metrics["reciprocity_index"] < self.low_reciprocity_threshold:
            patterns.append("low_reciprocity")

        if self.toxic_ring_detection:
            patterns.append("toxic_ring")

        if len(patterns) >= 2:
            return "combo"
        elif patterns:
            return patterns
        else:
            return "no_action"

    def refine_strategies(
        self,
        original_decision: StrategyDecision,
        fusion_result,
    ) -> StrategyDecision:
        """
        对原始策略决策进行共识化过滤和优化

        Args:
            original_decision: 策略引擎输出的原始决策
            fusion_result: 融合结果

        Returns:
            StrategyDecision: 优化后的策略决策（可能包含“调和”标签）
        """
        # 如果关闭了优先调和模式，则直接返回原决策
        if not self.prioritize_reconciliation:
            return original_decision

        # 计算共识指标
        consensus_metrics = self.compute_consensus_metrics(fusion_result)

        # 检测需要调和的模式
        pattern = self.detect_reconciliation_pattern(consensus_metrics)

        # 如果无需调和，直接返回原决策
        if pattern == "no_action":
            return original_decision

        # 根据模式对策略进行优化
        refined_strategies = []
        for strategy in original_decision.triggered_strategies:
            refined_strategy = self._apply_reconciliation(
                strategy, pattern, consensus_metrics
            )
            refined_strategies.append(refined_strategy)

        # 添加调和说明
        reasoning_parts = [original_decision.reasoning]
        reasoning_parts.append(
            f"[共识化过滤层] 检测到模式: {pattern}"
        )
        if pattern == "low_consensus":
            reasoning_parts.append(
                "共识度偏低，建议优先推荐能促进共同点的策略"
            )
        elif pattern == "low_reciprocity":
            reasoning_parts.append(
                "互惠指数偏低，建议优先推荐能促进双向沟通的策略"
            )
        elif pattern == "combo":
            reasoning_parts.append(
                "检测到复合风险：低共识 × 低互惠 × 情绪环，建议采取综合性调和策略"
            )

        # 返回更新后的决策（保留原始结构，仅优化策略列表和推理说明）
        return StrategyDecision(
            risk_level=original_decision.risk_level,
            risk_score=original_decision.risk_score,
            triggered_strategies=refined_strategies,
            fusion_result=original_decision.fusion_result,
            timestamp=original_decision.timestamp,
            reasoning="\n".join(reasoning_parts),
            requires_human=original_decision.requires_human,
        )

    def _apply_reconciliation(
        self,
        strategy: InterventionStrategy,
        pattern: str,
        consensus_metrics: Dict[str, float],
    ) -> InterventionStrategy:
        """
        根据调和模式对单个策略进行优化

        不改变策略的基本结构，只优化其描述和操作。
        """
        if pattern == "low_consensus":
            # 低共识模式：在原有操作基础上，加入“寻找共同点”建议
            new_actions = strategy.actions + ["寻找共同兴趣点"]
            new_description = strategy.description + "（建议同时寻找成员之间的共同点）"
            return InterventionStrategy(
                strategy_id=strategy.strategy_id,
                name=strategy.name,
                description=new_description,
                target_members=strategy.target_members,
                actions=new_actions,
                risk_level=strategy.risk_level,
                priority=strategy.priority,
            )

        elif pattern == "low_reciprocity":
            # 低互惠模式：在原有操作基础上，加入“促进双向交流”建议
            new_actions = strategy.actions + ["鼓励双向沟通"]
            new_description = strategy.description + "（建议优先促进成员之间的双向交流）"
            return InterventionStrategy(
                strategy_id=strategy.strategy_id,
                name=strategy.name,
                description=new_description,
                target_members=strategy.target_members,
                actions=new_actions,
                risk_level=strategy.risk_level,
                priority=strategy.priority,
            )

        elif pattern == "combo":
            # 复合模式：更综合的优化
            new_actions = strategy.actions + [
                "寻找共同点", "鼓励双向沟通", "插入中立事实"
            ]
            new_description = strategy.description + (
                "（复合风险模式：低共识 × 低互惠 × 情绪环，"
                "建议采取综合性调和策略：寻找共同点、促进双向沟通、插入中立事实）"
            )
            return InterventionStrategy(
                strategy_id=strategy.strategy_id,
                name=strategy.name,
                description=new_description,
                target_members=strategy.target_members,
                actions=new_actions,
                risk_level=strategy.risk_level,
                priority=strategy.priority,
            )

        # 默认：不修改
        return strategy

    def get_info(self) -> Dict[str, Any]:
        """获取过滤层信息"""
        return {
            "name": "ReconciliationLayer",
            "description": "共识化过滤层，用于优化策略推荐",
            "low_consensus_threshold": self.low_consensus_threshold,
            "low_reciprocity_threshold": self.low_reciprocity_threshold,
            "prioritize_reconciliation": self.prioritize_reconciliation,
        }
