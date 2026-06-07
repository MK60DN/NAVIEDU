"""
神经时序逻辑时空融合实现（可替换引擎）

该模块实现了基于神经时序逻辑（Neural Temporal Logic, NTL）的时空融合方案。
它是 Tender 框架的可替换融合引擎，适用于需要保持高可解释性的场景。

核心方法：
1. 将空间拓扑特征和时间因果特征编码为时序逻辑原子命题
2. 使用预定义的时序逻辑规则（如"如果离群比例高且因果密度低，则群体趋于分化"）
3. 通过可微的时序逻辑评估器计算每个规则的激活程度
4. 将规则激活程度加权融合为最终的融合特征向量

学术基础：
- 时序逻辑 (Pnueli, 1977): 通过"始终"、"最终"、"直到"等时序算子描述系统行为
- 可微时序逻辑 (Li et al., 2020): 将逻辑规则转化为可微分的数学表达式，
  使得逻辑推理可以被嵌入到深度学习模型中端到端训练
- 神经符号学习 (Garcez et al., 2019): 结合符号逻辑推理能力和神经网络表示能力

与 DCT-GNN 的区别：
- DCT-GNN：端到端黑盒学习，用 GNN 捕获依赖关系，可解释性较低
- 神经时序逻辑：使用预定义的可解释逻辑规则进行推理，
  每个规则都有明确的语义含义，群主可以理解"为什么推荐这个策略"
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import networkx as nx

from tender.fusion.base import BaseFusionModule, FusionResult


class NeuralTemporalLogicFusion(BaseFusionModule):
    """
    基于神经时序逻辑的时空融合模块

    将空间拓扑特征和时间因果特征适配到时序逻辑规则框架中，
    通过可微的规则评估器计算群体情绪态势，生成可解释的融合特征向量。

    Args:
        config: 配置字典，包含以下字段：
            - ntl_num_rules: 时序逻辑规则数量（默认12）
            - ntl_temperature: Gumbel-Softmax温度参数（默认1.0）
            - ntl_threshold: 规则激活阈值（默认0.5）
            - ntl_rule_regularizer: 规则正则化系数（默认0.01）
            - spatial_feature_dim: 空间拓扑特征维度（默认8）
            - temporal_feature_dim: 时间因果特征维度（默认8）
            - output_dim: 融合后特征维度（默认16）
            - forecast_horizon: 预测步数（默认1）
            - use_soft_logic: 是否使用模糊逻辑（默认True）
    """

    def __init__(self, config: Dict[str, Any]):
        # 神经时序逻辑核心参数
        self.ntl_num_rules = config.get("ntl_num_rules", 12)
        self.ntl_temperature = config.get("ntl_temperature", 1.0)
        self.ntl_threshold = config.get("ntl_threshold", 0.5)
        self.ntl_rule_regularizer = config.get("ntl_rule_regularizer", 0.01)
        self.use_soft_logic = config.get("use_soft_logic", True)

        # 特征维度配置
        self.spatial_feature_dim = config.get("spatial_feature_dim", 8)
        self.temporal_feature_dim = config.get("temporal_feature_dim", 8)
        self.output_dim = config.get("output_dim", 16)

        # 预测参数
        self.forecast_horizon = config.get("forecast_horizon", 1)

        # 初始化时序逻辑规则库
        self._rules = self._initialize_rules()

        # 规则权重（可学习参数）
        np.random.seed(42)
        self._rule_weights = np.ones(self.ntl_num_rules) / self.ntl_num_rules

    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """
        初始化预定义的时序逻辑规则库

        每条规则包含：
        - name: 规则名称（人类可读）
        - description: 规则描述的语义含义
        - formula: 逻辑公式的编码（包含参与的特征索引和逻辑操作符）
        - sensitivity: 规则对不同特征的敏感度向量

        默认包含 12 条规则，覆盖常见的群体情绪态势模式。

        Returns:
            List[Dict]: 规则列表
        """
        return [
            {
                "name": "high_division_risk",
                "description": "高分裂风险：离群比例高且聚类数量多",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 1, "feat_name": "outlier_ratio", "threshold": 0.3, "comparison": "gt"},
                        {"feat_idx": 0, "feat_name": "cluster_count", "threshold": 0.3, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3,
                                         0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "emotional_ring_present",
                "description": "情绪矛盾环存在：检测到环状结构且离群比例适中",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 2, "feat_name": "ring_exists", "threshold": 0.5, "comparison": "gt"},
                        {"feat_idx": 1, "feat_name": "outlier_ratio", "threshold": 0.1, "comparison": "lt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.2, 2.0, 0.0, 0.0, 0.0, 0.5, 0.8,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "high_cohesion_group",
                "description": "高凝聚力群体：聚类数量少、离群率低、互惠指数高",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 0, "feat_name": "cluster_count", "threshold": 0.2, "comparison": "lt"},
                        {"feat_idx": 1, "feat_name": "outlier_ratio", "threshold": 0.1, "comparison": "lt"},
                        {"feat_idx": 7, "feat_name": "reciprocity", "threshold": 0.5, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
                                         0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "super_spreader_dominated",
                "description": "超级传播者主导：存在超级传播者且因果密度高",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 9, "feat_name": "super_spreaders_ratio", "threshold": 0.2, "comparison": "gt"},
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.3, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         1.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "emotional_polarization",
                "description": "情绪极化：全局重心偏离原点且聚类数量少",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 3, "feat_name": "centroid_valence_abs", "threshold": 0.5, "comparison": "gt"},
                        {"feat_idx": 0, "feat_name": "cluster_count", "threshold": 0.2, "comparison": "lt"},
                    ],
                },
                "sensitivity": np.array([0.5, 0.0, 0.0, 1.5, 1.0, 0.5, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "emotional_fluctuation",
                "description": "情绪波动剧烈：因果密度高且存在环",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.4, "comparison": "gt"},
                        {"feat_idx": 2, "feat_name": "ring_exists", "threshold": 0.5, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "group_apathy",
                "description": "群体冷淡：全局重心接近零点、因果密度低",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 3, "feat_name": "centroid_valence_abs", "threshold": 0.2, "comparison": "lt"},
                        {"feat_idx": 4, "feat_name": "centroid_arousal_abs", "threshold": 0.2, "comparison": "lt"},
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.1, "comparison": "lt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0,
                                         0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "toxic_ring",
                "description": "对抗性情绪环：环状结构伴随高因果密度和低互惠指数",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 2, "feat_name": "ring_exists", "threshold": 0.5, "comparison": "gt"},
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.3, "comparison": "gt"},
                        {"feat_idx": 15, "feat_name": "reciprocity", "threshold": 0.3, "comparison": "lt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5]),
            },
            {
                "name": "information_bottleneck",
                "description": "信息瓶颈：少数成员集中大量因果出度",
                "formula": {
                    "operator": "or",
                    "inputs": [
                        {"feat_idx": 13, "feat_name": "max_out_degree", "threshold": 0.5, "comparison": "gt"},
                        {"feat_idx": 9, "feat_name": "super_spreaders_ratio", "threshold": 0.3, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0]),
            },
            {
                "name": "balanced_reciprocity",
                "description": "平衡互惠：互惠指数高且因果密度适中",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 15, "feat_name": "reciprocity", "threshold": 0.5, "comparison": "gt"},
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.2, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
            },
            {
                "name": "emotional_contagion",
                "description": "情绪传染：高因果密度且全局重心接近正情绪区域",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 8, "feat_name": "causal_density", "threshold": 0.3, "comparison": "gt"},
                        {"feat_idx": 3, "feat_name": "centroid_valence", "threshold": 0.3, "comparison": "gt"},
                    ],
                },
                "sensitivity": np.array([0.0, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0,
                                         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            },
            {
                "name": "group_fragmentation",
                "description": "群体碎片化：多聚类、高离群率、低互惠指数",
                "formula": {
                    "operator": "and",
                    "inputs": [
                        {"feat_idx": 0, "feat_name": "cluster_count", "threshold": 0.4, "comparison": "gt"},
                        {"feat_idx": 1, "feat_name": "outlier_ratio", "threshold": 0.3, "comparison": "gt"},
                        {"feat_idx": 15, "feat_name": "reciprocity", "threshold": 0.3, "comparison": "lt"},
                    ],
                },
                "sensitivity": np.array([1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            },
        ]

    def _evaluate_predicate(
        self, feature_value: float, threshold: float, comparison: str
    ) -> float:
        """
        评估一个逻辑谓词（原子命题）

        在模糊逻辑下，谓词的真值是一个 [0, 1] 之间的连续值。
        使用 Sigmoid 函数对比较操作进行平滑近似。

        Args:
            feature_value: 特征值
            threshold: 阈值
            comparison: 比较操作类型（"gt" 大于，"lt" 小于）

        Returns:
            float: 谓词真值 [0, 1]
        """
        if self.use_soft_logic:
            # 使用 Sigmoid 平滑近似
            steepness = 5.0 / self.ntl_temperature
            diff = feature_value - threshold

            if comparison == "gt":
                return 1.0 / (1.0 + np.exp(-steepness * diff))
            elif comparison == "lt":
                return 1.0 / (1.0 + np.exp(steepness * diff))
            else:
                return 0.0
        else:
            # 使用硬阈值（二值逻辑）
            if comparison == "gt":
                return 1.0 if feature_value > threshold else 0.0
            elif comparison == "lt":
                return 1.0 if feature_value < threshold else 0.0
            else:
                return 0.0

    def _evaluate_and(self, inputs: List[float]) -> float:
        """
        评估逻辑 AND 操作

        在模糊逻辑下，AND 使用乘积算子（product t-norm）。
        真值 = 各个子条件的真值的乘积。

        Args:
            inputs: 子条件的真值列表

        Returns:
            float: AND 操作的真值 [0, 1]
        """
        if not inputs:
            return 0.0

        if self.use_soft_logic:
            # 乘积 t-norm
            return np.prod(inputs)
        else:
            # 最小 t-norm（标准模糊逻辑）
            return np.min(inputs)

    def _evaluate_or(self, inputs: List[float]) -> float:
        """
        评估逻辑 OR 操作

        在模糊逻辑下，OR 使用概率和（probabilistic sum）。
        真值 = 1 - (1 - v1) * (1 - v2) * ...

        Args:
            inputs: 子条件的真值列表

        Returns:
            float: OR 操作的真值 [0, 1]
        """
        if not inputs:
            return 0.0

        if self.use_soft_logic:
            # 概率和 t-conorm
            result = 1.0
            for v in inputs:
                result *= (1.0 - v)
            return 1.0 - result
        else:
            # 最大 t-conorm（标准模糊逻辑）
            return np.max(inputs)

    def _evaluate_rule(
        self, rule: Dict[str, Any], fused_features: np.ndarray
    ) -> float:
        """
        评估单条时序逻辑规则

        递归地评估规则公式树中的每个节点。

        Args:
            rule: 规则字典
            fused_features: 16 维融合特征向量

        Returns:
            float: 规则的激活程度 [0, 1]
        """
        formula = rule["formula"]

        return self._evaluate_formula(formula, fused_features)

    def _evaluate_formula(
        self, formula: Dict, features: np.ndarray
    ) -> float:
        """
        递归评估逻辑公式

        Args:
            formula: 公式节点
            features: 特征向量

        Returns:
            float: 公式真值 [0, 1]
        """
        operator = formula.get("operator")

        if operator == "and":
            # 递归评估所有子条件并求 AND
            sub_results = [
                self._evaluate_formula(input_node, features)
                for input_node in formula["inputs"]
            ]
            return self._evaluate_and(sub_results)

        elif operator == "or":
            # 递归评估所有子条件并求 OR
            sub_results = [
                self._evaluate_formula(input_node, features)
                for input_node in formula["inputs"]
            ]
            return self._evaluate_or(sub_results)

        elif operator == "not":
            # NOT 操作（只有一个子条件）
            sub_result = self._evaluate_formula(formula["inputs"], features)
            return 1.0 - sub_result

        else:
            # 原子命题（谓词）
            feat_idx = formula.get("feat_idx", 0)
            if feat_idx < len(features):
                feature_value = features[feat_idx]
            else:
                feature_value = 0.0

            threshold = formula.get("threshold", 0.5)
            comparison = formula.get("comparison", "gt")

            return self._evaluate_predicate(feature_value, threshold, comparison)

    def _extract_spatial_features(
        self, topology_result: Any
    ) -> np.ndarray:
        """
        从拓扑分析结果提取空间特征（与 DCT-GNN 保持一致）

        Args:
            topology_result: 拓扑分析结果

        Returns:
            np.ndarray: 8 维空间特征向量
        """
        features = np.zeros(self.spatial_feature_dim)

        features[0] = topology_result.cluster_count / 10.0
        features[1] = topology_result.outlier_ratio
        features[2] = 1.0 if topology_result.ring_exists else 0.0

        centroid = topology_result.centroid
        features[3] = centroid[0] if len(centroid) > 0 else 0.0  # valence
        features[4] = centroid[1] if len(centroid) > 1 else 0.0  # arousal
        features[5] = centroid[2] if len(centroid) > 2 else 0.0  # focus

        features[6] = len(topology_result.h0_barcodes) / 20.0
        features[7] = len(topology_result.h1_barcodes) / 10.0

        features = np.clip(features, -1.0, 1.0)
        features[3:6] = np.abs(features[3:6])  # 使用绝对值表示偏离程度

        return features

    def _extract_temporal_features(
        self, causal_result: Any
    ) -> np.ndarray:
        """
        从因果分析结果提取时间特征（与 DCT-GNN 保持一致）

        Args:
            causal_result: 因果分析结果

        Returns:
            np.ndarray: 8 维时间特征向量
        """
        features = np.zeros(self.temporal_feature_dim)
        n_members = len(causal_result.out_degrees)

        if n_members == 0:
            return features

        features[0] = causal_result.causal_density
        features[1] = len(causal_result.super_spreaders) / max(n_members, 1)

        n_edges = len(causal_result.edges)
        features[2] = n_edges / max(n_members * (n_members - 1), 1)

        out_deg_values = list(causal_result.out_degrees.values())
        in_deg_values = list(causal_result.in_degrees.values())

        features[3] = np.mean(out_deg_values) / max(n_members, 1)
        features[4] = np.mean(in_deg_values) / max(n_members, 1)
        features[5] = max(out_deg_values) / max(n_members, 1) if out_deg_values else 0.0
        features[6] = max(in_deg_values) / max(n_members, 1) if in_deg_values else 0.0

        if n_edges > 0:
            graph = causal_result.causal_graph
            n_mutual = 0
            for u, v in graph.edges():
                if graph.has_edge(v, u):
                    n_mutual += 0.5
            features[7] = (2 * n_mutual) / n_edges
        else:
            features[7] = 0.0

        features = np.clip(features, 0.0, 1.0)

        return features

    def _compute_rule_based_forecast(
        self, rule_activations: np.ndarray
    ) -> float:
        """
        基于规则激活程度计算预测值

        如果存在高分裂风险或群体碎片化规则激活，预测值较低；
        如果高凝聚力群体规则激活，预测值较高。

        Args:
            rule_activations: 规则激活程度向量

        Returns:
            float: 预测值（表示下一窗口的群体情绪健康指数）
        """
        # 规则索引
        high_division_idx = 0     # 高分裂风险
        toxic_ring_idx = 7        # 对抗性情绪环
        fragmentation_idx = 11    # 群体碎片化
        cohesion_idx = 2          # 高凝聚力群体
        reciprocity_idx = 9       # 平衡互惠

        # 负面信号
        negative_signals = np.array([
            rule_activations[high_division_idx] if high_division_idx < len(rule_activations) else 0.0,
            rule_activations[toxic_ring_idx] if toxic_ring_idx < len(rule_activations) else 0.0,
            rule_activations[fragmentation_idx] if fragmentation_idx < len(rule_activations) else 0.0,
        ])
        negative_score = np.mean(negative_signals)

        # 正面信号
        positive_signals = np.array([
            rule_activations[cohesion_idx] if cohesion_idx < len(rule_activations) else 0.0,
            rule_activations[reciprocity_idx] if reciprocity_idx < len(rule_activations) else 0.0,
        ])
        positive_score = np.mean(positive_signals) if len(positive_signals) > 0 else 0.0

        # 健康指数 = 正面信号 - 负面信号 + 0.5（基准值）
        health_score = 0.5 + positive_score - negative_score
        health_score = np.clip(health_score, 0.0, 1.0)

        return float(health_score)

    def fuse(
        self,
        topology_result: Any,
        causal_result: Any,
        time_series_data: Dict[str, List[np.ndarray]],
        member_ids: List[str],
    ) -> FusionResult:
        """
        执行基于神经时序逻辑的时空融合

        流程：
        1. 提取空间特征和时间特征
        2. 拼接为 16 维融合特征向量
        3. 将特征向量映射到时序逻辑规则空间中评估
        4. 计算每条规则的激活程度
        5. 生成规则加权融合特征
        6. 基于规则激活进行预测

        Args:
            topology_result: 拓扑分析结果
            causal_result: 因果分析结果
            time_series_data: 时间序列数据
            member_ids: 成员ID列表

        Returns:
            FusionResult: 时空融合结果
        """
        # 1. 提取空间和时间特征
        spatial_features = self._extract_spatial_features(topology_result)
        temporal_features = self._extract_temporal_features(causal_result)

        # 2. 拼接为 16 维融合特征向量
        fused_features = np.concatenate([spatial_features, temporal_features])

        # 3. 评估每条规则的激活程度
        rule_activations = []
        rule_activation_details = []

        for rule in self._rules:
            activation = self._evaluate_rule(rule, fused_features)
            rule_activations.append(activation)
            rule_activation_details.append({
                "rule_name": rule["name"],
                "description": rule["description"],
                "activation": float(activation),
                "formula": rule["formula"],
            })

        rule_activations = np.array(rule_activations)

        # 4. 构建规则加权融合特征
        # 用规则激活向量对原始特征进行软注意力加权
        weighted_features = fused_features.copy()

        # 对每条规则，根据其敏感度向量对特征进行调节
        for i, activation in enumerate(rule_activations):
            sensitivity = self._rules[i]["sensitivity"]
            if i < len(self._rules):
                weighted_features = weighted_features + activation * sensitivity

        # 5. 构建动态因果拓扑图（用于可视化）
        fusion_graph = nx.Graph()
        for mid in member_ids:
            cluster_label = topology_result.cluster_labels.get(mid, -1)
            fusion_graph.add_node(
                mid,
                cluster_label=int(cluster_label),
                is_outlier=mid in topology_result.outlier_members,
                rule_activations=rule_activations.tolist(),
            )

        for edge in causal_result.edges:
            if edge.source in member_ids and edge.target in member_ids:
                fusion_graph.add_edge(
                    edge.source,
                    edge.target,
                    strength=edge.strength,
                    lag=edge.lag,
                )

        # 6. 计算基于规则的预测
        forecast_score = self._compute_rule_based_forecast(rule_activations)

        # 7. 构建节点特征矩阵（标准化为 6 维）
        n_members = len(member_ids)
        node_features = np.zeros((n_members, 6))
        for i, mid in enumerate(member_ids):
            node_features[i, 0] = topology_result.cluster_labels.get(mid, -1) + 1
            node_features[i, 1] = 1.0 if mid in topology_result.outlier_members else 0.0
            node_features[i, 2] = float(rule_activations[0](@ref)  # 高分裂风险模型
            node_features[i, 3] = float(rule_activations[2](@ref)  # 高凝聚力模型
            node_features[i, 4] = float(rule_activations[7](@ref)  # 对抗环模型
            node_features[i, 5] = float(rule_activations[1](@ref)  # 情绪环模型

        # 8. 构建邻接矩阵
        n_members = len(member_ids)
        adjacency_matrix = np.zeros((n_members, n_members))
        for edge in causal_result.edges:
            if edge.source in member_ids and edge.target in member_ids:
                i = member_ids.index(edge.source)
                j = member_ids.index(edge.target)
                adjacency_matrix[i, j] = edge.strength

        forecast = np.array([forecast_score, 0.0, 0.0])

        return FusionResult(
            feature_vector=weighted_features,
            fusion_graph=fusion_graph,
            forecast=forecast,
            spatial_features=spatial_features,
            temporal_features=temporal_features,
            node_features=node_features,
            adjacency_matrix=adjacency_matrix,
            metadata={
                "fuser": "NeuralTemporalLogicFusion",
                "ntl_num_rules": self.ntl_num_rules,
                "ntl_temperature": self.ntl_temperature,
                "ntl_threshold": self.ntl_threshold,
                "use_soft_logic": self.use_soft_logic,
                "rule_activations": rule_activations.tolist(),
                "rule_activation_details": rule_activation_details,
                "n_members": len(member_ids),
                "n_edges": len(causal_result.edges),
                "n_clusters": topology_result.cluster_count,
                "ring_exists": topology_result.ring_exists,
            },
        )

    def get_info(self) -> Dict[str, Any]:
        """获取当前融合模块的信息"""
        return {
            "name": "NeuralTemporalLogicFusion",
            "description": "基于神经时序逻辑的时空融合模块，可解释性强",
            "ntl_num_rules": self.ntl_num_rules,
            "ntl_temperature": self.ntl_temperature,
            "ntl_threshold": self.ntl_threshold,
            "use_soft_logic": self.use_soft_logic,
            "spatial_feature_dim": self.spatial_feature_dim,
            "temporal_feature_dim": self.temporal_feature_dim,
            "output_dim": self.output_dim,
            "rules": [
                {
                    "name": rule["name"],
                    "description": rule["description"],
                }
                for rule in self._rules
            ],
        }
