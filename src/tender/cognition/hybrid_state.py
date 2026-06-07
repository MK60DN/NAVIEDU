""" 混合认知状态分析引擎——认知状态分析模块

该模块实现了融合知识图谱与行为模式的混合认知状态分析策略（Strategy 3）。
它结合了 KnowledgeStateAnalyzer 的知识结构化分析能力
和 BehaviorStateAnalyzer 的行为模式分析能力，
通过一种加权融合机制，输出更全面、更鲁棒的认知状态判断。

核心思想：
- 知识图谱分析提供了"学什么"的结构化信息（认知的内容维度）
- 行为分析提供了"学得怎么样"的动态信息（认知的过程维度）
- 两者结合能够更全面地描绘成员的认知画像
- 当一方数据不足或置信度低时，另一方可以弥补

工作流程：
1. 并行运行知识图谱分析引擎和行为分析引擎
2. 分别获取两者的认知状态输出
3. 根据两个引擎的置信度，动态计算融合权重
4. 对关键指标（认知负荷、理解水平等）进行加权融合
5. 对离散指标（认知阶段、参与类型等）进行置信度投票

适用场景：
- 既有预设知识图谱，又有丰富行为数据的场景
- 需要高精度认知分析的场景（如个性化教学推荐）
- 作为知识图谱分析和行为分析的上游融合层

学术基础：
- 多模态融合理论 (Baltrušaitis et al., 2019)
- 知识追踪与行为分析的互补性 (Koedinger et al., 2012)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from tender.cognition.base import (
    BaseCognitionAnalyzer,
    CognitionState,
    BehaviorProfile,
    KnowledgeGraphConfig,
    CognitivePhase,
    EngagementType,
)
from tender.cognition.knowledge_state import KnowledgeStateAnalyzer
from tender.cognition.behavior_state import BehaviorStateAnalyzer


class HybridStateAnalyzer(BaseCognitionAnalyzer):
    """混合认知状态分析引擎

    融合知识图谱分析和行为模式分析的混合引擎。
    内部维护了两个子引擎，运行时并行计算并融合结果。

    Args:
        config: 配置字典，包含以下字段：
            - feature_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 16）
            - use_knowledge_graph: 是否使用知识图谱（默认 True）
            - knowledge_graph_path: 知识图谱配置文件路径（可选）
            - kg_embedding_method: 知识图谱嵌入方法（默认 "node2vec"）
            - window_size_seconds: 行为分析时间窗口（默认 300）
            - min_messages_for_analysis: 最少消息数量（默认 3）
            - fusion_method: 融合方法（默认 "confidence_weighted"）
                - "confidence_weighted": 按置信度加权
                - "average": 简单平均
                - "max_confidence": 取置信度更高的引擎结果
            - knowledge_weight: 知识图谱分析的默认权重（默认 0.5）
            - behavior_weight: 行为分析的默认权重（默认 0.5）
            - aggregation_strategy: 群体聚合策略（默认 "weighted"）
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化混合认知状态分析引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.feature_dim = config.get("feature_dim", 16)
        self.output_dim = config.get("output_dim", 16)

        # 融合参数
        self.fusion_method = config.get("fusion_method", "confidence_weighted")
        self.knowledge_weight = config.get("knowledge_weight", 0.5)
        self.behavior_weight = config.get("behavior_weight", 0.5)

        # 聚合参数
        self.aggregation_strategy = config.get("aggregation_strategy", "weighted")

        # 创建子引擎
        # 知识图谱引擎（如果配置了使用知识图谱）
        self._kg_analyzer = KnowledgeStateAnalyzer(config)

        # 行为分析引擎（始终存在，作为基础）
        self._behavior_analyzer = BehaviorStateAnalyzer(config)

        # 记录初始化信息
        self._init_info = (
            f"HybridStateAnalyzer initialized with "
            f"fusion={self.fusion_method}, "
            f"kg_weight={self.knowledge_weight}, "
            f"behavior_weight={self.behavior_weight}, "
            f"kg_analyzer={'enabled' if config.get('use_knowledge_graph', True) else 'disabled'}, "
            f"behavior_analyzer=enabled"
        )

    def configure_kg(self, kg_config: KnowledgeGraphConfig) -> None:
        """配置知识图谱子引擎

        Args:
            kg_config: 知识图谱配置
        """
        self._kg_analyzer.configure(kg_config)

    def analyze(
        self,
        member_messages: Dict[str, List[Dict[str, Any]]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None,
    ) -> Dict[str, CognitionState]:
        """分析所有成员的认知状态（混合分析）

        Args:
            member_messages: 成员消息字典
            knowledge_graph_config: 可选的知识图谱配置
            behavior_profiles: 成员行为档案（可选）

        Returns:
            Dict[str, CognitionState]: 成员 ID 到认知状态的映射
        """
        # 验证输入
        self.validate_inputs(member_messages)

        # 步骤1：并行运行两个子引擎
        kg_states: Dict[str, CognitionState] = {}
        behavior_states: Dict[str, CognitionState] = {}

        try:
            kg_states = self._kg_analyzer.analyze(
                member_messages=member_messages,
                knowledge_graph_config=knowledge_graph_config,
                behavior_profiles=behavior_profiles,
            )
        except Exception as e:
            print(f"警告：知识图谱分析引擎运行失败: {e}。将仅使用行为分析结果。")

        try:
            behavior_states = self._behavior_analyzer.analyze(
                member_messages=member_messages,
                behavior_profiles=behavior_profiles,
            )
        except Exception as e:
            print(f"警告：行为分析引擎运行失败: {e}。将仅使用知识图谱分析结果。")

        # 步骤2：融合两个引擎的结果
        member_states = {}
        all_member_ids = set(member_messages.keys())

        for member_id in all_member_ids:
            kg_state = kg_states.get(member_id)
            behavior_state = behavior_states.get(member_id)

            if kg_state is not None and behavior_state is not None:
                # 两者都有结果，进行融合
                fused_state = self._fuse_states(kg_state, behavior_state, member_id)
            elif kg_state is not None:
                # 仅知识图谱结果可用
                fused_state = kg_state
                fused_state.source_engine = "hybrid_kg_fallback"
            elif behavior_state is not None:
                # 仅行为分析结果可用
                fused_state = behavior_state
                fused_state.source_engine = "hybrid_behavior_fallback"
            else:
                # 两者都失败，创建默认状态
                fused_state = CognitionState(
                    member_id=member_id,
                    source_engine="hybrid_failed",
                    metadata={"warning": "知识图谱和行为分析均失败"},
                )

            member_states[member_id] = fused_state

        # 步骤3：计算群体状态（如果多个成员）
        if len(member_states) > 1:
            group_state = self.compute_group_state(member_states)
            member_states["__group__"] = group_state

        return member_states

    def analyze_single(
        self,
        member_id: str,
        messages: List[Dict[str, Any]],
        knowledge_graph_config: Optional[KnowledgeGraphConfig] = None,
        behavior_profile: Optional[BehaviorProfile] = None,
    ) -> CognitionState:
        """分析单个成员的认知状态（混合分析）

        Args:
            member_id: 成员唯一标识
            messages: 该成员的消息列表
            knowledge_graph_config: 可选的知识图谱配置
            behavior_profile: 该成员的行为档案（可选）

        Returns:
            CognitionState: 融合后的认知状态
        """
        # 并行运行两个子引擎的单成员分析
        kg_state = None
        behavior_state = None

        try:
            kg_state = self._kg_analyzer.analyze_single(
                member_id=member_id,
                messages=messages,
                behavior_profile=behavior_profile,
            )
        except Exception as e:
            print(f"警告：知识图谱单成员分析失败: {e}")

        try:
            behavior_state = self._behavior_analyzer.analyze_single(
                member_id=member_id,
                messages=messages,
                behavior_profile=behavior_profile,
            )
        except Exception as e:
            print(f"警告：行为分析单成员分析失败: {e}")

        # 融合结果
        if kg_state is not None and behavior_state is not None:
            return self._fuse_states(kg_state, behavior_state, member_id)
        elif kg_state is not None:
            return kg_state
        elif behavior_state is not None:
            return behavior_state
        else:
            return CognitionState(
                member_id=member_id,
                source_engine="hybrid_failed",
                metadata={"warning": "知识图谱和行为分析均失败"},
            )

    def compute_group_state(
        self,
        member_states: Dict[str, CognitionState],
    ) -> CognitionState:
        """从成员状态计算群体认知状态

        使用配置中设定的聚合策略。

        Args:
            member_states: 成员状态字典

        Returns:
            CognitionState: 群体的认知状态
        """
        return self._aggregate_states(member_states, self.aggregation_strategy)

    def _fuse_states(
        self,
        kg_state: CognitionState,
        behavior_state: CognitionState,
        member_id: str,
    ) -> CognitionState:
        """融合知识图谱和行为分析的结果

        根据配置的融合方法，对两个引擎的输出进行融合。

        Args:
            kg_state: 知识图谱分析的状态结果
            behavior_state: 行为分析的状态结果
            member_id: 成员唯一标识

        Returns:
            CognitionState: 融合后的认知状态
        """
        # 计算融合权重
        if self.fusion_method == "average":
            # 简单平均
            kg_weight = 0.5
            behavior_weight = 0.5

        elif self.fusion_method == "max_confidence":
            # 取置信度更高的引擎
            if kg_state.phase_confidence >= behavior_state.phase_confidence:
                kg_weight = 1.0
                behavior_weight = 0.0
            else:
                kg_weight = 0.0
                behavior_weight = 1.0

        else:  # "confidence_weighted" (默认)
            # 按置信度加权
            total_confidence = kg_state.phase_confidence + behavior_state.phase_confidence
            if total_confidence > 0:
                kg_weight = kg_state.phase_confidence / total_confidence
                behavior_weight = behavior_state.phase_confidence / total_confidence
            else:
                # 两者置信度都为零，使用预设权重
                kg_weight = self.knowledge_weight
                behavior_weight = self.behavior_weight

        # 融合连续指标
        cognitive_load = (
            kg_weight * kg_state.cognitive_load
            + behavior_weight * behavior_state.cognitive_load
        )
        understanding_level = (
            kg_weight * kg_state.understanding_level
            + behavior_weight * behavior_state.understanding_level
        )
        attention_score = (
            kg_weight * kg_state.attention_score
            + behavior_weight * behavior_state.attention_score
        )
        confusion_level = (
            kg_weight * kg_state.confusion_level
            + behavior_weight * behavior_state.confusion_level
        )
        cognitive_flexibility = (
            kg_weight * kg_state.cognitive_flexibility
            + behavior_weight * behavior_state.cognitive_flexibility
        )

        # 融合离散指标（认知阶段、参与类型）
        # 使用加权投票
        phase = self._weighted_phase_vote(
            kg_state.cognitive_phase, kg_weight,
            behavior_state.cognitive_phase, behavior_weight,
        )
        engagement = self._weighted_engagement_vote(
            kg_state.engagement_type, kg_weight,
            behavior_state.engagement_type, behavior_weight,
        )

        # 计算融合后的置信度
        fused_confidence = kg_state.phase_confidence * kg_weight + behavior_state.phase_confidence * behavior_weight

        # 合并知识点列表
        all_nodes = list(set(kg_state.knowledge_nodes + behavior_state.knowledge_nodes))

        # 合并嵌入向量
        fused_embedding = self._fuse_embeddings(
            kg_state.raw_embedding, kg_weight,
            behavior_state.raw_embedding, behavior_weight,
        )

        # 构建融合后的认知状态
        fused_state = CognitionState(
            member_id=member_id,
            cognitive_load=float(np.clip(cognitive_load, 0.0, 1.0)),
            understanding_level=float(np.clip(understanding_level, 0.0, 1.0)),
            cognitive_phase=phase,
            engagement_type=engagement,
            attention_score=float(np.clip(attention_score, 0.0, 1.0)),
            confusion_level=float(np.clip(confusion_level, 0.0, 1.0)),
            cognitive_flexibility=float(np.clip(cognitive_flexibility, 0.0, 1.0)),
            phase_confidence=float(np.clip(fused_confidence, 0.0, 1.0)),
            source_engine="hybrid_state",
            timestamp=max(kg_state.timestamp, behavior_state.timestamp),
            knowledge_nodes=all_nodes,
            raw_embedding=fused_embedding,
            metadata={
                "kg_weight": float(kg_weight),
                "behavior_weight": float(behavior_weight),
                "fusion_method": self.fusion_method,
                "kg_confidence": kg_state.phase_confidence,
                "behavior_confidence": behavior_state.phase_confidence,
                "kg_cognitive_load": kg_state.cognitive_load,
                "behavior_cognitive_load": behavior_state.cognitive_load,
                "kg_understanding": kg_state.understanding_level,
                "behavior_understanding": behavior_state.understanding_level,
                "kg_phase": kg_state.cognitive_phase.value,
                "behavior_phase": behavior_state.cognitive_phase.value,
            },
        )

        return fused_state

    def _weighted_phase_vote(
        self,
        phase_kg: CognitivePhase,
        weight_kg: float,
        phase_behavior: CognitivePhase,
        weight_behavior: float,
    ) -> CognitivePhase:
        """加权投票选择认知阶段

        Args:
            phase_kg: 知识图谱分析的认知阶段
            weight_kg: 知识图谱的权重
            phase_behavior: 行为分析的认知阶段
            weight_behavior: 行为分析的权重

        Returns:
            CognitivePhase: 投票选出的认知阶段
        """
        # 如果某个引擎的权重显著占优，直接采用该引擎的结果
        if weight_kg > 0.7:
            return phase_kg
        elif weight_behavior > 0.7:
            return phase_behavior

        # 如果两者一致，直接返回
        if phase_kg == phase_behavior:
            return phase_kg

        # 如果不一致，采用置信度更高的（已经包含在权重中）
        return phase_kg if weight_kg > weight_behavior else phase_behavior

    def _weighted_engagement_vote(
        self,
        engagement_kg: EngagementType,
        weight_kg: float,
        engagement_behavior: EngagementType,
        weight_behavior: float,
    ) -> EngagementType:
        """加权投票选择参与类型

        Args:
            engagement_kg: 知识图谱分析的参与类型
            weight_kg: 知识图谱的权重
            engagement_behavior: 行为分析的参与类型
            weight_behavior: 行为分析的权重

        Returns:
            EngagementType: 投票选出的参与类型
        """
        # 行为分析得出的参与类型通常更可靠（因为它直接基于行为数据）
        # 所以当权重相近时，偏重行为分析的结果
        adjusted_kg_weight = weight_kg * 0.7  # 降低知识图谱对参与类型的信任
        adjusted_behavior_weight = weight_behavior

        if adjusted_behavior_weight >= adjusted_kg_weight:
            return engagement_behavior
        else:
            return engagement_kg

    def _fuse_embeddings(
        self,
        kg_embedding: Optional[np.ndarray],
        kg_weight: float,
        behavior_embedding: Optional[np.ndarray],
        behavior_weight: float,
    ) -> Optional[np.ndarray]:
        """融合两个引擎的嵌入向量

        Args:
            kg_embedding: 知识图谱分析的嵌入向量
            kg_weight: 知识图谱的权重
            behavior_embedding: 行为分析的嵌入向量
            behavior_weight: 行为分析的权重

        Returns:
            Optional[np.ndarray]: 融合后的嵌入向量
        """
        if kg_embedding is None and behavior_embedding is None:
            return None

        if kg_embedding is None:
            return behavior_embedding

        if behavior_embedding is None:
            return kg_embedding

        # 确保维度一致
        max_dim = max(len(kg_embedding), len(behavior_embedding))
        kg_padded = np.pad(kg_embedding, (0, max_dim - len(kg_embedding)))
        behavior_padded = np.pad(behavior_embedding, (0, max_dim - len(behavior_embedding)))

        # 加权融合
        fused = kg_weight * kg_padded + behavior_weight * behavior_padded

        return fused
