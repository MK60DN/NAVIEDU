"""
 加权融合引擎——情绪-认知协同模块（已更新以适配认知模块）

 该模块实现了基于加权融合的协同策略（Strategy 1）。
 这是最基础的融合策略，通过线性加权组合情绪特征和认知特征。

 工作流程：
 1. 获取情绪特征向量 E 和认知特征向量 K
 2. 使用预设权重 α 和 β 计算加权和: F = α * E + β * K
 3. 计算协同度评分: 情绪和认知向量的夹角余弦
 4. 根据协同度分类协同模式

 更新说明：
 - 现在支持通过 member_pairs 接收外部认知模块的 CognitionState 对象
 - 在 metatdata 中传递认知模块的详细状态信息（如认知负荷、理解水平等）
 - 利用认知模块的 phase_confidence 调整融合权重

 Args:
     config: 配置字典，包含以下字段：
         - emotion_dim: 情绪特征维度（默认 16）
         - cognition_dim: 认知特征维度（默认 16）
         - output_dim: 输出维度（默认 32）
         - emotion_weight: 情绪权重（默认 0.5）
         - cognition_weight: 认知权重（默认 0.5）
         - cognition_source: 认知状态来源（默认 "internal"）
         - enable_cognition_metadata: 是否将 CognitionState 对象传播到结果元数据（默认 True）
 """

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from tender.synergy.base import (
    BaseSynergyEngine,
    SynergyResult,
    DominantDimension,
    SynergyMode,
    EmotionCognitionPair,
)


class WeightedFusionEngine(BaseSynergyEngine):
    """加权融合引擎（已更新以适配认知模块）

    该引擎是最基础的协同策略，通过线性加权组合情绪和认知特征。
    现在支持两种认知特征来源：
    1. internal（内部）：直接使用传入的特征向量
    2. external（外部）：通过 member_pairs 获取 CognitionState 对象，
       并将其信息编码到融合结果中

    Args:
        config: 配置字典，参考 config.py 中的 DEFAULT_CONFIG
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化加权融合引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.emotion_dim = config.get("emotion_dim", 16)
        self.cognition_dim = config.get("cognition_dim", 16)
        self.output_dim = config.get("output_dim", 32)

        # 融合权重参数
        self.emotion_weight = config.get("emotion_weight", 0.5)
        self.cognition_weight = config.get("cognition_weight", 0.5)

        # 认知模块对接参数（新增）
        self.cognition_source = config.get("cognition_source", "internal")
        self.enable_cognition_metadata = config.get("enable_cognition_metadata", True)

        # 计算和存储特征维度信息，用于运行时检查
        self._feature_dims = {
            "emotion": self.emotion_dim,
            "cognition": self.cognition_dim,
            "combined": self.output_dim,
        }

        # 记录初始化信息
        self._init_info = (
            f"WeightedFusionEngine initialized with "
            f"α={self.emotion_weight:.2f}, β={self.cognition_weight:.2f}, "
            f"dim_E={self.emotion_dim}, dim_K={self.cognition_dim}, "
            f"dim_out={self.output_dim}, "
            f"cognition_source={self.cognition_source}"
        )

    def fuse(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> SynergyResult:
        """融合情绪和认知特征（已更新）

        执行加权融合的核心逻辑：
        1. 验证输入的有效性
        2. 如果维度不匹配，进行维度对齐（截断或填充）
        3. 计算加权特征
        4. 计算协同度评分（现在考虑认知模块提供的置信度）
        5. 分类协同模式（现在利用认知模块的 phase 信息）
        6. 打包为 SynergyResult（在 metadata 中包含认知模块的状态信息）

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            member_pairs: 可选的成员级配对数据
                如果包含来自 tender.cognition 的 CognitionState 对象，
                其元数据将被提取并传递到结果中

        Returns:
            SynergyResult: 包含融合特征和协同分析的结果
        """
        # 步骤1：输入验证
        self.validate_inputs(emotion_features, cognition_features)

        # 步骤2：维度对齐
        emotion_aligned, cognition_aligned = self._align_dimensions(
            emotion_features, cognition_features
        )

        # === 新增：计算认知模块提供的动态权重调整因子 ===
        # 如果外部认知模块可用，利用其 phase_confidence 调整融合权重
        cognition_adjustment = self._compute_cognition_weight_adjustment(member_pairs)

        # 调整后的融合权重
        adjusted_emotion_weight = self.emotion_weight * (1.0 - cognition_adjustment)
        adjusted_cognition_weight = self.cognition_weight * (1.0 + cognition_adjustment)

        # 归一化
        total_weight = adjusted_emotion_weight + adjusted_cognition_weight
        if total_weight > 0:
            adjusted_emotion_weight /= total_weight
            adjusted_cognition_weight /= total_weight
        else:
            adjusted_emotion_weight = self.emotion_weight
            adjusted_cognition_weight = self.cognition_weight

        # 步骤3：计算加权特征（使用调整后的权重）
        combined_feature = (
            adjusted_emotion_weight * emotion_aligned
            + adjusted_cognition_weight * cognition_aligned
        )

        # 步骤4：计算协同度评分（现在包含认知模块信息）
        base_synergy_score = self.compute_synergy_score(
            emotion_aligned, cognition_aligned
        )

        # === 新增：根据认知模块的置信度调整协同度评分 ===
        synergy_score = self._adjust_synergy_with_cognition(
            base_synergy_score, member_pairs
        )

        # 步骤5：分类协同模式（现在利用认知阶段信息）
        dominant, mode = self._classify_synergy_mode_with_cognition(
            emotion_aligned, cognition_aligned, synergy_score, member_pairs
        )

        # 步骤6：生成建议提示（利用认知模块的详细信息）
        recommendation_hint = self._generate_hint_with_cognition(
            dominant, mode, synergy_score, member_pairs
        )

        # 步骤7：提取认知模块的元数据（新增）
        cognition_metadata = {}
        if self.enable_cognition_metadata:
            cognition_metadata = self._extract_cognition_metadata(member_pairs)

        # 步骤8：打包结果
        result = SynergyResult(
            combined_feature=combined_feature,
            synergy_score=synergy_score,
            dominant_dimension=dominant,
            synergy_mode=mode,
            emotion_feature=emotion_features,
            cognition_feature=cognition_features,
            recommendation_hint=recommendation_hint,
            metadata={
                "method": "weighted_fusion",
                "emotion_weight": adjusted_emotion_weight,
                "cognition_weight": adjusted_cognition_weight,
                "default_emotion_weight": self.emotion_weight,
                "default_cognition_weight": self.cognition_weight,
                "cognition_adjustment_factor": cognition_adjustment,
                # === 新增的认知模块元数据 ===
                "cognition_source": self.cognition_source,
                **cognition_metadata,  # 展开外部认知元数据
            },
        )

        return result

    def _compute_cognition_weight_adjustment(
        self,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> float:
        """计算认知模块带来的权重调整因子（新增）

        如果外部认知模块提供了高置信度的认知状态分析，
        则适当提高认知特征的融合权重。

        Returns:
            float: 权重调整因子，范围 [-0.3, 0.3]
                正值表示提高认知权重，负值表示降低认知权重
        """
        if not member_pairs or self.cognition_source != "external":
            return 0.0

        # 收集所有成员的认知置信度
        confidences = []
        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'phase_confidence'):
                confidences.append(cs.phase_confidence)

        if not confidences:
            return 0.0

        # 平均置信度
        avg_confidence = np.mean(confidences)

        # 映射到 [-0.3, 0.3]：
        # 高置信度 (>0.7) -> 提高认知权重
        # 低置信度 (<0.3) -> 降低认知权重（回到默认权重的均衡）
        adjustment = (avg_confidence - 0.5) * 0.6

        return float(np.clip(adjustment, -0.3, 0.3))

    def _adjust_synergy_with_cognition(
        self,
        base_score: float,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> float:
        """根据认知模块的信息调整协同度评分（新增）

        利用认知模块的困惑水平和认知灵活性来微调协同度评分。

        Args:
            base_score: 基础协同度评分（余弦相似度）
            member_pairs: 成员级配对数据

        Returns:
            float: 调整后的协同度评分
        """
        if not member_pairs or self.cognition_source != "external":
            return base_score

        # 收集认知模块的状态信息
        confusion_levels = []
        cognitive_flexibilities = []
        understanding_levels = []

        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'confusion_level'):
                confusion_levels.append(cs.confusion_level)
                cognitive_flexibilities.append(cs.cognitive_flexibility)
                understanding_levels.append(cs.understanding_level)

        if not confusion_levels:
            return base_score

        # 计算调整因子
        avg_confusion = np.mean(confusion_levels)
        avg_flexibility = np.mean(cognitive_flexibilities)
        avg_understanding = np.mean(understanding_levels)

        # 调整逻辑：
        # - 高困惑水平：降低协同度（可能意味着认知状态不稳定）
        # - 高认知灵活性：略微提高协同度（灵活意味着适应性强）
        # - 高理解水平：提高协同度（理解意味着状态一致）
        confusion_penalty = avg_confusion * 0.2
        flexibility_bonus = avg_flexibility * 0.1
        understanding_bonus = avg_understanding * 0.1

        adjustment = flexibility_bonus + understanding_bonus - confusion_penalty

        # 应用调整，确保结果在 [-1, 1] 范围内
        adjusted_score = float(np.clip(base_score + adjustment, -1.0, 1.0))

        return adjusted_score

    def _classify_synergy_mode_with_cognition(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        synergy_score: float,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> Tuple[DominantDimension, SynergyMode]:
        """结合认知模块信息分类协同模式（新增）

        在基础分类逻辑上，利用认知模块的认知负荷、理解水平等信息
        进行更精准的模式分类。

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            synergy_score: 协同度评分
            member_pairs: 成员级配对数据

        Returns:
            Tuple[DominantDimension, SynergyMode]: (主导维度, 协同模式)
        """
        # 如果没有认知模块信息，使用基础分类
        if not member_pairs or self.cognition_source != "external":
            return self.classify_synergy_mode(
                emotion_features, cognition_features, synergy_score
            )

        # 计算基础的特征幅度判断
        e_magnitude = np.linalg.norm(emotion_features.flatten())
        c_magnitude = np.linalg.norm(cognition_features.flatten())

        total = e_magnitude + c_magnitude + 1e-8
        e_ratio = e_magnitude / total

        if e_ratio > 0.6:
            dominant = DominantDimension.EMOTION
        elif e_ratio < 0.4:
            dominant = DominantDimension.COGNITION
        else:
            dominant = DominantDimension.BALANCED

        # === 利用认知模块信息进行精细分类 ===
        # 收集认知状态信息
        cognitive_loads = []
        understanding_levels = []
        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'cognitive_load'):
                cognitive_loads.append(cs.cognitive_load)
                understanding_levels.append(cs.understanding_level)

        avg_load = np.mean(cognitive_loads) if cognitive_loads else 0.5
        avg_understanding = np.mean(understanding_levels) if understanding_levels else 0.5

        # 基于协同度 + 认知状态进行模式分类
        if synergy_score > 0.5:
            mode = SynergyMode.HARMONIOUS
        elif synergy_score > 0.0:
            # 如果认知负荷高且理解水平低，更可能是认知过载
            if avg_load > 0.7 and avg_understanding < 0.4:
                mode = SynergyMode.COGNITIVE_OVERLOAD
            else:
                mode = SynergyMode.EMOTIONAL_OVERWHELM
        elif synergy_score > -0.5:
            mode = SynergyMode.CONFLICTING
        else:
            # 如果认知负荷极低，可能是脱离
            if avg_load < 0.3:
                mode = SynergyMode.DISENGAGED
            else:
                mode = SynergyMode.CONFLICTING

        return dominant, mode

    def _generate_hint_with_cognition(
        self,
        dominant: DominantDimension,
        mode: SynergyMode,
        synergy_score: float,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> str:
        """结合认知模块信息生成建议提示（新增）

        利用认知模块提供的认知负荷、理解水平、认知阶段等信息，
        生成更精准、更具上下文感知的建议。

        Args:
            dominant: 主导维度
            mode: 协同模式
            synergy_score: 协同度评分
            member_pairs: 成员级配对数据

        Returns:
            str: 建议提示文本
        """
        # 如果没有认知模块信息，使用基础提示生成
        if not member_pairs or self.cognition_source != "external":
            return self._generate_hint(dominant, mode, synergy_score)

        # 收集认知模块的详细信息
        cognitive_state_info = {}
        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'source_engine'):
                # 使用第一个有效的认知状态作为代表
                cognitive_state_info = {
                    "cognitive_load": cs.cognitive_load,
                    "understanding_level": cs.understanding_level,
                    "cognitive_phase": getattr(cs.cognitive_phase, 'value', str(cs.cognitive_phase)),
                    "attention_score": cs.attention_score,
                    "confusion_level": cs.confusion_level,
                    "cognitive_flexibility": cs.cognitive_flexibility,
                    "source_engine": cs.source_engine,
                    "knowledge_nodes": cs.knowledge_nodes if hasattr(cs, 'knowledge_nodes') else [],
                }
                break

        if not cognitive_state_info:
            return self._generate_hint(dominant, mode, synergy_score)

        # 根据认知状态生成针对性的建议
        load = cognitive_state_info.get("cognitive_load", 0.5)
        understanding = cognitive_state_info.get("understanding_level", 0.5)
        phase = cognitive_state_info.get("cognitive_phase", "unknown")
        attention = cognitive_state_info.get("attention_score", 0.5)
        confusion = cognitive_state_info.get("confusion_level", 0.3)

        # 结合协同模式生成建议
        if mode == SynergyMode.HARMONIOUS:
            if phase == "核心理解":
                hint = (
                    f"当前处于核心理解阶段（认知负荷={load:.2f}，理解水平={understanding:.2f}），"
                    f"情绪与认知状态和谐。建议维持当前节奏，提供适度的巩固练习。"
                )
            elif phase == "精通期":
                hint = (
                    f"群体已进入精通阶段，情绪状态良好。"
                    f"建议适当提升难度或引入新内容。"
                )
            else:
                hint = f"当前情绪与认知状态和谐（{phase}），建议维持当前节奏。"

        elif mode == SynergyMode.EMOTIONAL_OVERWHELM:
            hint = (
                f"情绪状态占主导，认知状态为{phase}（负荷={load:.2f}）。"
                f"注意力={attention:.2f}，困惑水平={confusion:.2f}。"
                f"建议优先关注情绪调节，如通过互动或放松活动缓解紧张。"
            )

        elif mode == SynergyMode.COGNITIVE_OVERLOAD:
            hint = (
                f"认知负荷较高（{load:.2f}），理解水平为{understanding:.2f}。"
                f"注意力集中程度={attention:.2f}。"
                f"建议适当降低难度、提供阶段性总结，或安排短暂休息。"
            )

        elif mode == SynergyMode.CONFLICTING:
            if confusion > 0.6:
                hint = (
                    f"情绪与认知存在冲突，且困惑水平较高（{confusion:.2f}）。"
                    f"建议通过问答或小组讨论澄清疑难，调解情绪氛围。"
                )
            else:
                hint = (
                    f"情绪与认知状态不匹配（{phase}阶段）。"
                    f"建议通过互动或讨论调节氛围，确保认知任务不受情绪干扰。"
                )

        else:  # DISENGAGED
            if attention < 0.3:
                hint = (
                    f"参与度较低，注意力集中程度为{attention:.2f}。"
                    f"认知状态为{phase}。建议增加互动性或挑战性任务以提升参与度。"
                )
            else:
                hint = (
                    f"参与度较低，当前处于{phase}阶段。"
                    f"建议引入新话题或激励手段激活群体参与。"
                )

        return hint

    def _align_dimensions(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对齐特征向量的维度

        确保两个特征向量具有相同的维度才能进行加权求和。

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            Tuple[np.ndarray, np.ndarray]: 对齐后的特征向量
        """
        e_flat = emotion_features.flatten()
        c_flat = cognition_features.flatten()

        # 对齐到目标输出维度
        target_dim = self.output_dim

        if len(e_flat) >= target_dim:
            e_aligned = e_flat[:target_dim]
        else:
            e_aligned = np.pad(e_flat, (0, target_dim - len(e_flat)))

        if len(c_flat) >= target_dim:
            c_aligned = c_flat[:target_dim]
        else:
            c_aligned = np.pad(c_flat, (0, target_dim - len(c_flat)))

        return e_aligned, c_aligned
