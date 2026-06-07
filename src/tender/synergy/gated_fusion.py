"""
 门控机制融合引擎——情绪-认知协同模块（已更新以适配认知模块）

 该模块实现了基于门控机制的协同策略（Strategy 2）。
 与加权融合使用固定权重不同，门控机制根据情绪和认知特征的内在特性
 动态计算融合权重。

 工作流程：
 1. 获取情绪特征向量 E 和认知特征向量 K
 2. 计算情绪和认知特征的内部方差
 3. 基于方差生成门控权重 g（0-1），g 越接近 1 表示认知权重越高
 4. 使用门控权重计算融合特征：F = (1-g) * E + g * K
 5. 生成协同度评分和模式分类

 更新说明：
 - 在结果元数据中传播外部认知模块的状态（如认知负荷、理解水平等）
 - 当外部认知模块可用且置信度较高时，门控权重会略微偏向认知方差的判定

 Args:
     config: 配置字典，包含以下字段：
         - emotion_dim: 情绪特征维度（默认 16）
         - cognition_dim: 认知特征维度（默认 16）
         - output_dim: 输出维度（默认 32）
         - gate_hidden_dim: 门控网络隐藏层维度（默认 16）
         - gate_activation: 门控激活函数，"sigmoid" 或 "softmax"（默认 "sigmoid"）
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


class GatedFusionEngine(BaseSynergyEngine):
    """门控机制融合引擎（已更新以适配认知模块）

    Args:
        config: 配置字典
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化门控机制融合引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.emotion_dim = config.get("emotion_dim", 16)
        self.cognition_dim = config.get("cognition_dim", 16)
        self.output_dim = config.get("output_dim", 32)

        # 门控网络参数
        self.gate_hidden_dim = config.get("gate_hidden_dim", 16)
        self.gate_activation = config.get("gate_activation", "sigmoid")

        # 认知模块对接参数（新增）
        self.cognition_source = config.get("cognition_source", "internal")
        self.enable_cognition_metadata = config.get("enable_cognition_metadata", True)

        # 初始化门控网络权重（启发式版本）
        # 在高维空间中，有一个简单的启发式方法：
        # 如果情绪的方差大于认知的方差，说明情绪活动更剧烈，应给予更高权重
        # 反之，认知更占主导
        self._gate_bias = 0.5  # 默认偏置，g = 0.5 表示同等重要

        # 记录初始化信息
        self._init_info = (
            f"GatedFusionEngine initialized with "
            f"activation={self.gate_activation}, "
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

        执行门控机制融合的核心逻辑：
        1. 验证输入的有效性
        2. 计算门控权重（动态调整策略，需集成外部认知模块的置信度）
        3. 计算融合特征
        4. 计算协同度评分
        5. 分类协同模式
        6. 打包为 SynergyResult（包含认知模块元数据）

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            member_pairs: 可选的成员级配对数据（此引擎主要利用其元数据）

        Returns:
            SynergyResult: 包含融合特征和协同分析的结果
        """
        # 步骤1：输入验证
        self.validate_inputs(emotion_features, cognition_features)

        # 步骤2：对齐维度
        emotion_aligned, cognition_aligned = self._align_dimensions(
            emotion_features, cognition_features
        )

        # 步骤3：计算门控权重
        # 如果外部认知模块可用且提供了高置信度，则继续使用方差动态计算权重
        # 否则使用门控偏置
        gate_weight = self._compute_gate_weight(
            emotion_aligned, cognition_aligned, member_pairs
        )

        # 步骤4：计算加权特征
        combined_feature = (
            (1.0 - gate_weight) * emotion_aligned
            + gate_weight * cognition_aligned
        )

        # 步骤5：计算协同度评分
        synergy_score = self.compute_synergy_score(
            emotion_aligned, cognition_aligned
        )

        # 步骤6：分类协同模式
        dominant, mode = self.classify_synergy_mode(
            emotion_aligned, cognition_aligned, synergy_score
        )

        # 步骤7：生成建议提示
        recommendation_hint = self._generate_hint(dominant, mode, synergy_score)

        # 步骤8：提取认知模块的元数据（新增）
        cognition_metadata = {}
        if self.enable_cognition_metadata:
            cognition_metadata = self._extract_cognition_metadata(member_pairs)

        # 步骤9：打包结果
        result = SynergyResult(
            combined_feature=combined_feature,
            synergy_score=synergy_score,
            dominant_dimension=dominant,
            synergy_mode=mode,
            emotion_feature=emotion_features,
            cognition_feature=cognition_features,
            recommendation_hint=recommendation_hint,
            metadata={
                "method": "gated_fusion",
                "gate_weight": float(gate_weight),
                "gate_activation": self.gate_activation,
                # === 新增的认知模块元数据 ===
                "cognition_source": self.cognition_source,
                **cognition_metadata,  # 展开外部认知元数据
            },
        )

        return result

    def _compute_gate_weight(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> float:
        """计算门控权重（已更新，集成外部认知模块的置信度）

        核心启发式逻辑：
        1. 计算情绪特征的方差和认知特征的方差
        2. 如果情绪方差 > 认知方差，说明情绪变化剧烈，应降低其权重（g 较大）
        3. 如果认知方差 > 情绪方差，说明认知变化剧烈，应降低其权重（g 较小）
        4. 如果外部认知模块提供了高置信度，则适当提高g（更信任认知）

        Args:
            emotion_features: 对齐后的情绪特征向量
            cognition_features: 对齐后的认知特征向量
            member_pairs: 成员级配对数据（用于获取外部认知置信度）

        Returns:
            float: 门控权重，范围 [0, 1]
        """
        # 计算特征方差
        e_var = np.var(emotion_features)
        c_var = np.var(cognition_features)

        # 防除零
        epsilon = 1e-8

        # 基于方差的启发式门控
        # 如果方差比为1，则使用默认偏置
        if abs(e_var - c_var) < epsilon:
            base_gate = self._gate_bias
        else:
            # g = c_var / (e_var + c_var)
            # g 越大，认知权重越大
            base_gate = c_var / (e_var + c_var + epsilon)

        # === 新增：利用外部认知模块的置信度调整门控 ===
        if self.cognition_source == "external" and member_pairs:
            # 收集认知模块的置信度
            confidences = []
            for pair in member_pairs:
                cs = pair.cognition_state
                if cs is not None and hasattr(cs, 'phase_confidence'):
                    confidences.append(cs.phase_confidence)

            if confidences:
                avg_confidence = np.mean(confidences)
                # 当认知模块的置信度较高时，更倾向于信任认知方差计算结果
                # 当置信度较低时，回归到默认偏置
                # 映射因子: 置信度从0到1，调整量从 -0.2 到 0.1
                confidence_adjustment = (avg_confidence - 0.5) * 0.6
                base_gate += confidence_adjustment

        # 裁剪到 [0, 1] 范围
        return float(np.clip(base_gate, 0.0, 1.0))

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
