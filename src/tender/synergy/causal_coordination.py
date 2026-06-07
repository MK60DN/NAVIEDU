""" 因果协调分析引擎——情绪-认知协同模块（已更新以适配认知模块）

该模块实现了基于因果推断的高级协同策略（Strategy 4）。
与前面三种策略不同，因果协调不直接关注“如何融合”，
而是回答一个更根本的问题：“情绪和认知之间是否存在因果关系？”

核心思想：
- 情绪和认知不是简单的平行关系，而是相互影响的动态系统
- “因为理解困难而感到焦虑”是一种方向（认知 → 情绪）
- “因为情绪低落而学习效率下降”是另一种方向（情绪 → 认知）
- 因果协调分析能够检测这种方向性，从而做出更精准的干预建议

工作流程：
1. 将情绪特征和认知特征视为两个时间序列
2. 使用因果推断方法（CCM，格兰杰检验等）检测方向性
3. 输出因果方向和强度
4. 基于因果关系给出针对性建议

适用场景：
- 需要理解“情绪与认知谁驱动谁”的研究场景
- 长时间序列数据（至少 10 个以上时间点）
- 前沿学术探索：研究情绪与认知的动态交互机制

学术基础：
- 收敛交叉映射 (Sugihara et al., 2012): 检测非线性系统中的因果关系
- 格兰杰因果检验 (Granger, 1969): 经典的时间序列因果检验
- 情绪-认知交互的动力学模型 (Lewis, 2005): 情绪与认知的动态耦合

更新说明：
- 在因果分析中集成外部认知模块提供的认知状态时间序列元数据
- 利用认知模块的困惑水平、理解水平等信息优化因果方向判断
- 在输出元数据中传播认知模块的详细信息
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from tender.synergy.base import (
    BaseSynergyEngine,
    SynergyResult,
    DominantDimension,
    SynergyMode,
    EmotionCognitionPair,
)


class CausalDirection(Enum):
    """因果方向枚举

    描述情绪与认知之间的因果影响方向。
    """
    EMOTION_TO_COGNITION = "emotion_to_cognition"  # 情绪影响认知
    COGNITION_TO_EMOTION = "cognition_to_emotion"  # 认知影响情绪
    BIDIRECTIONAL = "bidirectional"                # 双向因果
    INDEPENDENT = "independent"                    # 无显著因果关系


@dataclass
class CausalAnalysisResult:
    """因果分析结果的数据结构（已更新）

    存储情绪与认知之间的因果分析详细信息。
    新增了对认知模块元数据的支持。

    Attributes:
        direction: 因果方向
        strength: 因果强度 (0-1)，越大表示因果效应越强
        p_value: 显著性检验的 p 值
        lag: 检测到的因果滞后步数
        confidence: 因果分析的置信度 (0-1)
        time_series_length: 使用的时间序列长度
        method: 使用的因果分析方法
        cognition_metadata: 来自认知模块的元数据（新增）
        knowledge_nodes: 涉及的知识点列表（新增）
    """
    direction: CausalDirection
    strength: float
    p_value: float
    lag: int
    confidence: float
    time_series_length: int
    method: str
    detail: Dict[str, Any] = field(default_factory=dict)
    # === 新增字段 ===
    cognition_metadata: Dict[str, Any] = field(default_factory=dict)  # 认知模块元数据
    knowledge_nodes: List[str] = field(default_factory=list)          # 涉及的知识点


class CausalCoordinationEngine(BaseSynergyEngine):
    """因果协调分析引擎（已更新以适配认知模块）

    使用因果推断方法分析情绪与认知之间的动态因果关系。
    该引擎不直接融合特征，而是输出因果分析结果供下游模块使用。

    支持的因果方法：
    - "ccm": 收敛交叉映射（默认），适用于非线性关系
    - "granger": 格兰杰因果检验，适用于近似线性关系
    - "pearson": 皮尔逊相关（简化版），仅检测相关性，不区分方向

    选择建议：
    - 数据长度 > 30 且可能存在非线性关系 → CCM
    - 数据长度 10-30 且关系可能近似线性 → 格兰杰检验
    - 数据长度 < 10 或仅需相关性参考 → 皮尔逊相关

    Args:
        config: 配置字典，包含以下字段：
            - emotion_dim: 情绪特征维度（默认 16）
            - cognition_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 32）
            - causal_method: 因果分析方法（默认 "ccm"）
            - causal_lag: 因果滞后步数（默认 1）
            - significance_level: 显著性水平（默认 0.05）
            - max_emotion_features: 情绪特征数量上限（默认 5）
            - max_cognition_features: 认知特征数量上限（默认 5）
            - cognition_source: 认知状态来源（默认 "internal"）
            - enable_cognition_metadata: 是否传播认知模块元数据（默认 True）
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化因果协调分析引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.emotion_dim = config.get("emotion_dim", 16)
        self.cognition_dim = config.get("cognition_dim", 16)
        self.output_dim = config.get("output_dim", 32)

        # 因果分析方法参数
        self.causal_method = config.get("causal_method", "ccm")
        self.causal_lag = config.get("causal_lag", 1)
        self.significance_level = config.get("significance_level", 0.05)

        # 特征选择参数
        self.max_emotion_features = config.get("max_emotion_features", 5)
        self.max_cognition_features = config.get("max_cognition_features", 5)

        # 认知模块对接参数（新增）
        self.cognition_source = config.get("cognition_source", "internal")
        self.enable_cognition_metadata = config.get("enable_cognition_metadata", True)

        # 因果方向到情感描述的映射
        self._direction_labels = {
            CausalDirection.EMOTION_TO_COGNITION: "情绪驱动认知",
            CausalDirection.COGNITION_TO_EMOTION: "认知驱动情绪",
            CausalDirection.BIDIRECTIONAL: "双向因果",
            CausalDirection.INDEPENDENT: "无明显因果关系",
        }

        # 记录初始化信息
        self._init_info = (
            f"CausalCoordinationEngine initialized with "
            f"method={self.causal_method}, "
            f"lag={self.causal_lag}, "
            f"significance={self.significance_level}, "
            f"max_features_E={self.max_emotion_features}, "
            f"max_features_K={self.max_cognition_features}, "
            f"cognition_source={self.cognition_source}"
        )

    def fuse(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> SynergyResult:
        """情绪与认知的因果协调分析（已更新）

        核心区别：此引擎不执行特征融合，而是进行因果分析。
        融合特征由因果分析结果构建，而非直接拼接特征。

        新增支持：
        - 从 member_pairs 中提取认知模块的时间序列元数据
        - 利用认知模块的困惑水平等信息优化因果方向判断

        Args:
            emotion_features: 情绪特征时间序列
                形状: (n_timepoints, d_emotion) 或 (n_timepoints,)
                注意：至少需要 n_timepoints >= 10 才能获得可靠结果
            cognition_features: 认知特征时间序列
                形状: (n_timepoints, d_cognition) 或 (n_timepoints,)
            member_pairs: 可选的成员级配对数据
                如果包含来自 tender.cognition 的 CognitionState 对象，
                其元数据将被提取并用于优化因果分析

        Returns:
            SynergyResult: 包含因果分析的结果
                - combined_feature: 基于因果分析构建的综合特征向量
                - synergy_score: 因果强度（正表示正相关因果，负表示负相关因果）
                - dominant_dimension: 因果方向决定的主导维度
                - synergy_mode: 基于因果模式的协同模式
        """
        # 步骤1：输入验证
        self.validate_inputs(emotion_features, cognition_features)

        # 步骤2：准备时间序列
        e_ts, c_ts = self._prepare_time_series(
            emotion_features, cognition_features
        )

        # 步骤3：提取认知模块元数据（新增）
        cognition_meta = self._collect_cognition_metadata(member_pairs)

        # 步骤4：执行因果分析（现在利用认知元数据进行优化）
        causal_result = self._run_causal_analysis(e_ts, c_ts)

        # === 新增：将认知模块元数据注入因果分析结果 ===
        if cognition_meta:
            causal_result.cognition_metadata = cognition_meta
            # 利用认知模块信息优化因果方向判断
            causal_result = self._refine_with_cognition(causal_result, cognition_meta)

        # 步骤5：基于因果结果构建融合特征
        combined_feature = self._build_causal_feature(causal_result, e_ts, c_ts)

        # 步骤6：计算协同度（基于因果强度和方向）
        synergy_score = self.compute_synergy_score(
            emotion_features, cognition_features
        )

        # 步骤7：分类协同模式
        dominant, mode = self.classify_synergy_mode(
            emotion_features, cognition_features, synergy_score
        )

        # 步骤8：生成基于因果分析的建议提示（利用认知模块信息）
        recommendation_hint = self._generate_causal_hint(causal_result)

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
                "method": "causal_coordination",
                "causal_method": self.causal_method,
                "causal_direction": causal_result.direction.value,
                "causal_strength": causal_result.strength,
                "causal_p_value": causal_result.p_value,
                "causal_lag": causal_result.lag,
                "causal_confidence": causal_result.confidence,
                "direction_label": self._direction_labels.get(
                    causal_result.direction, "unknown"
                ),
                "time_series_length": causal_result.time_series_length,
                # === 新增的认知模块元数据 ===
                "cognition_source": self.cognition_source,
                "cognitive_metadata": causal_result.cognition_metadata,
                "knowledge_nodes": causal_result.knowledge_nodes,
            },
        )

        return result

    def _collect_cognition_metadata(
        self,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> Dict[str, Any]:
        """从 member_pairs 中收集认知模块的元数据（新增）

        提取认知模块提供的认知状态信息，用于优化因果分析。

        Args:
            member_pairs: 成员级配对数据

        Returns:
            Dict[str, Any]: 认知模块元数据字典
        """
        if not member_pairs or self.cognition_source != "external":
            return {}

        metadata = {}
        cognitive_loads = []
        understanding_levels = []
        confusion_levels = []
        all_knowledge_nodes = []

        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'cognitive_load'):
                cognitive_loads.append(cs.cognitive_load)
                understanding_levels.append(cs.understanding_level)
                confusion_levels.append(cs.confusion_level if hasattr(cs, 'confusion_level') else 0.3)
                if hasattr(cs, 'knowledge_nodes') and cs.knowledge_nodes:
                    all_knowledge_nodes.extend(cs.knowledge_nodes)

        if not cognitive_loads:
            return {}

        metadata = {
            "avg_cognitive_load": float(np.mean(cognitive_loads)),
            "avg_understanding_level": float(np.mean(understanding_levels)),
            "avg_confusion_level": float(np.mean(confusion_levels)),
            "n_members_with_cognition": len(cognitive_loads),
            "unique_knowledge_nodes": list(set(all_knowledge_nodes)),
            "source_engine": member_pairs[0].cognition_state.source_engine
                if hasattr(member_pairs[0].cognition_state, 'source_engine') else "unknown",
        }

        return metadata

    def _refine_with_cognition(
        self,
        causal_result: CausalAnalysisResult,
        cognition_meta: Dict[str, Any],
    ) -> CausalAnalysisResult:
        """利用认知模块信息优化因果分析结果（新增）

        核心逻辑：
        - 如果认知模块的困惑水平较高，适当降低对“认知→情绪”方向的置信度
        - 如果理解水平较低，提高对“情绪→认知”方向的判断权重
        - 注入知识节点信息

        Args:
            causal_result: 原始因果分析结果
            cognition_meta: 认知模块元数据

        Returns:
            CausalAnalysisResult: 优化后的因果分析结果
        """
        avg_confusion = cognition_meta.get("avg_confusion_level", 0.3)
        avg_understanding = cognition_meta.get("avg_understanding_level", 0.5)

        # 根据困惑水平调整
        if avg_confusion > 0.6 and causal_result.direction == CausalDirection.COGNITION_TO_EMOTION:
            # 高困惑水平下，认知对情绪的影响可能被放大，稍微降低置信度
            causal_result.confidence = max(0.0, causal_result.confidence - 0.1)

        elif avg_understanding < 0.3 and causal_result.direction == CausalDirection.EMOTION_TO_COGNITION:
            # 低理解水平下，情绪对认知的影响可能更显著，稍微提高强度
            causal_result.strength = min(1.0, causal_result.strength + 0.05)

        # 注入知识节点信息
        knowledge_nodes = cognition_meta.get("unique_knowledge_nodes", [])
        if knowledge_nodes:
            causal_result.knowledge_nodes = knowledge_nodes

        return causal_result

    def compute_synergy_score(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> float:
        """计算情绪与认知的协同度得分

        在因果协调中，协同度得分等于因果强度（带方向）。
        - 正值: 情绪和认知同向变化（如都增加或都减少）
        - 负值: 情绪和认知反向变化

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            float: 协同度得分，范围 [-1, 1]
        """
        e_flat = emotion_features.flatten()
        c_flat = cognition_features.flatten()

        # 确保长度一致
        min_len = min(len(e_flat), len(c_flat))
        e_flat = e_flat[:min_len]
        c_flat = c_flat[:min_len]

        # 计算皮尔逊相关系数
        if len(e_flat) < 2:
            return 0.0

        corr_matrix = np.corrcoef(e_flat, c_flat)
        correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

        return correlation

    def classify_synergy_mode(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        synergy_score: float,
    ) -> Tuple[DominantDimension, SynergyMode]:
        """分类协同模式（基于因果方向）

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            synergy_score: 协同度得分

        Returns:
            Tuple[DominantDimension, SynergyMode]: (主导维度, 协同模式)
        """
        # 计算特征幅度
        e_magnitude = np.linalg.norm(emotion_features.flatten())
        c_magnitude = np.linalg.norm(cognition_features.flatten())

        # 判断主导维度
        total = e_magnitude + c_magnitude
        e_ratio = e_magnitude / total if total > 0 else 0.5

        if e_ratio > 0.6:
            dominant = DominantDimension.EMOTION
        elif e_ratio < 0.4:
            dominant = DominantDimension.COGNITION
        else:
            dominant = DominantDimension.BALANCED

        # 基于协同度判断协同模式
        if synergy_score > 0.5:
            mode = SynergyMode.HARMONIOUS
        elif synergy_score > 0.0:
            mode = SynergyMode.EMOTIONAL_OVERWHELM
        elif synergy_score > -0.5:
            mode = SynergyMode.CONFLICTING
        else:
            mode = SynergyMode.DISENGAGED

        return dominant, mode

    def _prepare_time_series(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从特征向量中提取时间序列（原有逻辑，保持不变）"""
        # 确保是二维数组
        e_2d = emotion_features.reshape(-1, self.emotion_dim) if emotion_features.ndim == 1 else emotion_features
        c_2d = cognition_features.reshape(-1, self.cognition_dim) if cognition_features.ndim == 1 else cognition_features

        # 获取时间步数
        n_timepoints = min(e_2d.shape[0], c_2d.shape

        if n_timepoints < self.causal_lag + 3:
            # 时间序列太短，使用聚合特征
            e_ts = np.mean(e_2d, axis=1)
            c_ts = np.mean(c_2d, axis=1)
        else:
            # 选择方差最大的几个特征维度
            e_variances = np.var(e_2d, axis=0)
            c_variances = np.var(c_2d, axis=0)

            k_e = min(self.max_emotion_features, e_2d.shape
            k_c = min(self.max_cognition_features, c_2d.shape

            top_e_indices = np.argsort(e_variances)[-k_e:]
            top_c_indices = np.argsort(c_variances)[-k_c:]

            e_ts = np.mean(e_2d[:, top_e_indices], axis=1)
            c_ts = np.mean(c_2d[:, top_c_indices], axis=1)

        return e_ts[:n_timepoints], c_ts[:n_timepoints]

    def _run_causal_analysis(
        self,
        e_ts: np.ndarray,
        c_ts: np.ndarray,
    ) -> CausalAnalysisResult:
        """执行因果分析（原有逻辑，保持不变）"""
        if self.causal_method == "ccm":
            return self._run_ccm_analysis(e_ts, c_ts)
        elif self.causal_method == "granger":
            return self._run_granger_analysis(e_ts, c_ts)
        elif self.causal_method == "pearson":
            return self._run_pearson_analysis(e_ts, c_ts)
        else:
            return self._run_ccm_analysis(e_ts, c_ts)

    def _run_ccm_analysis(
        self,
        e_ts: np.ndarray,
        c_ts: np.ndarray,
    ) -> CausalAnalysisResult:
        """执行收敛交叉映射（CCM）因果分析（原有逻辑，保持不变）"""
        n = len(e_ts)
        lag = self.causal_lag
        E = lag + 1

        if n < E + 3:
            return CausalAnalysisResult(
                direction=CausalDirection.INDEPENDENT,
                strength=0.0,
                p_value=1.0,
                lag=lag,
                confidence=0.0,
                time_series_length=n,
                method="ccm",
                detail={"reason": "时间序列太短，无法执行 CCM 分析"},
            )

        pred_e_to_c = self._simplex_projection(e_ts, c_ts, E, lag)
        skill_e_to_c = self._compute_prediction_skill(c_ts[E+lag:], pred_e_to_c)

        pred_c_to_e = self._simplex_projection(c_ts, e_ts, E, lag)
        skill_c_to_e = self._compute_prediction_skill(e_ts[E+lag:], pred_c_to_e)

        total_skill = skill_e_to_c + skill_c_to_e

        if total_skill < 0.1:
            direction = CausalDirection.INDEPENDENT
            strength = 0.0
            confidence = 0.0
            p_value = 0.5
        elif skill_e_to_c > skill_c_to_e * 1.5:
            direction = CausalDirection.EMOTION_TO_COGNITION
            strength = float(skill_e_to_c)
            confidence = float(min(1.0, total_skill))
            p_value = float(max(0.01, 1.0 - confidence))
        elif skill_c_to_e > skill_e_to_c * 1.5:
            direction = CausalDirection.COGNITION_TO_EMOTION
            strength = float(skill_c_to_e)
            confidence = float(min(1.0, total_skill))
            p_value = float(max(0.01, 1.0 - confidence))
        elif total_skill > 0.3:
            direction = CausalDirection.BIDIRECTIONAL
            strength = float(max(skill_e_to_c, skill_c_to_e))
            confidence = float(min(1.0, total_skill))
            p_value = float(max(0.01, 1.0 - confidence))
        else:
            direction = CausalDirection.INDEPENDENT
            strength = float(total_skill)
            confidence = float(total_skill)
            p_value = 0.5

        return CausalAnalysisResult(
            direction=direction,
            strength=strength,
            p_value=p_value,
            lag=lag,
            confidence=confidence,
            time_series_length=n,
            method="ccm",
            detail={
                "skill_e_to_c": float(skill_e_to_c),
                "skill_c_to_e": float(skill_c_to_e),
                "embedding_dimension": E,
                "total_points": n,
            },
        )

    def _simplex_projection(
        self, source: np.ndarray, target: np.ndarray, E: int, lag: int
    ) -> np.ndarray:
        """单纯形投影（原有逻辑，保持不变）"""
        n = len(source)
        n_vectors = n - E - lag + 1
        predictions = np.zeros(n_vectors)

        for i in range(n_vectors):
            # 构建目标流形点
            target_point = np.array([
                target[i + j * lag]
                for j in range(E)
            ])

            # 寻找最近的 E+1 个邻居（排除自身）
            distances = []
            for j in range(n_vectors):
                if j == i:
                    continue
                source_point = np.array([
                    source[j + k * lag]
                    for k in range(E)
                ])
                dist = np.linalg.norm(target_point - source_point)
                distances.append((dist, j))

            distances.sort(key=lambda x: x
            neighbors = distances[:E+1]

            # 加权平均预测
            total_weight = 0.0
            predicted_value = 0.0
            for dist, idx in neighbors:
                weight = np.exp(-dist / (max(1e-8, distances))
                total_weight += weight
                predicted_value += weight * source[idx + E + lag - 1]

            if total_weight > 0:
                predictions[i] = predicted_value / total_weight

        return predictions

    def _compute_prediction_skill(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> float:
        """计算预测技能（原有逻辑，保持不变）"""
        if len(actual) < 3 or len(predicted) < 3:
            return 0.0

        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        corr_matrix = np.corrcoef(actual, predicted)
        correlation = corr_matrix[0, 1]

        if np.isnan(correlation):
            return 0.0

        skill = max(0.0, float(correlation))
        return skill

    def _run_granger_analysis(
        self, e_ts: np.ndarray, c_ts: np.ndarray
    ) -> CausalAnalysisResult:
        """执行格兰杰因果检验（简化版，原有逻辑，保持不变）"""
        n = len(e_ts)
        lag = self.causal_lag

        if n < lag * 2 + 5:
            return CausalAnalysisResult(
                direction=CausalDirection.INDEPENDENT,
                strength=0.0,
                p_value=1.0,
                lag=lag,
                confidence=0.0,
                time_series_length=n,
                method="granger",
                detail={"reason": "时间序列太短"},
            )

        def autoregressive_predict(ts: np.ndarray, lags: int) -> Tuple[np.ndarray, float]:
            """自回归预测"""
            n_ts = len(ts)
            if n_ts < lags + 2:
                return np.array([]), float('inf')
            X = np.zeros((n_ts - lags, lags))
            y = ts[lags:]
            for i in range(lags, n_ts):
                X[i - lags] = ts[i - lags:i]
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)
                predictions = X @ coeffs
                error = np.mean((y - predictions) ** 2)
                return predictions, error
            except np.linalg.LinAlgError:
                return np.array([]), float('inf')

        def cross_prediction(
            source: np.ndarray, target: np.ndarray, lags: int
        ) -> Tuple[np.ndarray, float]:
            """交叉预测"""
            n_st = len(source)
            if n_st < lags * 2 or len(target) < lags * 2:
                return np.array([]), float('inf')
            X = np.zeros((n_st - lags, lags * 2))
            y = target[lags:]
            for i in range(lags, n_st):
                X[i - lags] = np.concatenate([
                    target[i - lags:i],
                    source[i - lags:i]
                ])
            try:
                coeffs = np.linalg.lstsq(X, y, rcond=None)
                predictions = X @ coeffs
                error = np.mean((y - predictions) ** 2)
                return predictions, error
            except np.linalg.LinAlgError:
                return np.array([]), float('inf')

        _, error_c_base = autoregressive_predict(c_ts, lag)
        _, error_c_cross = cross_prediction(e_ts, c_ts, lag)

        _, error_e_base = autoregressive_predict(e_ts, lag)
        _, error_e_cross = cross_prediction(c_ts, e_ts, lag)

        improvement_e_to_c = (error_c_base - error_c_cross) / (error_c_base + 1e-8)
        improvement_c_to_e = (error_e_base - error_e_cross) / (error_e_base + 1e-8)

        if improvement_e_to_c < 0 and improvement_c_to_e < 0:
            direction = CausalDirection.INDEPENDENT
            strength = 0.0
            confidence = 0.0
            p_value = 0.5
        elif improvement_e_to_c > improvement_c_to_e * 1.5:
            direction = CausalDirection.EMOTION_TO_COGNITION
            strength = float(min(1.0, improvement_e_to_c))
            confidence = float(min(1.0, abs(improvement_e_to_c)))
            p_value = float(max(0.01, 1.0 - confidence))
        elif improvement_c_to_e > improvement_e_to_c * 1.5:
            direction = CausalDirection.COGNITION_TO_EMOTION
            strength = float(min(1.0, improvement_c_to_e))
            confidence = float(min(1.0, abs(improvement_c_to_e)))
            p_value = float(max(0.01, 1.0 - confidence))
        elif improvement_e_to_c > 0 or improvement_c_to_e > 0:
            direction = CausalDirection.BIDIRECTIONAL
            strength = float(min(1.0, max(improvement_e_to_c, improvement_c_to_e)))
            confidence = float(min(1.0, abs(improvement_e_to_c) + abs(improvement_c_to_e)))
            p_value = float(max(0.01, 1.0 - confidence * 0.5))
        else:
            direction = CausalDirection.INDEPENDENT
            strength = 0.0
            confidence = 0.0
            p_value = 0.5

        return CausalAnalysisResult(
            direction=direction,
            strength=strength,
            p_value=p_value,
            lag=lag,
            confidence=confidence,
            time_series_length=n,
            method="granger",
            detail={
                "improvement_e_to_c": float(improvement_e_to_c),
                "improvement_c_to_e": float(improvement_c_to_e),
            },
        )

    def _run_pearson_analysis(
        self, e_ts: np.ndarray, c_ts: np.ndarray
    ) -> CausalAnalysisResult:
        """执行皮尔逊相关分析（简化版，原有逻辑，保持不变）"""
        n = len(e_ts)

        if n < 3:
            return CausalAnalysisResult(
                direction=CausalDirection.INDEPENDENT,
                strength=0.0,
                p_value=1.0,
                lag=0,
                confidence=0.0,
                time_series_length=n,
                method="pearson",
                detail={"reason": "样本量太小"},
            )

        corr_matrix = np.corrcoef(e_ts, c_ts)
        corr = corr_matrix[0, 1]

        if np.isnan(corr):
            return CausalAnalysisResult(
                direction=CausalDirection.INDEPENDENT,
                strength=0.0,
                p_value=1.0,
                lag=0,
                confidence=0.0,
                time_series_length=n,
                method="pearson",
                detail={"reason": "相关系数为 NaN"},
            )

        abs_corr = abs(corr)
        confidence = abs_corr
        p_value = max(0.01, 1.0 - abs_corr)

        if abs_corr < 0.1:
            direction = CausalDirection.INDEPENDENT
            strength = 0.0
        elif corr > 0:
            direction = CausalDirection.BIDIRECTIONAL
            strength = float(abs_corr)
        else:
            direction = CausalDirection.BIDIRECTIONAL
            strength = float(abs_corr)

        return CausalAnalysisResult(
            direction=direction,
            strength=strength,
            p_value=p_value,
            lag=0,
            confidence=confidence,
            time_series_length=n,
            method="pearson",
            detail={
                "correlation": float(corr) if not np.isnan(corr) else 0.0
            },
        )

    def _build_causal_feature(
        self,
        causal_result: CausalAnalysisResult,
        e_ts: np.ndarray,
        c_ts: np.ndarray,
    ) -> np.ndarray:
        """基于因果分析结果构建综合特征向量（原有逻辑，保持不变）"""
        d = self.output_dim
        feature = np.zeros(d)

        direction_onehot = np.zeros(4)
        direction_idx = {
            CausalDirection.EMOTION_TO_COGNITION: 0,
            CausalDirection.COGNITION_TO_EMOTION: 1,
            CausalDirection.BIDIRECTIONAL: 2,
            CausalDirection.INDEPENDENT: 3,
        }
        idx = direction_idx.get(causal_result.direction, 3)
        direction_onehot[idx] = 1.0
        feature[0:4] = direction_onehot

        feature[4] = causal_result.strength
        feature[5] = causal_result.confidence

        max_lag = max(1, self.causal_lag)
        feature[6] = min(1.0, causal_result.lag / max_lag)

        direction_dist = np.zeros(4)
        direction_dist[idx] = causal_result.confidence
        remaining = 1.0 - causal_result.confidence
        if len([i for i in range(4) if i != idx]) > 0:
            direction_dist[[i for i in range(4) if i != idx]] = remaining / 3.0
        feature[7:11] = direction_dist

        if d > 11:
            remaining_dims = d - 11
            if len(causal_result.knowledge_nodes) > 0:
                num_knowledge = min(
                    len(causal_result.knowledge_nodes), remaining_dims - 1
                )
                for i in range(num_knowledge):
                    if 11 + i < d:
                        feature[11 + i] = 0.5  # 指示有知识点关联的占位符

            if len(e_ts) > 0 and len(c_ts) > 0:
                e_var = np.var(e_ts)
                c_var = np.var(c_ts)
                feature[11 + remaining_dims - 1] = np.clip(e_var - c_var, 0.0, 1.0)

        return feature

    def _generate_causal_hint(
        self,
        causal_result: CausalAnalysisResult,
    ) -> str:
        """生成基于因果分析的建议提示（已更新，利用认知模块信息）"""
        direction = causal_result.direction
        strength = causal_result.strength
        confidence = causal_result.confidence
        method = causal_result.method.upper()

        # === 新增：利用认知模块信息生成更精准的建议 ===
        cognition_info = causal_result.cognition_metadata
        cognitive_load = cognition_info.get("avg_cognitive_load", None) if cognition_info else None
        understanding_level = cognition_info.get("avg_understanding_level", None) if cognition_info else None

        base_info = f"[{method}分析] 因果强度={strength:.2f}, 置信度={confidence:.2f}。"

        if direction == CausalDirection.EMOTION_TO_COGNITION:
            if confidence > 0.6:
                hint = (
                    f"{base_info} "
                    f"检测到情绪正在驱动认知状态。"
                    f"建议：优先关注情绪调节，因为改善情绪将有助于提升认知表现。"
                    f"可以尝试引入轻松话题或正向激励。"
                )
                # 如果认知负荷高，补充建议
                if cognitive_load is not None and cognitive_load > 0.6:
                    hint += f" 当前认知负荷较高（{cognitive_load:.2f}），情绪调节可能更为关键。"
            else:
                hint = (
                    f"{base_info} "
                    f"初步迹象显示情绪可能影响认知。"
                    f"建议：关注情绪波动对学习效率的潜在影响。"
                )

        elif direction == CausalDirection.COGNITION_TO_EMOTION:
            if confidence > 0.6:
                hint = (
                    f"{base_info} "
                    f"检测到认知状态正在驱动情绪变化。"
                    f"建议：优先优化认知任务设计，因为改善认知体验将有助于提升情绪状态。"
                    f"可以调整任务难度或提供更清晰的知识引导。"
                )
                # 如果理解水平低，补充建议
                if understanding_level is not None and understanding_level < 0.3:
                    hint += f" 当前理解水平偏低（{understanding_level:.2f}），建议降低难度或提供辅助学习材料。"
            else:
                hint = (
                    f"{base_info} "
                    f"初步迹象显示认知难度可能影响情绪。"
                    f"建议：关注当前任务对群体情绪的潜在影响。"
                )

        elif direction == CausalDirection.BIDIRECTIONAL:
            if confidence > 0.6:
                hint = (
                    f"{base_info} "
                    f"检测到情绪与认知存在双向因果耦合。"
                    f"建议：采取综合干预策略。同时关注情绪氛围和认知负荷。"
                    f"例如，在提供认知支持的同时引入情绪疏导。"
                )
                # 如果有认知负荷信息，补充建议
                if cognitive_load is not None:
                    hint += f" 当前认知负荷={cognitive_load:.2f}，建议根据负荷水平动态调整。"
            else:
                hint = (
                    f"{base_info} "
                    f"情绪与认知存在一定的交互影响。"
                    f"建议：保持当前节奏，密切观察两者变化。"
                )

        else:  # INDEPENDENT
            if strength > 0.2:
                hint = (
                    f"{base_info} "
                    f"未检测到明确的因果关系，但两者存在相关性。"
                    f"建议：当前情绪和认知相对独立，可以分别进行优化。"
                )
            else:
                hint = (
                    f"{base_info} "
                    f"情绪与认知之间未发现显著的统计关系。"
                    f"建议：延长观察时间或收集更多数据以获得可靠结论。"
                )

        # === 新增：注入知识节点信息 ===
        knowledge_nodes = causal_result.knowledge_nodes
        if knowledge_nodes:
            hint += f" 当前涉及的知识点: {'、'.join(knowledge_nodes[:5])}。"

        return hint
