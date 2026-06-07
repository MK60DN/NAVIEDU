"""
 分层推理引擎——情绪-认知协同模块（已更新以适配认知模块）

 该模块实现了基于分层推理的协同策略（Strategy 3）。
 与加权融合和门控机制不同，分层推理不直接融合特征，
 而是先确定一个维度的状态，再用该状态来解释另一个维度。

 核心思想：
 - 情绪和认知并非平行关系，而是存在层级结构
 - 在某些情况下，认知状态决定了情绪的解释框架
 - 在另一些情况下，情绪状态决定了认知的处理方式
 - 分层推理模拟了人类认知中的"背景-前景"交互

 工作流程（默认模式：认知优先）：
 1. 首先分析认知状态（如：正在学习新概念，处于理解困难阶段）
 2. 然后在该认知状态下分析情绪（如：理解困难导致低愉悦度、高唤醒度）
 3. 判断该情绪是否"适应"当前认知状态
 4. 如果适应（如困惑时的焦虑），则不视为问题
 5. 如果不适应（如抵触情绪），则标记为需要关注

 适用场景：
 - 在线教育、培训场景
 - 需要先理解"学什么"再判断"心情如何"
 - 情绪与认知存在明确的时间先后或逻辑依赖关系

 学术基础：
 - 认知评价理论 (Lazarus, 1991)：情绪是对认知评价的反应
 - 认知负荷理论 (Sweller, 1988)：认知负荷影响情绪体验
 - 情感-认知交互模型 (Pessoa, 2008)：情感与认知的分层处理机制
 """

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from tender.synergy.base import (
    BaseSynergyEngine,
    SynergyResult,
    DominantDimension,
    SynergyMode,
    EmotionCognitionPair,
)

# 导入新认知模块的核心数据结构
try:
    from tender.cognition.base import CognitionState as ExternalCognitionState, CognitivePhase, EngagementType
    _COGNITION_MODULE_AVAILABLE = True
except ImportError:
    # 回退：定义兼容的桩类型
    class ExternalCognitionState:
        """桩类：当认知模块不可用时的回退"""
        def __init__(self):
            self.cognitive_load = 0.5
            self.understanding_level = 0.5
            self.cognitive_phase = "前期探索"
            self.engagement_type = "passive"
            self.attention_score = 0.5
            self.confusion_level = 0.3
            self.cognitive_flexibility = 0.5
            self.phase_confidence = 0.5
            self.source_engine = "unknown"
            self.timestamp = 0.0
            self.knowledge_nodes = []
    _COGNITION_MODULE_AVAILABLE = False


@dataclass
class CognitionState:
    """认知状态的数据结构（已更新）

    用于在分层推理中描述当前群体的认知状态。

    修改说明：
    - 新增字段 attention_score, confusion_level, cognitive_flexibility
    - 新增字段 source_engine（记录认知状态来源）
    - 新增字段 external_state（保留对外部 CognitionState 对象的引用）

    Attributes:
        member_id: 成员唯一标识（如果为群体分析，设为 "__group__"）
        cognitive_load: 认知负荷水平 (0-1)
        understanding_level: 理解水平 (0-1)
        engagement_type: 参与类型描述（"passive", "moderate", "focused", "intense"）
        phase_description: 认知阶段描述（如"前期探索"、"核心理解"等）
        expected_emotion: 该认知状态下期望的情绪模式描述
        confidence: 认知状态分析的置信度 (0-1)
        attention_score: 注意力集中程度 (0-1)，新增
        confusion_level: 困惑水平 (0-1)，新增
        cognitive_flexibility: 认知灵活性 (0-1)，新增
        source_engine: 生成该状态的引擎名称，新增
        external_state: 对外部 CognitionState 对象的引用（如来自 tender.cognition），新增
        knowledge_nodes: 涉及的知识点 ID 列表，新增
        timestamp: 时间戳，新增
    """
    member_id: str = "__group__"
    cognitive_load: float = 0.5
    understanding_level: float = 0.5
    engagement_type: str = "passive"
    phase_description: str = "前期探索"
    expected_emotion: str = "中性"
    confidence: float = 0.5

    # === 新增字段 ===
    attention_score: float = 0.5            # 注意力集中程度
    confusion_level: float = 0.3            # 困惑水平
    cognitive_flexibility: float = 0.5      # 认知灵活性
    source_engine: str = "internal"         # 认知状态来源引擎
    external_state: Optional[Any] = None    # 对外部 CognitionState 对象的引用
    knowledge_nodes: List[str] = field(default_factory=list)  # 涉及的知识点列表
    timestamp: float = 0.0                  # 时间戳


class LayeredReasoningEngine(BaseSynergyEngine):
    """分层推理引擎（已更新以适配认知模块）

    该引擎实现了分层推理的协同策略。
    现在支持两种认知状态来源：
    1. internal（内部）：使用引擎自带的启发式方法分析认知状态
    2. external（外部）：直接接收来自 tender.cognition 模块的 CognitionState 对象

    Args:
        config: 配置字典，包含以下字段：
            - emotion_dim: 情绪特征维度（默认 16）
            - cognition_dim: 认知特征维度（默认 16）
            - output_dim: 输出维度（默认 32）
            - cognition_first: 是否先分析认知再分析情绪（默认 True）
            - cognition_threshold: 认知状态判断阈值（默认 0.5）
            - cognition_source: 认知状态来源（默认 "internal"）
                - "internal": 使用内部启发式方法
                - "external": 使用外部认知模块
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化分层推理引擎

        Args:
            config: 配置字典
        """
        # 核心维度参数
        self.emotion_dim = config.get("emotion_dim", 16)
        self.cognition_dim = config.get("cognition_dim", 16)
        self.output_dim = config.get("output_dim", 32)

        # 分层推理参数
        self.cognition_first = config.get("cognition_first", True)
        self.cognition_threshold = config.get("cognition_threshold", 0.5)
        self.emotion_threshold = config.get("emotion_threshold", 0.5)

        # 认知模块对接参数（新增）
        self.cognition_source = config.get("cognition_source", "internal")
        self.enable_cognition_metadata = config.get("enable_cognition_metadata", True)

        # 认知阶段的期望情绪模板（加强版）
        # 根据不同的认知阶段，定义了该阶段下"自然"或"适应"的情绪状态
        self._adaptive_emotion = {
            "前期探索": {"valence": 0.4, "arousal": 0.6},      # 好奇但有些不确定
            "核心理解": {"valence": 0.3, "arousal": 0.7},      # 专注但可能有些焦虑
            "应用巩固": {"valence": 0.6, "arousal": 0.5},      # 愉悦且适度兴奋
            "疲劳期":   {"valence": 0.3, "arousal": 0.3},      # 疲倦且低落
            "精通期":   {"valence": 0.8, "arousal": 0.6},      # 满足且兴奋
            "intense":  {"valence": 0.5, "arousal": 0.8},      # 高强度参与
            "focused":  {"valence": 0.6, "arousal": 0.6},      # 专注
            "moderate": {"valence": 0.5, "arousal": 0.5},      # 中度参与
            "passive":  {"valence": 0.4, "arousal": 0.3},      # 被动参与
        }

        # 认知阶段列表（用于 one-hot 编码）
        self._cognition_phases = [
            "前期探索", "核心理解", "应用巩固", "疲劳期", "精通期",
            "intense", "focused", "moderate", "passive"
        ]

        self._init_info = (
            f"LayeredReasoningEngine initialized with "
            f"cognition_first={self.cognition_first}, "
            f"threshold={self.cognition_threshold}, "
            f"cognition_source={self.cognition_source}, "
            f"phases={len(self._cognition_phases)}"
        )

    def fuse(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> SynergyResult:
        """融合情绪和认知特征（分层推理 - 已更新）

        分层推理的核心逻辑分为两个阶段：
        阶段一：分析第一层的状态
        阶段二：基于第一层状态解释第二层

        新增支持：
        - 如果 member_pairs 中包含来自外部认知模块的 CognitionState，直接使用
        - 否则使用内部启发式方法分析认知状态

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量
            member_pairs: 可选的成员级配对数据（可选，用于细化分析）

        Returns:
            SynergyResult: 分层推理后的协同分析结果
        """
        # 步骤1：输入验证
        self.validate_inputs(emotion_features, cognition_features)

        # 步骤2：对齐维度
        e_aligned, c_aligned = self._align_features(emotion_features, cognition_features)

        # 步骤3：分析认知状态（支持内部和外部两种来源）
        cognition_state = self._analyze_cognition_state(c_aligned, member_pairs)

        # 步骤4：分析情绪状态（第二层），基于认知状态
        emotion_analysis = self._analyze_emotion_in_context(e_aligned, cognition_state)

        # 步骤5：计算协同度（基于情绪是否适应认知状态）
        synergy_score = emotion_analysis["adaptation_score"]

        # 步骤6：构建融合特征（包含两层分析结果）
        combined_feature = self._build_combined_feature(cognition_state, emotion_analysis)

        # 步骤7：分类协同模式
        dominant, mode = self.classify_synergy_mode(e_aligned, c_aligned, synergy_score)

        # 步骤8：收集认知模块的元数据（如果启用）
        cognition_metadata = {}
        if self.enable_cognition_metadata:
            cognition_metadata = self._extract_cognition_metadata(member_pairs)

        # 步骤9：生成建议提示（基于分层分析，现在利用更多认知信息）
        recommendation_hint = self._generate_contextual_hint(cognition_state, emotion_analysis)

        # 步骤10：打包结果
        result = SynergyResult(
            combined_feature=combined_feature,
            synergy_score=synergy_score,
            dominant_dimension=dominant,
            synergy_mode=mode,
            emotion_feature=e_aligned,
            cognition_feature=c_aligned,
            recommendation_hint=recommendation_hint,
            metadata={
                "method": "layered_reasoning",
                "cognition_first": self.cognition_first,
                "cognition_source": self.cognition_source,
                "cognition_phase": cognition_state.phase_description,
                "cognitive_load": cognition_state.cognitive_load,
                "understanding_level": cognition_state.understanding_level,
                "expected_emotion": cognition_state.expected_emotion,
                "emotion_adaptation": emotion_analysis["is_adaptive"],
                "adaptation_score": emotion_analysis["adaptation_score"],
                "reasoning_path": (
                    "cognition → emotion" if self.cognition_first
                    else "emotion → cognition"
                ),
                # === 新增的认知模块元数据 ===
                "attention_score": cognition_state.attention_score,
                "confusion_level": cognition_state.confusion_level,
                "cognitive_flexibility": cognition_state.cognitive_flexibility,
                "source_engine": cognition_state.source_engine,
                "knowledge_nodes": cognition_state.knowledge_nodes,
                "external_cognition_available": _COGNITION_MODULE_AVAILABLE,
                **cognition_metadata,  # 展开外部认知元数据
            },
        )

        return result

    def compute_synergy_score(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> float:
        """计算情绪与认知的协同度得分

        在分层推理中，协同度衡量的是：
        "情绪在当前认知状态下是否适应"

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            float: 协同度得分，范围 [-1, 1]
        """
        # 对齐特征
        e_aligned, c_aligned = self._align_features(emotion_features, cognition_features)

        # 分析认知状态（无 member_pairs 时使用内部方法）
        cognition_state = self._analyze_cognition_state(c_aligned, None)

        # 评估情绪适应性
        emotion_analysis = self._analyze_emotion_in_context(e_aligned, cognition_state)

        return emotion_analysis["adaptation_score"]

    def classify_synergy_mode(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
        synergy_score: float,
    ) -> Tuple[DominantDimension, SynergyMode]:
        """分类协同模式

        在分层推理中，模式分类结合了分层顺序。

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

        # 结合协同度和分层顺序判断模式
        if synergy_score > 0.5:
            mode = SynergyMode.HARMONIOUS
        elif synergy_score > 0.0:
            if self.cognition_first:
                # 认知优先模式：轻微不协调由情绪引起
                mode = SynergyMode.EMOTIONAL_OVERWHELM
            else:
                mode = SynergyMode.COGNITIVE_OVERLOAD
        elif synergy_score > -0.5:
            mode = SynergyMode.CONFLICTING
        else:
            mode = SynergyMode.DISENGAGED

        return dominant, mode

    def _analyze_cognition_state(
        self,
        cognition_features: np.ndarray,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> CognitionState:
        """分析认知状态（已更新：支持外部认知模块）

        现在支持两种模式：
        1. 如果 cognition_source 为 "external" 且 member_pairs 中包含外部 CognitionState 对象，
           则直接使用该对象创建认知状态
        2. 否则使用原有的内部启发式方法

        Args:
            cognition_features: 认知特征向量
            member_pairs: 成员级配对数据（可能包含外部 CognitionState 对象）

        Returns:
            CognitionState: 认知状态分析结果
        """
        # 尝试从 member_pairs 中提取外部认知状态（新增）
        external_state = self._try_extract_external_state(member_pairs)

        if external_state is not None and self.cognition_source == "external":
            # 使用外部认知模块的状态直接构建内部 CognitionState
            return self._convert_external_to_internal(external_state)

        # 回退到内部启发式方法
        return self._internal_analyze(cognition_features)

    def _try_extract_external_state(
        self,
        member_pairs: Optional[List[EmotionCognitionPair]] = None,
    ) -> Optional[Any]:
        """尝试从 member_pairs 中提取外部 CognitionState 对象（新增）

        Args:
            member_pairs: 成员级配对数据

        Returns:
            Optional[Any]: 如果找到外部 CognitionState 对象，返回第一个有效的；否则返回 None
        """
        if not member_pairs:
            return None

        for pair in member_pairs:
            cs = pair.cognition_state
            if cs is not None and hasattr(cs, 'source_engine') and hasattr(cs, 'cognitive_load'):
                # 这是一个来自新认知模块的 CognitionState 对象
                return cs

        return None

    def _convert_external_to_internal(
        self,
        external_state: Any,
    ) -> CognitionState:
        """将外部 CognitionState 对象转换为内部 CognitionState（新增）

        这是适配层的核心方法，负责将 tender.cognition 模块的 CognitionState
        转换为 layered_reasoning.py 内部使用的 CognitionState。

        Args:
            external_state: 来自 tender.cognition 的 CognitionState 对象

        Returns:
            CognitionState: 内部使用的认知状态
        """
        # 获取认知阶段的描述字符串
        try:
            phase_desc = external_state.cognitive_phase.value
        except AttributeError:
            phase_desc = str(external_state.cognitive_phase)

        # 获取参与类型的描述字符串
        try:
            engagement_str = external_state.engagement_type.value
        except AttributeError:
            try:
                engagement_str = str(external_state.engagement_type)
            except AttributeError:
                engagement_str = "moderate"

        # 生成期望情绪描述
        expected_template = self._adaptive_emotion.get(
            phase_desc, {"valence": 0.5, "arousal": 0.5}
        )
        expected_emotion = self._describe_expected_emotion(expected_template)

        # 构建内部 CognitionState
        state = CognitionState(
            member_id=getattr(external_state, 'member_id', "__group__"),
            cognitive_load=float(external_state.cognitive_load),
            understanding_level=float(external_state.understanding_level),
            engagement_type=engagement_str,
            phase_description=phase_desc,
            expected_emotion=expected_emotion,
            confidence=float(getattr(external_state, 'phase_confidence', 0.5)),
            # === 新增字段的映射 ===
            attention_score=float(getattr(external_state, 'attention_score', 0.5)),
            confusion_level=float(getattr(external_state, 'confusion_level', 0.3)),
            cognitive_flexibility=float(getattr(external_state, 'cognitive_flexibility', 0.5)),
            source_engine=getattr(external_state, 'source_engine', 'external'),
            external_state=external_state,
            knowledge_nodes=list(getattr(external_state, 'knowledge_nodes', [])),
            timestamp=float(getattr(external_state, 'timestamp', 0.0)),
        )

        return state

    def _internal_analyze(
        self,
        cognition_features: np.ndarray,
    ) -> CognitionState:
        """内部启发式分析认知状态（原有逻辑，提取为独立方法）

        Args:
            cognition_features: 认知特征向量

        Returns:
            CognitionState: 认知状态分析结果
        """
        c_flat = cognition_features.flatten()

        # 计算认知负荷（特征向量的方差和幅度）
        cognitive_load = float(np.clip(
            (np.var(c_flat) + np.linalg.norm(c_flat) * 0.1) * 2.0,
            0.0, 1.0
        ))

        # 计算理解水平（特征向量的稳定性）
        mean_abs = float(np.clip(np.mean(np.abs(c_flat)) * 3.0, 0.0, 1.0))
        understanding_level = mean_abs

        # 判断认知阶段
        phase = self._classify_cognition_phase(cognitive_load, understanding_level)

        # 获取期望情绪
        expected_emotion_template = self._adaptive_emotion.get(
            phase, {"valence": 0.5, "arousal": 0.5}
        )

        # 构建情感描述
        expected_desc = self._describe_expected_emotion(expected_emotion_template)

        # 计算参与类型
        engagement_type = self._determine_engagement_type(cognitive_load)

        # 计算注意力得分（基于认知负荷和理解的组合）
        attention_score = float(np.clip(
            1.0 - cognitive_load * 0.3 + understanding_level * 0.2,
            0.0, 1.0
        ))

        # 计算困惑水平（高认知负荷 + 低理解 = 高困惑）
        confusion_level = float(np.clip(
            cognitive_load * (1.0 - understanding_level) * 1.5,
            0.0, 1.0
        ))

        # 计算认知灵活性（中高认知负荷 + 中高理解 = 高灵活性）
        cognitive_flexibility = float(np.clip(
            (cognitive_load + understanding_level) * 0.6,
            0.0, 1.0
        ))

        # 构建认知状态
        state = CognitionState(
            cognitive_load=cognitive_load,
            understanding_level=understanding_level,
            engagement_type=engagement_type,
            phase_description=phase,
            expected_emotion=expected_desc,
            confidence=float(np.clip(1.0 - cognitive_load * 0.3, 0.0, 1.0)),
            # === 新增字段 ===
            attention_score=attention_score,
            confusion_level=confusion_level,
            cognitive_flexibility=cognitive_flexibility,
            source_engine="internal",
        )

        return state

    def _classify_cognition_phase(
        self,
        cognitive_load: float,
        understanding_level: float,
    ) -> str:
        """分类认知阶段

        基于认知负荷和理解水平的组合判断当前阶段。

        Args:
            cognitive_load: 认知负荷 (0-1)
            understanding_level: 理解水平 (0-1)

        Returns:
            str: 认知阶段描述
        """
        if cognitive_load > 0.6:
            if understanding_level < 0.4:
                return "疲劳期"
            else:
                return "核心理解"
        else:
            if understanding_level < 0.3:
                return "前期探索"
            elif understanding_level > 0.7:
                return "精通期"
            else:
                return "应用巩固"

    def _determine_engagement_type(self, cognitive_load: float) -> str:
        """确定参与类型

        Args:
            cognitive_load: 认知负荷水平

        Returns:
            str: 参与类型描述
        """
        if cognitive_load > 0.7:
            return "intense"       # 高强度参与
        elif cognitive_load > 0.4:
            return "focused"       # 专注参与
        elif cognitive_load > 0.2:
            return "moderate"      # 中度参与
        else:
            return "passive"       # 被动参与

    def _describe_expected_emotion(
        self,
        template: Dict[str, float],
    ) -> str:
        """描述认知状态下期望的情绪模式

        Args:
            template: 期望情绪模板 {"valence": float, "arousal": float}

        Returns:
            str: 情绪模式描述
        """
        v = template["valence"]
        a = template["arousal"]

        if v > 0.5 and a > 0.5:
            return "兴奋/积极"
        elif v > 0.5 and a <= 0.5:
            return "满足/愉悦"
        elif v <= 0.5 and a > 0.5:
            return "紧张/焦虑"
        elif v <= 0.5 and a <= 0.5:
            return "疲倦/低落"
        else:
            return "中性"

    def _analyze_emotion_in_context(
        self,
        emotion_features: np.ndarray,
        cognition_state: CognitionState,
    ) -> Dict[str, Any]:
        """在认知状态下分析情绪适应性

        判断当前情绪是否适应于当前认知状态。
        适应性是指：情绪与认知状态相匹配，是认知过程的自然反馈。

        新增：利用 cognition_state 中的 attention_score 和 confusion_level
        来更精确地判断情绪适应性。

        Args:
            emotion_features: 情绪特征向量
            cognition_state: 认知状态分析结果

        Returns:
            Dict: 包含适应性评估的字典
                - is_adaptive: bool，是否适应
                - adaptation_score: float，适应性评分 (-1 to 1)
                - deviation_reason: str，偏离原因
                - observed_valence: float，观察到的愉悦度
                - observed_arousal: float，观察到的唤醒度
        """
        e_flat = emotion_features.flatten()

        # 从特征中提取情绪模式
        observed_valence = float(np.clip(np.mean(e_flat) * 2.0, -1.0, 1.0))
        observed_arousal = float(np.clip(np.var(e_flat) * 3.0, 0.0, 1.0))

        # 获取期望情绪模板
        phase = cognition_state.phase_description
        expected = self._adaptive_emotion.get(
            phase, {"valence": 0.5, "arousal": 0.5}
        )

        # 计算偏离程度
        valence_diff = abs(observed_valence - expected["valence"])
        arousal_diff = abs(observed_arousal - expected["arousal"])

        # === 新增：考虑困惑水平和注意力得分的调节 ===
        # 如果困惑水平高，期望情绪的容许范围更宽
        confusion = cognition_state.confusion_level
        attention = cognition_state.attention_score

        # 困惑调节因子：困惑程度越高，容许的偏离越大
        confusion_factor = 1.0 + confusion * 0.5
        # 注意力调节因子：注意力越高，期望的契合度越高
        attention_factor = 1.0 + (1.0 - attention) * 0.3

        # 计算调节后的适应性评分
        deviation = ((valence_diff + arousal_diff) / 2.0) * confusion_factor * attention_factor
        adaptation_score = float(np.clip(1.0 - deviation * 2.0, -1.0, 1.0))

        # 判断是否适应
        is_adaptive = adaptation_score > 0.3

        # 生成偏离原因
        if not is_adaptive:
            if valence_diff > arousal_diff:
                deviation_reason = (
                    f"愉悦度偏离期望 (期望={expected['valence']:.2f}, "
                    f"观测={observed_valence:.2f})"
                )
            else:
                deviation_reason = (
                    f"唤醒度偏离期望 (期望={expected['arousal']:.2f}, "
                    f"观测={observed_arousal:.2f})"
                )

            # 新增：结合认知状态提供更详细的偏离原因
            if confusion > 0.6:
                deviation_reason += "，伴随较高困惑水平"
            if attention < 0.3:
                deviation_reason += "，注意力集中程度较低"
        else:
            deviation_reason = "情绪基本适应于当前认知状态"

        return {
            "is_adaptive": is_adaptive,
            "adaptation_score": adaptation_score,
            "deviation_reason": deviation_reason,
            "observed_valence": observed_valence,
            "observed_arousal": observed_arousal,
        }

    def _build_combined_feature(
        self,
        cognition_state: CognitionState,
        emotion_analysis: Dict[str, Any],
    ) -> np.ndarray:
        """构建综合特征向量（已更新）

        将分层推理的状态信息编码为一个固定维度的特征向量。
        新增了对 attention_score, confusion_level, cognitive_flexibility 的编码。

        Args:
            cognition_state: 认知状态
            emotion_analysis: 情绪分析结果

        Returns:
            np.ndarray: 综合特征向量 (output_dim,)
        """
        # 构建分层特征向量
        d = self.output_dim

        # 基础向量：包含认知状态和情绪适应性信息
        base = np.zeros(d)

        # 编码认知阶段 (one-hot 风格)
        phase_idx = (
            self._cognition_phases.index(cognition_state.phase_description)
            if cognition_state.phase_description in self._cognition_phases
            else 0
        )
        base[0] = phase_idx / len(self._cognition_phases)  # 归一化阶段索引

        # 编码认知负荷和理解水平
        base[1] = cognition_state.cognitive_load
        base[2] = cognition_state.understanding_level

        # 编码情绪适应性
        base[3] = emotion_analysis["adaptation_score"]
        base[4] = 1.0 if emotion_analysis["is_adaptive"] else 0.0

        # 编码观察到的情绪
        base[5] = emotion_analysis["observed_valence"]
        base[6] = emotion_analysis["observed_arousal"]

        # 编码认知置信度
        base[7] = cognition_state.confidence

        # === 新增：编码认知模块的额外维度 ===
        # base[8]: 注意力集中程度
        base[8] = cognition_state.attention_score
        # base[9]: 困惑水平
        base[9] = cognition_state.confusion_level
        # base[10]: 认知灵活性
        base[10] = cognition_state.cognitive_flexibility

        # 填充剩余维度（使用零，或小噪声）
        if d > 11:
            # 使用小噪声填充以保持数值稳定性
            base[11:] = np.random.randn(d - 11) * 0.01

        return base

    def _align_features(
        self,
        emotion_features: np.ndarray,
        cognition_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对齐特征

        确保两个特征向量可以进行比较操作。

        Args:
            emotion_features: 情绪特征向量
            cognition_features: 认知特征向量

        Returns:
            Tuple[np.ndarray, np.ndarray]: 对齐后的特征向量
        """
        e_flat = emotion_features.flatten()
        c_flat = cognition_features.flatten()

        # 对齐到较短维度
        aligned_dim = min(len(e_flat), len(c_flat))

        return e_flat[:aligned_dim], c_flat[:aligned_dim]

    def _generate_contextual_hint(
        self,
        cognition_state: CognitionState,
        emotion_analysis: Dict[str, Any],
    ) -> str:
        """生成上下文感知的建议提示（已更新）

        基于分层分析的认知状态和情绪适应性生成建议。
        新增了对 attention_score, confusion_level 等信息的利用。

        Args:
            cognition_state: 认知状态
            emotion_analysis: 情绪分析结果

        Returns:
            str: 建议提示文本
        """
        phase = cognition_state.phase_description
        load = cognition_state.cognitive_load
        understanding = cognition_state.understanding_level
        is_adaptive = emotion_analysis["is_adaptive"]
        deviation_reason = emotion_analysis["deviation_reason"]

        # === 新增：获取额外的认知指标 ===
        attention = cognition_state.attention_score
        confusion = cognition_state.confusion_level
        flexibility = cognition_state.cognitive_flexibility

        # 基于认知阶段生成基础提示
        phase_hints = {
            "前期探索": "群体处于前期探索阶段，多数成员正在接触新知识。",
            "核心理解": "群体进入核心理解阶段，成员正在消化核心概念。",
            "应用巩固": "群体处于应用巩固阶段，成员正在将知识付诸实践。",
            "疲劳期": "群体出现认知疲劳迹象，成员可能难以继续高效学习。",
            "精通期": "群体已达到精通水平，成员对当前内容有良好掌握。",
        }

        # 生成负荷描述
        if load > 0.7:
            load_desc = "认知负荷较高，"
        elif load > 0.4:
            load_desc = "认知负荷适中，"
        else:
            load_desc = "认知负荷较低，"

        # 生成理解描述
        if understanding > 0.7:
            understanding_desc = "理解程度良好。"
        elif understanding > 0.4:
            understanding_desc = "理解程度中等。"
        else:
            understanding_desc = "理解程度偏低。"

        # === 新增：利用 attention, confusion, flexibility 生成更精细的描述 ===
        attention_desc = ""
        if attention < 0.3:
            attention_desc = "注意力集中程度较低，可能存在分心现象。"
        elif attention < 0.5:
            attention_desc = "注意力集中程度一般。"
        else:
            attention_desc = "注意力较为集中。"

        confusion_desc = ""
        if confusion > 0.6:
            confusion_desc = "困惑水平较高，成员可能感到理解困难。"
        elif confusion > 0.4:
            confusion_desc = "存在一定程度的困惑。"
        else:
            confusion_desc = "困惑水平较低，整体思路清晰。"

        flexibility_desc = ""
        if flexibility > 0.7:
            flexibility_desc = "认知灵活性高，成员能够灵活切换思路。"
        elif flexibility < 0.3:
            flexibility_desc = "认知灵活性较低，可能存在思维定势。"

        # 生成情绪评估
        if is_adaptive:
            emotion_desc = "当前情绪状态适应于认知阶段。"
        else:
            emotion_desc = f"情绪状态与认知阶段不匹配: {deviation_reason}。"

        # === 新增：利用更丰富的认知指标生成更精准的建议 ===
        if not is_adaptive and load > 0.6:
            suggestion = "建议暂停或降低难度，先处理情绪再继续认知任务。"
        elif not is_adaptive:
            suggestion = "建议通过互动或讨论调节情绪氛围。"
        elif load > 0.7:
            if attention < 0.3:
                suggestion = "认知负荷较高且注意力不足，建议提供短暂休息或切换活动类型。"
            else:
                suggestion = "建议提供阶段性总结或休息提醒。"
        elif load < 0.3 and understanding < 0.3:
            if confusion > 0.5:
                suggestion = "理解程度偏低且困惑水平较高，建议提供更清晰的讲解或示例。"
            else:
                suggestion = "建议增加互动或挑战性任务以提升参与度。"
        elif confusion > 0.6:
            suggestion = "困惑水平较高，建议通过问答或小组讨论澄清疑难。"
        elif attention < 0.3:
            suggestion = "注意力集中程度较低，建议引入新话题或互动元素。"
        else:
            suggestion = "建议维持当前节奏。"

        # 组合提示
        base_hint = phase_hints.get(phase, f"当前认知阶段: {phase}。")
        hint = (
            f"{base_hint} "
            f"{load_desc}{understanding_desc} "
            f"{attention_desc} "
            f"{confusion_desc} "
            f"{flexibility_desc} "
            f"{emotion_desc} "
            f"{suggestion}"
        )

        return hint
