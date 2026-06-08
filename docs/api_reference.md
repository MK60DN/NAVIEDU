# API 参考文档

> **Tender  — Embracing Heterogeneity**
>
> 版本：1.0.0 | 最后更新：2026-06-05

本文档提供了 Tender 框架的核心类和函数的 API 参考。按模块组织结构，分别列出每个模块的主要公开接口。

---

## 目录

- [1. 管道入口 (pipeline)](#1-管道入口-pipeline)
- [2. 情绪向量化 (emotion_vectorizer)](#2-情绪向量化-emotion_vectorizer)
- [3. 空间拓扑分析 (topology_analysis)](#3-空间拓扑分析-topology_analysis)
- [4. 时间因果分析 (causal_analysis)](#4-时间因果分析-causal_analysis)
- [5. 时空融合 (fusion)](#5-时空融合-fusion)
- [6. 认知状态分析 (cognition)](#6-认知状态分析-cognition)
- [7. 情绪-认知协同 (synergy)](#7-情绪-认知协同-synergy)
- [8. 策略推理 (strategy)](#8-策略推理-strategy)
- [9. 异质性分析 (heterogeneity)](#9-异质性分析-heterogeneity)
- [10. 不匹配检测 (mismatch)](#10-不匹配检测-mismatch)
- [11. 可视化 (visualization)](#11-可视化-visualization)
- [12. 配置与通用模块 (config, common)](#12-配置与通用模块-config-common)

---

## 1. 管道入口 (pipeline)

### `tender.pipeline.TenderPipeline`

Tender 框架的主管道编排器。加载配置，初始化各分析模块，依次执行分析管道。

```python
class TenderPipeline:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """初始化管道。

        Args:
            config: 配置文件路径（字符串）或配置字典。
        """
        pass

    def analyze_window(self, messages: Dict[str, List[Dict[str, Any]]]) -> PipelineResult:
        """分析单个时间窗口。

        Args:
            messages: 时间窗口内的消息，格式为 {member_id: [message_dict, ...]}。
                      每个 message_dict 至少包含 'content' 和 'timestamp' 字段。

        Returns:
            PipelineResult: 包含所有分析步骤的结果。
        """
        pass

    def analyze_multi_windows(self, windows: List[Dict[str, List[Dict[str, Any]]]]) -> List[PipelineResult]:
        """分析多个时间窗口。

        Args:
            windows: 时间窗口列表。

        Returns:
            List[PipelineResult]: 每个窗口的分析结果。
        """
        pass

    @property
    def config(self) -> Dict[str, Any]:
        """获取当前管道配置。"""
        pass

    @property
    def history(self) -> Dict[str, Any]:
        """获取管道运行历史。"""
        pass
```

### `tender.pipeline.PipelineResult`

管道的完整分析输出，包含所有分析步骤的结果。

```python
@dataclass
class PipelineResult:
    window_timestamp: float                         # 窗口时间戳
    emotion_vectors: Dict[str, List[EmotionVector]]  # 情绪向量结果
    topology_result: TopologyResult                  # 拓扑分析结果
    causal_result: CausalResult                      # 因果分析结果
    cognition_states: Dict[str, CognitionState]      # 认知分析结果 (🆕)
    fusion_result: FusionResult                      # 融合结果
    mismatch_metrics: Dict[str, MismatchMetrics]     # 不匹配检测结果 (🆕)
    heterogeneity_metrics: HeterogeneityMetrics       # 异质性分析结果 (🆕)
    synergy_result: SynergyResult                    # 协同分析结果 (🆕)
    strategy_result: StrategyResult                  # 策略推理结果
    final_strategies: List[StrategyResult]            # 异质性协调后的最终策略 (🆕)
```

### `tender.pipeline.load_config`

```python
def load_config(path: str) -> Dict[str, Any]:
    """加载并验证 YAML 配置文件。

    Args:
        path: 配置文件路径。

    Returns:
        Dict[str, Any]: 验证通过的配置字典。

    Raises:
        ConfigValidationError: 如果配置文件存在或缺少必需字段。
    """
    pass
```

---

## 2. 情绪向量化 (emotion_vectorizer)

### `tender.emotion_vectorizer.EmotionVector`

三维情绪向量。

```python
@dataclass
class EmotionVector:
    valence: float          # 愉悦度 [-1, 1]
    arousal: float          # 唤醒度 [0, 1]
    focus: float           # 专注度 [0, 1]
    confidence: float       # 置信度 [0, 1]
    timestamp: float        # 时间戳
    source: str             # 来源引擎
    metadata: Dict          # 额外元数据

    def to_array(self) -> np.ndarray:
        """转换为 numpy 数组 [valence, arousal, focus]。”""
        pass
```

### `tender.emotion_vectorizer.NeuroSymbolicVectorizer`

默认的情绪向量化引擎，结合 LLM 和符号规则。

```python
class NeuroSymbolicVectorizer(BaseEmotionVectorizer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def vectorize(
        self,
        messages: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[EmotionVector]]:
        """批量向量化。

        Args:
            messages: {member_id: [message_dict, ...]}

        Returns:
            Dict[str, List[EmotionVector]]: {member_id: [EmotionVector, ...]}
        """
        pass
```

### `tender.emotion_vectorizer.MultimodalVectorizer`

多模态融合的情绪向量化引擎。

```python
class MultimodalVectorizer(BaseEmotionVectorizer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def vectorize(self, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[EmotionVector]]:
        pass
```

---

## 3. 空间拓扑分析 (topology_analysis)

### `tender.topology_analysis.TopologyResult`

空间拓扑分析的结果。

```python
@dataclass
class TopologyResult:
    n_clusters: int                     # 情绪簇数量
    cluster_labels: np.ndarray            # 每个成员的簇标签
    ring_exists: bool                    # 是否存在环状结构
    outlier_ratio: float                 # 离群点比例
    outlier_scores: Dict[str, float]     # 每个成员的离群得分
    global_centroid: np.ndarray          # 全局重心
    point_cloud: np.ndarray             # 情绪点云
    trajectory: np.ndarray              # 群体重心时间序列
    edges: List[Tuple[str, str, float]]  # 拓扑边
```

### `tender.topology_analysis.PersistentLaplacianAnalyzer`

默认空间拓扑分析引擎。

```python
class PersistentLaplacianAnalyzer(BaseTopologyAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def analyze(self, emotion_vectors: Dict[str, List[EmotionVector]]) -> TopologyResult:
        pass
```

### `tender.topology_analysis.TopologicalGradientFlowAnalyzer`

替代空间拓扑分析引擎。

```python
class TopologicalGradientFlowAnalyzer(BaseTopologyAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def analyze(self, emotion_vectors: Dict[str, List[EmotionVector]]) -> TopologyResult:
        pass
```

---

## 4. 时间因果分析 (causal_analysis)

### `tender.causal_analysis.CausalEdge`

因果边。

```python
@dataclass
class CausalEdge:
    source: str      # 原因成员
    target: str      # 结果成员
    strength: float  # 因果强度 [0, 1]
    lag: int         # 时间滞后
```

### `tender.causal_analysis.CausalResult`

时间因果分析的结果。

```python
@dataclass
class CausalResult:
    causal_edges: List[CausalEdge]    # 因果边列表
    in_degrees: Dict[str, int]        # 入度
    out_degrees: Dict[str, int]       # 出度
    super_spreaders: List[str]        # 超级传播者
    causal_density: float             # 因果网络密度
```

### `tender.causal_analysis.ConvergentCrossMappingAnalyzer`

默认因果分析引擎。

```python
class ConvergentCrossMappingAnalyzer(BaseCausalAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def analyze(
        self,
        emotion_vectors: Dict[str, List[EmotionVector]],
        history_window: int = 5
    ) -> CausalResult:
        pass
```

---

## 5. 时空融合 (fusion)

### `tender.fusion.FusionResult`

时空融合的结果。

```python
@dataclass
class FusionResult:
    fused_features: np.ndarray           # 融合特征向量
    health_index: float                  # 群体健康度
    forecast: np.ndarray                 # 下一窗口预测
    dynamic_graph: nx.DiGraph            # 动态因果拓扑图
```

### `tender.fusion.DCTGNNEngine`

默认融合引擎，基于图神经网络。

```python
class DCTGNNEngine(BaseFusionEngine):
    def __init__(self, config: Dict[str, Any]):
        pass

    def fuse(
        self,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        emotion_vectors: Dict[str, List[EmotionVector]]
    ) -> FusionResult:
        pass
```

---

## 6. 认知状态分析 (cognition)

### `tender.cognition.CognitionState`

结构化认知状态。

```python
@dataclass
class CognitionState:
    cognitive_load: float           # 认知负荷 [0, 1]
    understanding_level: float      # 理解水平 [0, 1]
    confusion_level: float          # 困惑水平 [0, 1]
    attention_score: float          # 注意力 [0, 1]
    cognitive_flexibility: float    # 认知灵活性 [0, 1]
    phase_confidence: float         # 阶段置信度 [0, 1]
    cognitive_phase: CognitivePhase # 认知阶段
```

### `tender.cognition.HybridStateAnalyzer`

默认认知分析引擎，融合 NLP、行为特征和知识图谱。

```python
class HybridStateAnalyzer(BaseCognitionAnalyzer):
    def __init__(self, config: Dict[str, Any]):
        pass

    def analyze(
        self,
        messages: Dict[str, List[Dict[str, Any]]],
        emotion_vectors: Dict[str, List[EmotionVector]]
    ) -> Dict[str, CognitionState]:
        pass
```

---

## 7. 情绪-认知协同 (synergy)

### `tender.synergy.SynergyResult`

协同分析结果。

```python
@dataclass
class SynergyResult:
    combined_feature: np.ndarray   # 融合特征向量
    synergy_score: float           # 协同度 [0, 1]
    dominant_dimension: str        # 主导维度
    synergy_mode: SynergyMode      # 协同模式
    adaptation_score: float        # 情绪适应度 [0, 1]
    recommendation: str            # 建议
```

### `tender.synergy.LayeredReasoningEngine`

默认协同引擎，基于"认知优先"的分层推理。

```python
class LayeredReasoningEngine(BaseSynergyEngine):
    def __init__(self, config: Dict[str, Any]):
        pass

    def fuse(
        self,
        emotion_features: np.ndarray,
        cognition_states: Dict[str, CognitionState],
        member_pairs: List[Tuple[str, str]]
    ) -> SynergyResult:
        pass
```

---

## 8. 策略推理 (strategy)

### `tender.strategy.StrategyResult`

策略推理结果。

```python
@dataclass
class StrategyResult:
    risk_level: RiskLevel          # 风险等级
    selected_action: int           # 策略 ID
    target_members: List[str]      # 目标成员
    confidence: float              # 置信度
    rationale: str                 # 决策理由
    specific_actions: List[str]    # 具体动作列表
```

### `tender.strategy.CausalRLEngine`

默认策略推理引擎，基于因果强化学习。

```python
class CausalRLEngine(BaseStrategyEngine):
    def __init__(self, config: Dict[str, Any]):
        pass

    def reason(
        self,
        fusion_result: FusionResult,
        synergy_result: SynergyResult,
        mismatch_metrics: Dict[str, MismatchMetrics]
    ) -> StrategyResult:
        pass

    def train(self, batch):
        """训练 DQN 网络。"""
        pass
```

### `tender.strategy.HeterogeneityCoordinationLayer` (🆕)

异质性协调层，替代原共识化过滤层。

```python
class HeterogeneityCoordinationLayer:
    def __init__(self, config: Dict[str, Any]):
        pass

    def coordinate(
        self,
        base_strategies: List[StrategyResult],
        heterogeneity_metrics: HeterogeneityMetrics,
        mismatch_metrics: Dict[str, MismatchMetrics]
    ) -> List[StrategyResult]:
        """异质性协调后的最终策略列表。"""
        pass
```

---

## 9. 异质性分析 (heterogeneity) (🆕)

### `tender.heterogeneity.HeterogeneityMetrics`

群体异质性度量。

```python
@dataclass
class HeterogeneityMetrics:
    topological_richness: float     # 拓扑丰富度
    loop_strength: float            # 环状矛盾强度
    causal_fragmentation: float     # 因果碎片化
    component_separation: float     # 组件分离度
    temporal_asynchrony: float      # 时间异步
    linguistic_divergence: float    # 语言离散
    participation_gini: float       # 参与度基尼系数
    cluster_ids: List[int]          # 簇 ID 列表
    cluster_members: Dict[int, List[str]]  # {簇 ID: [成员 ID]}
    outlier_types: Dict[str, IsolateType]  # {成员 ID: 离群类型}
```

### `tender.heterogeneity.TopologicalDisconnectAnalyzer`

拓扑脱离分析器。

```python
class TopologicalDisconnectAnalyzer(BaseHeterogeneityAnalyzer):
    def compute_disconnect(
        self,
        member_id: str,
        topology_result: TopologyResult,
        causal_result: CausalResult,
        cognition_states: Dict[str, CognitionState]
    ) -> DisconnectScore:
        pass
```

### `tender.heterogeneity.IsolateAnalyzer`

离群者类型分类器。

```python
class IsolateAnalyzer(BaseHeterogeneityAnalyzer):
    def classify(
        self,
        disconnect_score: DisconnectScore,
        personal_profile: PersonalProfile,
        group_profile: TopologyResult
    ) -> IsolateType:
        """返回离群类型。"""
        pass
```

---

## 10. 不匹配检测 (mismatch) (🆕)

### `tender.mismatch.MismatchMetrics`

个人-群体不匹配度量。

```python
@dataclass
class MismatchMetrics:
    structural_distance: float       # 拓扑不匹配距离
    dynamic_distance: float          # 动态不匹配距离
    personal_self_consistency: float # 个人自洽性
```

### `tender.mismatch.TopologicalMismatchDetector`

拓扑不匹配检测器。

```python
class TopologicalMismatchDetector(BaseMismatchDetector):
    def compute_distance(
        self,
        personal_point_cloud: np.ndarray,
        group_point_cloud: np.ndarray
    ) -> float:
        pass
```

---

## 11. 可视化 (visualization)

### `tender.visualization.HeterogeneityPlotter`

异质性可视化。

```python
class HeterogeneityPlotter:
    def plot_scatter(
        self,
        heterogeneity_metrics: HeterogeneityMetrics,
        topology_result: TopologyResult,
        save_path: str
    ):
        pass
```

---

## 12. 配置与通用模块 (config, common)

### `tender.config.global_config`

```python
class GlobalConfig:
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """加载 YAML 配置并合并默认值。"""
        pass

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """验证配置完整性。"""
        pass
```
