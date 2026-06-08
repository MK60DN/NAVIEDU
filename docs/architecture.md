# 架构设计文档

> **Tender v2.0 — Embracing Heterogeneity**
>
> 版本：2.0.0 | 最后更新：2026-06-05

## 1. 架构总览

Tender 采用 **分层插件架构**（Layered Plugin Architecture），从上到下分为四层：

| 层次 | 名称 | 职责 | 可替换性 |
|:---:|:---|:---|:---:|
| L4 | **应用与可视化层** | CLI 工具、API 接口、可视化输出 | 完全可替换 |
| L3 | **策略与决策层** | 协同分析 → 策略推理 → 异质性协调 → 干预输出 | 引擎级可替换 |
| L2 | **核心分析层** | 情绪向量化 → 拓扑分析 → 因果分析 → 认知分析 → 融合 | 引擎级可替换 |
| L1 | **基础设施层** | 配置管理、数据模型、通用工具、异常处理 | 通用，不可替换 |

### 整体架构图

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  L4: 应用与可视化层                                                         │
│  ┌──────────┐  ┌───────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ CLI 工具 │  │ API 服务  │  │ 可视化引擎      │  │ 自定义应用       │   │
│  │ (tender) │  │ (REST)    │  │ (plotly/pyvis)  │  │ (集成到产品)     │   │
│  └──────────┘  └───────────┘  └─────────────────┘  └──────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│  L3: 策略与决策层                                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐    │
│  │ 情绪-认知协同  │  │ 策略推理      │  │ 异质性协调层               │    │
│  │ (synergy/)     │→ │ (strategy/)   │→ │ (heterogeneity_coordination)│   │
│  │                │  │                │  │ (原共识化过滤层·增强版)     │    │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘    │
│         ↑                     ↑                       ↑                     │
│         │                     │                       │                     │
│  ┌──────┴─────────────────────┴───────────────────────┴──────┐             │
│  │  L2: 核心分析层                                            │             │
│  │  ┌──────────┐ ┌──────────────┐ ┌──────────────────────┐  │             │
│  │  │ 情绪向量 │ │ 空间拓扑分析  │ │ 时间因果分析         │  │             │
│  │  │ vectorizer│→│ topology      │→│ causal               │  │             │
│  │  └──────────┘ └──────────────┘ └──────────────────────┘  │             │
│  │       ↓                                                    │             │
│  │  ┌──────────────────────┐ ┌──────────────────────┐       │             │
│  │  │ 🆕 认知状态分析      │ │ 时空融合             │       │             │
│  │  │ cognition            │→│ fusion               │       │             │
│  │  └──────────────────────┘ └──────────────────────┘       │             │
│  └───────────────────────────────────────────────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│  L1: 基础设施层                                                             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────────────┐    │
│  │ 配置管理   │ │ 数据模型   │ │ 通用工具   │ │ 异常处理             │    │
│  │ (config/)  │ │ (common/)  │ │ (utils)    │ │ (exceptions)         │    │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 新增模块定位

### 2.1 🆕 异质性分析模块 (heterogeneity/)

**定位**：L2 核心分析层的一部分，作为拓扑分析和因果分析的补充分析模块。

**为什么放在 L2？**  
异质性分析直接依赖于拓扑分析和因果分析的输出结果（如距离矩阵、因果图），它是对这些基础分析结果的二次提炼。它不负责生成原始特征，而是负责识别原始特征中的差异模式。

```text
拓扑分析 ──→ [簇标签、环检测、离群得分]
                                ↓
因果分析 ──→ [因果边、出入度、因果密度]  →  异质性分析 → [异质性指标、离群者分类]
                                ↑
认知分析 ──→ [认知负荷、理解水平]
```

### 2.2 不匹配检测模块 (mismatch/)

**定位**：L3 策略与决策层的一部分，位于协同分析之前。

**为什么放在 L3？**  
不匹配检测回答的是 “这个人是否适应当前的社会环境” 这个决策问题。它消费 L2 层的所有分析输出，为 L3 的策略制定提供关键决策变量。

```text
L2 核心分析结果
    ↓
不匹配检测 →  [个人 vs 群体的拓扑距离、动态距离、自洽性]
    ↓
情绪-认知协同 →  [协同度、主导维度、协同模式]
    ↓
策略推理 →  [风险等级、推荐策略]
    ↓
异质性协调 →  [最终策略输出（允许不融入）]
```

## 3. 关键设计决策

### 3.1 引擎化设计

所有核心模块（情绪向量化、拓扑分析、因果分析、融合、认知分析、协同、策略）均支持多引擎切换。引擎切换通过配置文件 `config.yaml` 的 `engine` 字段实现。

```yaml
# 示例：拓扑分析引擎切换
# config.yaml
topology_analysis:
  engine: "persistent_laplacian"  # 可选: topological_gradient_flow
```

```python
# 内部实现（工厂模式）
engine_class = ENGINE_MAP[config["topology_analysis"]["engine"]]
analyzer = engine_class(config["topology_analysis"])
```

**引擎映射表：**

| 模块 | 引擎名 | 实现类 | 文件 |
|:---|:---|:---|:---|
| emotion_vectorizer | neuro_symbolic | NeuroSymbolicVectorizer | neuro_symbolic.py |
| | multimodal | MultimodalVectorizer | multimodal.py |
| topology_analysis | persistent_laplacian | PersistentLaplacianAnalyzer | persistent_laplacian.py |
| | topological_gradient_flow | TopologicalGradientFlowAnalyzer | topological_gradient_flow.py |
| causal_analysis | convergent_cross_mapping | ConvergentCrossMappingAnalyzer | convergent_cross_mapping.py |
| | structural_causal_model | StructuralCausalModelAnalyzer | structural_causal_model.py |
| | pc_lingam | PCLiNGAMAnalyzer | pc_lingam.py |
| fusion | dct_gnn | DCTGNNEngine | dct_gnn.py |
| | neural_temporal_logic | NeuralTemporalLogicEngine | neural_temporal_logic.py |
| cognition | knowledge_state | KnowledgeStateAnalyzer | knowledge_state.py |
| | behavior_state | BehaviorStateAnalyzer | behavior_state.py |
| | hybrid_state | HybridStateAnalyzer | hybrid_state.py |
| | neural_state | NeuralStateAnalyzer | neural_state.py |
| synergy | weighted_fusion | WeightedFusionEngine | weighted_fusion.py |
| | gated_fusion | GatedFusionEngine | gated_fusion.py |
| | layered_reasoning | LayeredReasoningEngine | layered_reasoning.py |
| | causal_coordination | CausalCoordinationEngine | causal_coordination.py |
| strategy | causal_rl | CausalRLStrategyEngine | causal_rl.py |
| | llm_strategist | LLMStrategistEngine | llm_strategist.py |

### 3.2 管道编排

`pipeline/orchestrator.py` 是框架的入口点。它定义了分析管道中各模块的执行顺序。

**单窗口分析流程：**

1. 加载配置 ✅
2. 加载数据 ✅
3. 情绪向量化 → 得到 EmotionVector 字典 ✅
4. 空间拓扑分析 → 得到 TopologyResult ✅
5. 时间因果分析 → 得到 CausalResult ✅
6. 认知状态分析 → 得到 CognitionState 字典 ✅  (🆕)
7. 时空融合 → 得到 FusionResult ✅
8. 不匹配检测 → 得到 MismatchMetrics 字典 ✅ (🆕)
9. 情绪-认知协同 → 得到 SynergyResult ✅  (🆕)
10. 策略推理 → 得到 StrategyResult ✅
11. 异质性协调 → 得到 FinalStrategyList ✅ (🆕)
12. 可视化输出 ✅

**数据流：**

```python
# 管道编排器核心方法（伪代码）
def analyze_window(self, messages):
    # 步骤 1-2: 向量化
    emotion_vectors = self.vectorizer.vectorize(messages)
    
    # 步骤 3: 拓扑分析
    topology_result = self.topology_analyzer.analyze(emotion_vectors)
    
    # 步骤 4: 因果分析
    causal_result = self.causal_analyzer.analyze(emotion_vectors, history_window=5)
    
    # 步骤 5: 认知分析 (🆕)
    cognition_states = self.cognition_analyzer.analyze(messages, emotion_vectors)
    
    # 步骤 6: 融合
    fusion_result = self.fusion_engine.fuse(
        topology_result, causal_result, emotion_vectors
    )
    
    # 步骤 7: 不匹配检测 (🆕)
    mismatch_metrics = self.mismatch_detector.detect(
        personal_profiles=...,
        group_profile=topology_result
    )
    
    # 步骤 8: 协同分析 (🆕)
    synergy_result = self.synergy_engine.fuse(
        fusion_result.emotion_features,
        cognition_states,
        member_pairs=...
    )
    
    # 步骤 9: 策略推理
    strategy_result = self.strategy_engine.reason(
        fusion_result, synergy_result, mismatch_metrics
    )
    
    # 步骤 10: 异质性协调 (🆕)
    final_strategies = self.heterogeneity_coordination.coordinate(
        strategy_result,
        heterogeneity_metrics=...,
        mismatch_metrics=mismatch_metrics
    )
    
    return PipelineResult(
        emotion_vectors=emotion_vectors,
        topology_result=topology_result,
        causal_result=causal_result,
        cognition_states=cognition_states,  # 🆕
        fusion_result=fusion_result,
        mismatch_metrics=mismatch_metrics,  # 🆕
        synergy_result=synergy_result,      # 🆕
        strategy_result=strategy_result,
        final_strategies=final_strategies   # 🆕
    )
```

### 3.3 数据模型

Tender v2.0 定义了以下核心数据类（位于各模块的 `base.py` 或 `*_result.py`）：

| 数据类 | 模块 | 核心字段 |
|:---|:---|:---|
| EmotionVector | emotion_vectorizer | valence, arousal, focus, confidence, timestamp |
| TopologyResult | topology_analysis | n_clusters, ring_exists, outlier_ratio, outlier_scores |
| CausalResult | causal_analysis | causal_edges, in_degrees, out_degrees, super_spreaders |
| FusionResult | fusion | fused_features, health_index, forecast |
| CognitionState 🆕 | cognition | cognitive_load, understanding_level, confusion_level, cognitive_phase |
| SynergyResult 🆕 | synergy | combined_feature, synergy_score, dominant_dimension, synergy_mode |
| HeterogeneityMetrics 🆕 | heterogeneity | topological_richness, causal_fragmentation, component_separation |
| MismatchMetrics 🆕 | mismatch | structural_distance, dynamic_distance, personal_self_consistency |
| StrategyResult | strategy | risk_level, selected_strategy, confidence |

## 4. 配置管理

配置采用层次化合并策略：

1. **默认配置** (`config/defaults.py` 中的 `DEFAULT_CONFIG`)
2. **全局配置文件** (`config.yaml`)
3. **本地配置文件** (`config/local.yaml`, `.gitignore` 忽略)
4. **环境变量覆盖** (`${API_KEY}`)
5. **运行时参数覆盖** (`pipeline = TenderPipeline(config, override={...})`)

## 5. 异常处理

所有模块自定义异常继承自 `TenderException`（位于 `common/exceptions.py`）：

```text
TenderException (基类)
├── ConfigError              # 配置错误
├── DataError                # 数据错误
├── VectorizationError       # 向量化错误
├── TopologyAnalysisError    # 拓扑分析错误
├── CausalAnalysisError      # 因果分析错误
├── CognitionAnalysisError   # 认知分析错误 (🆕)
├── FusionError              # 融合错误
├── SynergyError             # 协同错误 (🆕)
├── HeterogeneityError       # 异质性分析错误 (🆕)
├── MismatchError            # 不匹配检测错误 (🆕)
├── StrategyError            # 策略错误
└── VisualizationError       # 可视化错误
```

## 6. 测试策略

测试遵循与模块结构相同的目录层次。每个模块在 `tests/` 下都有对应的测试文件或目录。

**重点测试覆盖：**

- **单元测试**：每个分析引擎的独立功能
- **集成测试**：全链路管道端到端测试
- **异质性测试** 🆕：测试 `heterogeneity/` 模块的各种边缘情况（空群体、完全同构群体、极度异质群体等）
- **不匹配测试** 🆕：测试 `mismatch/` 模块的各种拓扑距离计算
