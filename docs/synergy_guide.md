# 情绪-认知协同指南

> **Tender  — Embracing Heterogeneity**
>
> 本指南详细介绍如何使用 `synergy/` 模块将情绪分析结果与认知分析结果进行深度融合。

## 1. 什么是情绪-认知协同？

**情绪-认知协同**是指将群体成员的情绪状态（愉悦度、唤醒度、专注度）与认知状态（认知负荷、理解水平、困惑水平）进行融合分析，回答的核心问题是：

> **"当前的情绪状态是否适应当前的认知阶段？"**

### 为什么需要协同？

| 只看情绪 | 只看认知 | 协同分析 |
|:---|:---|:---|
| "群氛围很差，焦虑度很高" | "理解水平低，认知负荷高" | "焦虑是因为听不懂 → 优先解决认知问题" |
| "情绪很积极，愉悦度高" | "认知负荷很低" | "太轻松可能是无聊 → 增加挑战性" |
| "某个成员情绪激动" | "该成员困惑水平低" | "他的激动可能是兴奋而非焦虑" |

## 2. 四种协同策略

Tender 提供四种不同的协同引擎，适用于不同的场景：

| 策略 | 引擎 | 核心逻辑 | 适用场景 |
|:---|:---|:---|:---|
| **W1: 加权融合** | `weighted_fusion` | `F = αE + βK`，线性加权 | 快速基线，简单场景 |
| **W2: 门控融合** | `gated_fusion` | 根据特征方差动态计算门控权重 | 数据特征变化明显的场景 |
| **W3: 分层推理** | `layered_reasoning` | **先分析认知，再判断情绪是否适应** | 在线教育、培训场景（默认） |
| **W4: 因果协调** | `causal_coordination` | 用 CCM/格兰杰检验检测"情绪↔认知"因果方向 | 学术研究、长期追踪 |

### 2.1 加权融合 (Weighted Fusion)

最简单的协同策略，将情绪特征和认知特征进行线性加权：

```python
combined_feature = emotion_weight * emotion_features + cognition_weight * cognition_features
synergy_score = cosine_similarity(emotion_features, cognition_features)
```

**优点**：速度快，可解释性强。

**缺点**：固定权重无法适应动态变化。

### 2.2 门控融合 (Gated Fusion)

使用门控网络动态计算融合权重：

```python
gate = sigmoid(W * [emotion_features, cognition_features] + b)
combined_feature = gate * emotion_features + (1 - gate) * cognition_features
```

**优点**：能根据输入特征自动调整权重，适应性强。

**缺点**：需要训练，可解释性较低。

### 2.3 分层推理 (Layered Reasoning)

Tender v2.0 的默认协同策略。基于"认知优先"的假设：

```
阶段一：分析认知状态 → 
阶段二：查询该认知状态下的"期望情绪" →
阶段三：比较实际情绪与期望情绪的差异 →
阶段四：输出适应度评分
```

**核心规则（内置）**：

| 认知阶段 | 期望情绪 |
|:---|:---|
| 核心理解阶段 | valence: 0.3, arousal: 0.7, focus: 0.8（适度焦虑+高专注） |
| 应用巩固阶段 | valence: 0.6, arousal: 0.4, focus: 0.7（平和+专注） |
| 前期探索阶段 | valence: 0.5, arousal: 0.6, focus: 0.6（好奇+适度兴奋） |

### 2.4 因果协调 (Causal Coordination)

使用时间因果分析方法（CCM/格兰杰检验）分析情绪与认知之间的因果方向：

- **情绪 → 认知**：改善情绪能提升理解水平
- **认知 → 情绪**：认知负荷过高导致情绪恶化
- **无显著因果**：情绪与认知相对独立

## 3. 快速开始

### 3.1 基础使用

```python
from tender.synergy.layered_reasoning import LayeredReasoningEngine

# 初始化协同引擎
synergy_engine = LayeredReasoningEngine(config["synergy"])

# 执行协同分析
synergy_result = synergy_engine.fuse(
    emotion_features=fusion_result.emotion_features,   # 融合模块输出的情绪特征
    cognition_states=cognition_states,                   # 认知分析输出的认知状态
    member_pairs=member_pairs,                          # 成员配对数据
)

print(synergy_result)
# SynergyResult(
#     combined_feature=[...],    # 32维融合特征向量
#     synergy_score=0.85,        # 协同度 (0-1)
#     dominant_dimension="cognition",  # 主导维度
#     synergy_mode="HARMONIOUS",       # 协同模式
#     adaptation_score=0.78,           # 情绪-认知适应度
#     recommendation="当前情绪与认知状态匹配，建议维持当前节奏。"
# )
```

### 3.2 选择不同的协同策略

```python
# 加权融合
from tender.synergy.weighted_fusion import WeightedFusionEngine

engine = WeightedFusionEngine(config)
result = engine.fuse(
    emotion_features=emotion_features,
    cognition_states=cognition_states,
)

# 门控融合
from tender.synergy.gated_fusion import GatedFusionEngine

engine = GatedFusionEngine(config)
result = engine.fuse(
    emotion_features=emotion_features,
    cognition_states=cognition_states,
)

# 分层推理（默认推荐）
from tender.synergy.layered_reasoning import LayeredReasoningEngine

engine = LayeredReasoningEngine(config)
result = engine.fuse(
    emotion_features=emotion_features,
    cognition_states=cognition_states,
    member_pairs=member_pairs,
)

# 因果协调
from tender.synergy.causal_coordination import CausalCoordinationEngine

engine = CausalCoordinationEngine(config)
result = engine.fuse(
    emotion_time_series=emotion_time_series,
    cognition_time_series=cognition_time_series,
)
```

## 4. 协同模式与干预建议

### 4.1 协同模式

| 模式 | 描述 | 情绪-E | 认知-K | 协同度 | 建议方向 |
|:---|:---|:---|:---|:---|:---|
| HARMONIOUS | 和谐 | 适应 | 正常 | > 0.7 | 维持当前节奏 |
| EMOTIONAL_OVERWHELM | 情绪主导 | 强烈 | 正常 | < 0.5 | 优先情绪调节 |
| COGNITIVE_OVERLOAD | 认知过载 | 正常 | 高负荷 | < 0.5 | 降低难度/提供休息 |
| CONFLICTING | 冲突 | 不匹配 | 不稳定 | < 0.3 | 通过互动调节氛围 |
| DISENGAGED | 脱离 | 平淡 | 低投入 | < 0.3 | 增加互动或挑战 |

### 4.2 适应度评分 (Adaptation Score)

```python
def compute_adaptation(self, emotion, cognition_phase):
    """
    计算情绪对当前认知阶段的适应度。
    
    基于预定义的期望情绪与实际情绪的差异。
    差异越小，适应度越高。
    """
    expected_emotion = self._get_expected_emotion(cognition_phase)
    
    valence_adaptation = 1.0 - abs(emotion.valence - expected_emotion.valence)
    arousal_adaptation = 1.0 - abs(emotion.arousal - expected_emotion.arousal)
    focus_adaptation = 1.0 - abs(emotion.focus - expected_emotion.focus)
    
    return (valence_adaptation + arousal_adaptation + focus_adaptation) / 3.0
```

## 5. 完整示例

```python
from tender.pipeline import TenderPipeline, load_config
from tender.synergy.layered_reasoning import LayeredReasoningEngine
from tender.visualization.synergy_heatmap import SynergyHeatmapPlotter

# 加载配置
config = load_config("config.yaml")

# 运行管道
pipeline = TenderPipeline(config)
result = pipeline.analyze_window(member_messages)

# 初始化协同引擎
synergy_engine = LayeredReasoningEngine(config["synergy"])

# 执行协同分析
synergy_result = synergy_engine.fuse(
    emotion_features=result.fusion_result.emotion_features,
    cognition_states=result.cognition_states,
    member_pairs=member_pairs,
)

# 打印结果
print(f"协同模式: {synergy_result.synergy_mode}")
print(f"协同度: {synergy_result.synergy_score:.2f}")
print(f"主导维度: {synergy_result.dominant_dimension}")
print(f"情绪适应度: {synergy_result.adaptation_score:.2f}")
print(f"建议: {synergy_result.recommendation}")

# 可视化
plotter = SynergyHeatmapPlotter(config["visualization"])
plotter.plot(
    member_ids=member_ids,
    synergy_scores=synergy_result.per_member_adaptation,
    save_path="output/visualizations/synergy_heatmap.html"
)
```

## 6. 配置参数说明

协同模块配置 (`config.yaml`)

```yaml
synergy:
  engine: "layered_reasoning"       # 可选: weighted_fusion | gated_fusion | layered_reasoning | causal_coordination
  emotion_dim: 16                   # 情绪特征维度
  cognition_dim: 16                 # 认知特征维度
  output_dim: 32                    # 融合输出维度
  cognition_source: "internal"      # internal | external（是否使用外部认知模块）
  
  weighted_fusion:
    emotion_weight: 0.5            # 情绪权重
    cognition_weight: 0.5          # 认知权重
    
  gated_fusion:
    gate_hidden_dim: 16            # 门控网络隐藏层维度
    gate_activation: "sigmoid"      # sigmoid | softmax
    
  layered_reasoning:
    priority: "cognition_first"     # cognition_first | emotion_first
    adaptation_thresholds:
      valence: 0.3                  # 愉悦度适应度阈值
      arousal: 0.3                  # 唤醒度适应度阈值
      focus: 0.2                    # 专注度适应度阈值
      
  causal_coordination:
    causal_method: "ccm"            # ccm | granger | pearson
    causal_lag: 1                   # 因果滞后步长
    max_emotion_features: 5         # 最大情绪特征数
    max_cognition_features: 5       # 最大认知特征数
```
