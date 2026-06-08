# 异质性分析指南

> **Tender v2.0 — Embracing Heterogeneity**
>
> 本指南详细介绍如何使用 `heterogeneity/` 模块对群体中的异质性进行系统识别与量化。

## 1. 什么是群体异质性？

**群体异质性**是指群体成员在行为模式、认知状态、情绪特征和网络拓扑结构上的差异程度。传统群体分析将异质性视为"噪声"或"问题"，而 Tender v2.0 将其视为群体健康度和多样性的重要指标。

### 异质性的四个维度

| 维度 | 指标 | 意义 |
|:---|:---|:---|
| **拓扑异质性** | 拓扑脱离度、环状结构持久性 | 谁在网络结构上"游离" |
| **因果异质性** | 因果网络碎片化、影响力集中度 | 谁拥有不成比例的影响力 |
| **行为异质性** | 时间异步、参与度不均、语言离散 | 谁的行为模式与众不同 |
| **认知异质性** | 认知负荷差异、理解水平离散 | 谁的知识状态与主流不同 |

## 2. 核心概念

### 2.1 脱离度 (Disconnect Score)

**定义**：衡量一个成员在拓扑结构上偏离群体的程度。基于四个子维度的加权平均：

```
DisconnectScore = w1 * SpaceDisconnect + w2 * CausalDisconnect + w3 * CognitionDisconnect
```

其中各子维度为 0-1 之间的浮点数，分数越高表示脱离程度越大。

### 2.2 离群者类型 (Isolate Type)

根据脱离的成因和成员的自我状态，将离群者分为四种类型：

| 类型 | 特征 | 干预建议 |
|:---|:---|:---|
| **自愿隔离者** | 高脱离度 + 高自洽性 | 不干预，尊重其选择 |
| **被迫排斥者** | 高脱离度 + 低自洽性 | 软性牵线搭桥 |
| **意见领袖偏差者** | 高影响力 + 意见偏离主流 | 保留其视角，鼓励建设性冲突 |
| **纯粹异质性者** | 高统计差异 + 高幸福感 | 拥抱多样性 |

## 3. 快速开始

### 3.1 基础使用

```python
from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer
from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer

# 初始化分析器
disconnect_analyzer = TopologicalDisconnectAnalyzer(config)
isolate_analyzer = IsolateAnalyzer(config)

# 计算脱离度
disconnect_results = disconnect_analyzer.compute_disconnect(
    member_id="user_1",
    topology_result=topology_result,
    causal_result=causal_result,
    cognition_states=cognition_states
)

print(disconnect_results)
# DisconnectScore(
#     total=0.82,
#     space_disconnect=0.90,
#     causal_disconnect=0.75,
#     cognition_disconnect=0.80
# )

# 离群者类型分类
isolate_type = isolate_analyzer.classify(
    disconnect_score=disconnect_results,
    personal_profile=personal_profile,
    group_profile=group_profile
)
print(isolate_type)
# IsolateType.VOLUNTARY_ISOLATE
```

### 3.2 批量分析

```python
# 对所有成员进行脱离度分析
all_disconnect_scores = {}
for member_id in member_ids:
    score = disconnect_analyzer.compute_disconnect(
        member_id=member_id,
        topology_result=topology_result,
        causal_result=causal_result,
        cognition_states=cognition_states
    )
    all_disconnect_scores[member_id] = score

# 按脱离度排序
sorted_scores = sorted(
    all_disconnect_scores.items(),
    key=lambda x: x[1].total,
    reverse=True
)

print("Top 5 脱离度最高的成员：")
for member_id, score in sorted_scores[:5]:
    print(f"  {member_id}: {score.total:.2f} ({score.type.value})")
```

## 4. API 参考

### TopologicalDisconnectAnalyzer

**构造函数参数**：
- `config` (Dict): 包含 space_weight, causal_weight, cognition_weight 等配置参数

**核心方法**：

```python
def compute_disconnect(
    self,
    member_id: str,
    topology_result: TopologyResult,
    causal_result: CausalResult,
    cognition_states: Dict[str, CognitionState]
) -> DisconnectScore:
    """
    计算成员相对于群体拓扑结构的脱离程度。

    Args:
        member_id: 成员 ID
        topology_result: 拓扑分析结果
        causal_result: 因果分析结果
        cognition_states: 认知状态字典

    Returns:
        DisconnectScore: 包含总分和各子维度分数
    """
    pass
```

### LoopDetector

```python
class LoopDetector:
    def detect_loops(
        self,
        topology_result: TopologyResult
    ) -> List[LoopInfo]:
        """
        检测群体中的环状矛盾结构。

        Args:
            topology_result: 拓扑分析结果

        Returns:
            List[LoopInfo]: 环结构信息列表
        """
        pass
```

### CausalFragmentationAnalyzer

```python
class CausalFragmentationAnalyzer:
    def compute_fragmentation(
        self,
        causal_result: CausalResult
    ) -> FragmentMetrics:
        """
        计算因果网络碎片化程度。

        Args:
            causal_result: 因果分析结果

        Returns:
            FragmentMetrics: 包含组件数量、碎片化指数等
        """
        pass
```

### ParticipationGiniAnalyzer

```python
class ParticipationGiniAnalyzer:
    def compute_gini(
        self,
        participation_data: Dict[str, int]
    ) -> float:
        """
        计算参与度分布的基尼系数。

        Args:
            participation_data: {member_id: message_count}

        Returns:
            float: 基尼系数 (0=完全平等, 1=完全不平等)
        """
        pass
```

## 5. 完整示例

```python
from tender.pipeline import TenderPipeline, load_config
from tender.heterogeneity.topology_analysis import TopologicalDisconnectAnalyzer
from tender.heterogeneity.causal_analysis import CausalFragmentationAnalyzer
from tender.heterogeneity.behavior_analysis import (
    TemporalAsynchronyAnalyzer,
    ParticipationGiniAnalyzer
)
from tender.heterogeneity.isolate_analyzer import IsolateAnalyzer

# 加载配置
config = load_config("config.yaml")

# 运行基础分析管道
pipeline = TenderPipeline(config)
result = pipeline.analyze_window(member_messages)

# 异质性分析
heterogeneity_results = {}

# 1. 拓扑脱离分析
disconnect_analyzer = TopologicalDisconnectAnalyzer(config["heterogeneity"])
for member_id in member_ids:
    score = disconnect_analyzer.compute_disconnect(
        member_id=member_id,
        topology_result=result.topology_result,
        causal_result=result.causal_result,
        cognition_states=result.cognition_states
    )
    heterogeneity_results[member_id] = score

# 2. 因果碎片化分析
frag_analyzer = CausalFragmentationAnalyzer(config["heterogeneity"])
fragment_metrics = frag_analyzer.compute_fragmentation(result.causal_result)
print(f"因果网络碎片化指数: {fragment_metrics.fragmentation_index:.2f}")

# 3. 参与度不均分析
gini_analyzer = ParticipationGiniAnalyzer(config["heterogeneity"])
message_counts = {k: len(v) for k, v in member_messages.items()}
gini_coefficient = gini_analyzer.compute_gini(message_counts)
print(f"参与度基尼系数: {gini_coefficient:.2f}")

# 4. 时间异步分析
async_analyzer = TemporalAsynchronyAnalyzer(config["heterogeneity"])
asynchrony_score = async_analyzer.compute_asynchrony(member_messages)
print(f"时间异步度: {asynchrony_score:.2f}")

# 5. 离群者类型分类
isolate_analyzer = IsolateAnalyzer(config["heterogeneity"])
for member_id, score in heterogeneity_results.items():
    if score.total > config["heterogeneity"]["topology_disconnect"]["outlier_threshold"]:
        isolate_type = isolate_analyzer.classify(
            disconnect_score=score,
            personal_profile=result.personal_profiles.get(member_id),
            group_profile=result.topology_result
        )
        print(f"{member_id}: 脱离度={score.total:.2f}, 类型={isolate_type.value}")
```

## 6. 输出解读

### 脱离度解读

| 脱离度范围 | 含义 | 建议动作 |
|:---|:---|:---|
| 0.0 - 0.3 | 正常融入 | 无特殊动作 |
| 0.3 - 0.6 | 轻度脱离 | 观察，收集更多数据 |
| 0.6 - 0.8 | 中度脱离 | 检查离群者类型，决定是否干预 |
| 0.8 - 1.0 | 高度脱离 | 根据类型决定干预策略 |

### 异质性综合指标

| 指标 | 范围 | 解读 |
|:---|:---|:---|
| 拓扑丰富度 | [0, ∞) | 越高表示群体结构越丰富多样 |
| 因果碎片化指数 | [0, 1] | 越高表示群体越分裂 |
| 影响力集中度(基尼) | [0, 1] | 越高表示少数人控制多数人 |
| 时间异步度 | [0, 1] | 越高表示成员活动时间越不同步 |
| 参与度基尼系数 | [0, 1] | 越高表示参与越不均 |
| 语言离散度 | [0, 1] | 越高表示语言风格越多样 |
