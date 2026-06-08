# 个人-群体不匹配检测指南

> **Tender  — Embracing Heterogeneity**
>
> 本指南详细介绍如何使用 `mismatch/` 模块检测个人与群体之间的"不匹配"程度，并尊重个人的独立性。

## 1. 什么是不匹配？

**不匹配**是指个人的行为模式、情绪状态、认知特征与群体的主流模式之间存在系统性差异。不匹配不同于异质性——异质性是群体的宏观属性，而不匹配是**个人层面的微观指标**。

### 关键哲学

> **不匹配 ≠ 问题。**

Tender  的核心创新在于：**只有当个人同时处于"低自洽性"（自我不一致）和"高不匹配"时，才需要干预。** 一个自洽的个体，即使与群体完全不匹配，也属于"健康"状态。

### 不匹配的三种类型

| 类型 | 检测方法 | 意义 |
|:---|:---|:---|
| **拓扑不匹配** | 持续图 Bottleneck 距离 | 个人拓扑结构与群体拓扑结构在"形状"上的差异 |
| **动态不匹配** | 动态时间规整 (DTW) | 个人时间序列与群体重心轨迹在"节奏"上的差异 |
| **协同不匹配** | 情绪-认知协同度 | 个人的情绪-认知耦合模式与群体主流模式的差异 |

## 2. 核心概念

### 2.1 拓扑不匹配距离 (Topological Mismatch Distance)

**定义**：使用持续同调（Persistent Homology）计算两个人情绪点云的持续图（Persistence Diagram），然后计算两个持续图之间的 Bottleneck 距离或 Wasserstein 距离。

**数学公式**：

```
d_B(P, Q) = inf_{γ} sup_{p ∈ P} ‖p - γ(p)‖_∞
```

其中：
- `P` 和 `Q` 分别是个人和群体的持续图（包含所有点的出生-死亡坐标）
- `γ` 是持续图之间的所有可能的双射
- `‖·‖_∞` 是 L∞ 范数

**取值范围**：`[0, +∞)`，值越大表示拓扑结构差异越大。

### 2.2 动态不匹配距离 (Dynamic Mismatch Distance)

**定义**：使用动态时间规整 (DTW) 计算个人情绪-认知时间序列与群体重心时间序列之间的对齐距离。

```
DTW(X, Y) = min_{π} Σ_{(i,j) ∈ π} d(x_i, y_j)
```

其中 `π` 是时间轴之间的所有可能对齐路径。

**取值范围**：归一化到 `[0, 1]`，值越大表示时间动态差异越大。

### 2.3 自洽性 (Self-Consistency)

**定义**：衡量个人自身状态在时间上的稳定性和可预测性。一个高自洽的人，其行为模式是稳定的、可预测的，即使这些模式与群体不同。

```
SelfConsistency = 1 - MSE(AutoRegression(X) - X)
```

**取值范围**：`[0, 1]`，值越大表示自洽性越高。

## 3. 快速开始

### 3.1 基础使用

```python
from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector
from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector
from tender.mismatch.personal_independence import PersonalIndependenceModel

# 初始化检测器
topo_detector = TopologicalMismatchDetector(config)
dyna_detector = DynamicMismatchDetector(config)
independence_model = PersonalIndependenceModel(config)

# 为特定成员计算拓扑不匹配距离
member_id = "user_1"
personal_point_cloud = personal_profiles[member_id].point_cloud
group_point_cloud = topology_result.point_cloud

topo_distance = topo_detector.compute_distance(
    personal_point_cloud=personal_point_cloud,
    group_point_cloud=group_point_cloud
)
print(f"拓扑不匹配距离: {topo_distance:.3f}")

# 计算动态不匹配距离
personal_ts = personal_profiles[member_id].time_series
group_ts = topology_result.trajectory

dyna_distance = dyna_detector.compute_distance(
    personal_ts=personal_ts,
    group_ts=group_ts
)
print(f"动态不匹配距离: {dyna_distance:.3f}")

# 计算自洽性
self_consistency = independence_model.compute_self_consistency(
    personal_ts
)
print(f"自洽性: {self_consistency:.3f}")

# 综合判断是否需要干预
if dyna_distance > 0.7 and self_consistency < 0.5:
    print(f"警告: {member_id} 处于高度不匹配且低自洽状态，建议关注")
elif dyna_distance > 0.7 and self_consistency > 0.7:
    print(f"信息: {member_id} 是自愿独行者，尊重其选择")
else:
    print(f"正常: {member_id} 状态健康")
```

### 3.2 批量不匹配检测

```python
# 对所有成员进行不匹配检测
mismatch_results = {}
for member_id in member_ids:
    topo_dist = topo_detector.compute_distance(
        personal_point_cloud=personal_profiles[member_id].point_cloud,
        group_point_cloud=topology_result.point_cloud
    )
    dyna_dist = dyna_detector.compute_distance(
        personal_ts=personal_profiles[member_id].time_series,
        group_ts=topology_result.trajectory
    )
    self_cons = independence_model.compute_self_consistency(
        personal_profiles[member_id].time_series
    )
    mismatch_results[member_id] = MismatchMetrics(
        structural_distance=topo_dist,
        dynamic_distance=dyna_dist,
        personal_self_consistency=self_cons
    )

# 找出需要关注的成员
at_risk = [
    mid for mid, m in mismatch_results.items()
    if m.dynamic_distance > 0.7 and m.personal_self_consistency < 0.5
]
print(f"需要关注的成员: {len(at_risk)} 人")

# 找出独立健康的成员
healthy_independent = [
    mid for mid, m in mismatch_results.items()
    if m.dynamic_distance > 0.7 and m.personal_self_consistency > 0.7
]
print(f"独立健康成员: {len(healthy_independent)} 人")
```

## 4. API 参考

### TopologicalMismatchDetector

```python
class TopologicalMismatchDetector:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，需包含：
                - distance_metric: "bottleneck" | "wasserstein"
                - significance_level: float (默认 0.05)
        """
        pass

    def compute_distance(
        self,
        personal_point_cloud: np.ndarray,
        group_point_cloud: np.ndarray,
    ) -> float:
        """
        计算个人与群体拓扑结构的不匹配距离。

        Args:
            personal_point_cloud: 个人情绪点云 (n1, 3)
            group_point_cloud: 群体情绪点云 (n2, 3)

        Returns:
            float: 拓扑不匹配距离
        """
        pass

    def compute_pairwise_matrix(
        self,
        point_clouds: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        计算所有成员两两之间的拓扑距离矩阵。

        Args:
            point_clouds: {member_id: point_cloud}

        Returns:
            np.ndarray: 距离矩阵 (n_members, n_members)
        """
        pass
```

### DynamicMismatchDetector

```python
class DynamicMismatchDetector:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，需包含：
                - dtw_radius: int (DTW 搜索半径)
                - use_multi_scale: bool (是否使用多尺度分析)
                - scales: List[int] (时间尺度列表)
        """
        pass

    def compute_distance(
        self,
        personal_ts: np.ndarray,
        group_ts: np.ndarray,
    ) -> float:
        """
        计算个人与群体动态模式的不匹配距离。

        Args:
            personal_ts: 个人时间序列 (T, D)
            group_ts: 群体重心时间序列 (T, D)

        Returns:
            float: 动态不匹配距离 (归一化到 [0, 1])
        """
        pass

    def compute_multi_scale_distance(
        self,
        personal_ts: np.ndarray,
        group_ts: np.ndarray,
    ) -> Dict[int, float]:
        """
        计算多尺度下的动态不匹配距离。

        Returns:
            Dict[int, float]: {scale: distance}
        """
        pass
```

### PersonalIndependenceModel

```python
class PersonalIndependenceModel:
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典，需包含：
                - model_type: "hmm" | "vae" | "arima"
                - self_consistency_threshold: float
                - wellbeing_threshold: float
        """
        pass

    def compute_self_consistency(
        self,
        time_series: np.ndarray,
    ) -> float:
        """
        计算个人的自洽性。

        使用自回归模型（ARIMA/HMM）的预测误差来衡量。
        预测误差越小，自洽性越高。

        Args:
            time_series: 个人时间序列 (T, D)

        Returns:
            float: 自洽性得分 [0, 1]
        """
        pass

    def predict_independence_trajectory(
        self,
        time_series: np.ndarray,
        horizon: int = 5,
    ) -> np.ndarray:
        """
        预测个人的独立性轨迹（未来时间窗口的不匹配程度）。

        Args:
            time_series: 历史时间序列
            horizon: 预测步数

        Returns:
            np.ndarray: 预测的独立性得分 (horizon,)
        """
        pass
```

## 5. 完整示例

```python
from tender.pipeline import TenderPipeline, load_config
from tender.mismatch.topological_mismatch_detector import TopologicalMismatchDetector
from tender.mismatch.dynamic_mismatch_detector import DynamicMismatchDetector
from tender.mismatch.personal_independence import PersonalIndependenceModel
from tender.visualization.mismatch_plot import MismatchHeatmapPlotter

# 加载配置并运行管道
config = load_config("config.yaml")
pipeline = TenderPipeline(config)
result = pipeline.analyze_window(member_messages)

# 初始化检测器
topo_detector = TopologicalMismatchDetector(config["mismatch"])
dyna_detector = DynamicMismatchDetector(config["mismatch"])
independence_model = PersonalIndependenceModel(config["mismatch"])

# 提取个人点云（从情感向量构建）
personal_point_clouds = {}
for member_id, vectors in result.emotion_vectors.items():
    personal_point_clouds[member_id] = np.array([v.to_array() for v in vectors])

# 批量化计算
for member_id in member_ids:
    topo_dist = topo_detector.compute_distance(
        personal_point_clouds[member_id],
        result.topology_result.point_cloud
    )
    dyna_dist = dyna_detector.compute_distance(
        personal_profiles[member_id].time_series,
        result.topology_result.trajectory
    )
    self_cons = independence_model.compute_self_consistency(
        personal_profiles[member_id].time_series
    )

    # 决策逻辑
    if topo_dist < 0.3 and dyna_dist < 0.3:
        status = "高度匹配"
        action = "无特殊动作"
    elif self_cons > 0.7 and dyna_dist > 0.7:
        status = "独立健康"
        action = "尊重独立性"
    elif self_cons < 0.5 and dyna_dist > 0.7:
        status = "需要关注"
        action = "个性化支持"
    else:
        status = "一般"
        action = "观察"

    print(f"{member_id}: {status} | 拓扑={topo_dist:.2f}, 动态={dyna_dist:.2f}, 自洽={self_cons:.2f} → {action}")
```

## 6. 输出解读

### 不匹配评估矩阵

| 拓扑距离 | 动态距离 | 自洽性 | 结论 | 建议 |
|:---|:---|:---|:---|:---|
| 低 | 低 | 高 | 高度融入 | 维持现状 |
| 低 | 低 | 低 | 迷失追随者 | 帮助建立自我认知 |
| 高 | 高 | 高 | 自愿独行者 | 赋予独立空间 |
| 高 | 高 | 低 | 需要关注者 | 个性化支持 |
| 高 | 低 | 高 | 结构独特者 | 观察，可能为创新者 |
| 低 | 高 | 高 | 节奏独立者 | 尊重个人节奏 |

### 干预判定条件

```python
def should_intervene(mismatch: MismatchMetrics) -> bool:
    """
    判断是否需要进行干预。

    核心原则：只有同时满足"高度不匹配"和"低自洽性"时才干预。
    """
    return (
        (mismatch.structural_distance > 0.7 or mismatch.dynamic_distance > 0.7)
        and mismatch.personal_self_consistency < 0.5
    )
```
