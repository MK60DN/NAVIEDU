# Tender — 情绪时空动力学分析与干预框架

> **用拓扑学刻画情绪的空间结构，用因果推断揭示情绪的时间流向。**

Tender 是一个融合了**持续同调**（Persistent Homology）与**格兰杰因果检验**（Granger Causality Test）的群体情绪动力学开源框架。它能够实时分析群聊中成员的情绪状态，监测情绪的“形状”与“流动”，并基于教育学知识蒸馏的引导逻辑，为群主提供精准的干预策略建议。

---

## 目录

- [项目简介与核心思想](#项目简介与核心思想)
- [核心特性](#核心特性)
- [整体架构](#整体架构)
- [快速开始](#快速开始)
- [Pipeline 工作流程](#pipeline-工作流程)
- [模块详解](#模块详解)
  - [1. 情绪向量化模块](#1-情绪向量化模块-emotion_vectorizer)
  - [2. 空间拓扑分析模块](#2-空间拓扑分析模块-topology_analysis)
  - [3. 时间因果分析模块](#3-时间因果分析模块-causal_analysis)
  - [4. 时空融合模块](#4-时空融合模块-fusion)
  - [5. 策略推理模块](#5-策略推理模块-strategy)
  - [6. 管道编排模块](#6-管道编排模块-pipeline)
- [配置文件](#配置文件)
- [安装依赖](#安装依赖)
- [使用示例](#使用示例)
- [项目路线图](#项目路线图)
- [学术基础](#学术基础)
- [许可证](#许可证)

---

## 项目简介与核心思想

传统群聊管理依赖群主的主观经验，难以量化群体情绪的微妙变化。
Tender 用一套可计算、可解释、可干预的方法论来解决这一问题，其核心在于模块可替换：

空间结构 —— 情绪拓扑分析引擎：

当前实现：通过持续同调（ripser/gudhi）分析情绪点云，发现情绪簇、矛盾环和离群者。
可替换方案：切换到持续拉普拉斯算子以获得更精细的流形结构，或使用拓扑梯度流来追踪情绪结构的动态演变。

时间流向 —— 情绪因果推断引擎：

当前实现：通过格兰杰因果检验（statsmodels）构建成员间的情绪传染网络，识别“超级传播者”。
可替换方案：对非线性情绪传染，可替换为收敛交叉映射（CCM）；对于因果发现，可使用结构因果模型（SCM）或LiNGAM算法。

时空融合 —— 多维特征整合引擎：

当前实现：基于特征向量拼接与VAR模型，将拓扑与因果特征统一为12维向量，并预测下一窗口趋势。
可替换方案：可替换为动态因果图神经网络，或神经时序逻辑来实现更复杂的时空推理。

闭环干预 —— 可配置策略引擎：

当前实现：基于预定义规则，根据离群比例、因果密度等阈值匹配干预策略。
可替换方案：您可以无缝接入轻量级机器学习模型，实现数据驱动的动态策略评分。

模块化基因：Tender 的每一个引擎都基于抽象的 Base 类设计。您只需在 config.yaml 中修改一行配置（如 engine: "convergent_cross_mapping"），即可更换核心算法，而无需改动任何上游分析代码。


---

## 核心特性

- **拓扑情绪分析**：利用持续同调检测情绪点云中的聚类、分歧环和孤立节点，揭示群体情绪的“隐形结构”。
- **因果情绪推断**：基于格兰杰因果检验，量化成员之间情绪影响的**方向**、**强度**和**滞后性**。
- **时空融合表示**：将拓扑特征（聚类数、环标志、离群比例）与因果特征（因果密度、超级传播者数量）拼接为 12 维融合特征向量。
- **时序预测**：使用向量自回归模型（VAR）预测下一时间步的情绪状态，实现**前瞻性干预**。
- **策略即代码**：基于规则与轻量级模型的分层策略引擎，支持**观察 → 提醒 → 介入 → 紧急**四级风险响应。
- **插件式架构**：每个模块都基于抽象基类设计，支持通过配置文件（`config.yaml`）一键切换算法实现（如将格兰杰因果替换为收敛交叉映射 CCM）。
- **可视化友好**：输出可直接用于 `ECharts`、`Plotly`、`NetworkX` 的动态因果拓扑图与情绪热力图。

---

## 整体架构

┌────────────────────────────────────────────────────────────────┐
│                        数据采集层                               │
│              群聊消息实时流（文本、表情、@、图片）                 │
└────────────────────────┬───────────────────────────────────────┘
▼
┌────────────────────────────────────────────────────────────────┐
│                   情绪向量化层（LLM / 神经符号）                  │
│             输出：[愉悦度, 唤醒度, 专注度] ∈ R³                  │
└────────────────────────┬───────────────────────────────────────┘
▼
┌────────────────────────────────────────────────────────────────┐
│                      时空分析引擎                               │
│  ┌─────────────────┐         ┌─────────────────────┐          │
│  │ 空间拓扑分析模块 │         │  时间因果分析模块    │          │
│  │  · 持续同调      │         │  · 格兰杰因果检验    │          │
│  │  · HDBSCAN聚类   │◄──────►│  · 有向因果网络      │          │
│  │  · 情绪环检测    │         │  · 超级传播者识别    │          │
│  └─────────────────┘         └─────────────────────┘          │
└────────────────────────┬───────────────────────────────────────┘
▼
┌────────────────────────────────────────────────────────────────┐
│                      时空融合模块                                │
│          · 特征向量拼接  · 动态因果拓扑图  · VAR 预测            │
└────────────────────────┬───────────────────────────────────────┘
▼
┌────────────────────────────────────────────────────────────────┐
│                      策略推理与干预层                            │
│    · 风险评分  · 策略匹配  · 自动化干预（半自动/全自动）          │
└────────────────────────┬───────────────────────────────────────┘
▼
┌────────────────────────────────────────────────────────────────┐
│                      交互层（群主仪表板）                         │
│    · 实时情绪热力图  · 因果网络可视化  · 运营建议卡片             │
└────────────────────────────────────────────────────────────────┘

**核心数据流**：原始消息 → `EmotionVector` → `TopologyResult` + `CausalResult` → `FusionResult` → `StrategyDecision`

---

## 快速开始

### 环境要求

- Python 3.10+
- 可选（为了最佳性能）：CUDA 11.x + torch（本地 LLM 部署）

### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/tender.git
cd tender

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置（编辑 config.yaml）
#    至少需要设置 emotion_vectorizer 的 model_name 和 api_url

# 4. 运行入口示例
python -m tender.pipeline.example
```

注意：如果使用本地 LLM（如 Llama-3-8B），请确保模型已下载并配置正确的路径。

 Pipeline 工作流程
Tender 的整个分析流程通过 TenderPipeline 编排，每调用一次 analyze_window() 即可完成一个时间窗口的完整分析：
analyze_window(member_messages, window_start, window_end)
├── 1️情绪向量化（Vectorize）
│     输入：成员消息字典  →  输出：{成员ID: EmotionVector}
│
├── 2️空间拓扑分析（Topology）
│     输入：情绪向量字典  →  输出：TopologyResult（聚类数、环标志、离群比例等）
│
├── 3️时间因果分析（Causality）
│     输入：情绪时间序列历史  →  输出：CausalResult（因果网络、超级传播者等）
│
├── 4️时空融合（Fusion）
│     输入：TopologyResult + CausalResult + TimeSeries
│     → 输出：FusionResult（12维特征向量 + 预测 + 融合图）
│
└── 5️策略推理（Strategy）
      输入：FusionResult  →  输出：StrategyDecision（风险等级 + 策略列表）

模块详解
1. 情绪向量化模块 emotion_vectorizer
这一模块负责将群聊中原始的文本消息转换为连续的三维情绪向量 [valence, arousal, focus]，是整个分析流程的入口。
架构设计：模块遵循插件式架构，基于 BaseEmotionVectorizer 抽象基类定义统一接口，所有具体实现都必须继承该类并保证方法签名一致。目前框架提供了三种可替换的实现方案。
LLMVectorizer 是框架的默认实现。它直接使用像 DeepSeek 或 LLaMA 这样的大语言模型，从成员的消息文本中端到端地推断出三个情绪维度的数值。这种方法利用了大模型强大的语义理解能力，不需要额外的特征工程，但由于是"黑箱"推理，可解释性相对有限。
作为可替换方案，NeuroSymbolicVectorizer 采用了更透明的方法。它分两步工作：首先让 LLM 从文本中提取结构化的情绪因果事件（例如"A因为B的批评感到愤怒"），然后通过预定义的符号规则库将这些事件映射到情绪维度空间。这种方法的优势在于可解释性强，每个情绪向量的生成都有明确的语义路径可追溯。
另一个可替换方案是 MultimodalVectorizer，它将分析范围从纯文本拓展到了多模态信号。除了文本语义分析外，它还融合了表情符号的使用频率与类型、消息长度与发送频率、响应时间间隔以及消息时序模式（如深夜活跃度）等行为信号，适合具备多模态数据采集能力的环境。
接口示例：
```
result = vectorizer.vectorize(member_messages, window_start, window_end)
# result.vectors = {"user_1": EmotionVector(valence=0.6, arousal=0.4, focus=0.7), ...}
```

2. 空间拓扑分析模块 topology_analysis
这一模块的核心任务是对"情绪点云"（即所有成员情绪向量在三维空间中的分布）进行拓扑结构分析，揭示群体情绪的隐形空间结构——哪些成员情绪相近（聚类）、是否存在矛盾情绪循环（环状结构）、以及哪些成员情绪孤立（离群点）。
架构设计：模块同样基于插件式架构，BaseTopologyAnalyzer 定义了统一的抽象接口。框架默认使用 PersistentHomologyAnalyzer，它通过持续同调（Persistent Homology）计算情绪点云在不同尺度下的拓扑特征，并借助 HDBSCAN（一种无需预设聚类数量的密度聚类算法）自动发现情绪派系。
预留的两个可替换方案中，PersistentLaplacianAnalyzer 基于持续拉普拉斯算子，能够提供比经典持续同调更精细的流形结构信息；TopologicalGradientFlowAnalyzer 则从拓扑梯度流的角度，追踪情绪结构在时间尺度上的动态演变过程。
核心输出（TopologyResult）：

cluster_count：情绪派系的数量，即自然形成的情绪子群数量。
ring_exists：布尔值，表示是否存在情绪矛盾环（A批评B，B支持C，C反对A这样的循环）。
outlier_ratio：离群成员的比例，即未归属于任何情绪簇的成员占比。
centroid：全局情绪重心，以 [valence, arousal, focus] 三维向量表示整个群体的情绪基调。

3. 时间因果分析模块 causal_analysis
如果说空间拓扑分析给出了情绪的"快照"结构，那么时间因果分析就是揭示情绪的"流动"关系。这个模块的目标是构建一个成员之间的有向因果网络，量化一个人的情绪变化是否以及在多大程度上"导致"了另一个人的情绪变化。
架构设计：基于 BaseCausalAnalyzer 抽象基类，GrangerCausalityAnalyzer 是默认实现，它利用 statsmodels 库中的格兰杰因果检验（Granger Causality Test）来进行分析。格兰杰因果的本质是统计预测关系：如果加入成员X的情绪历史值能显著提高对成员Y的预测精度，我们就认为X是Y的格兰杰原因。
模块提供了三种可替换方案以满足不同的数据特性。ConvergentCrossMappingAnalyzer（收敛交叉映射）适用于非线性系统，当情绪传染关系表现为非线性耦合时比格兰杰因果更准确。StructuralCausalModelAnalyzer（结构因果模型）基于 Judea Pearl 的因果图谱理论，能够处理更复杂的因果结构，包括隐藏的混杂变量。PCLiNGAMAnalyzer 则结合了 PC 算法与线性非高斯无环模型，擅长在观测数据中发现因果方向。
核心输出（CausalResult）：

causal_graph：一个 NetworkX 有向图，其中节点是成员，边表示因果影响的方向。
super_spreaders：超级传播者列表，即在因果网络中出度排名前10%的成员，他们是情绪传染的关键节点。
causal_density：因果密度，取值0到1之间，表示网络中已检测到的因果关系的密集程度。

4. 时空融合模块 fusion
这一模块是整个框架的"中枢神经"，它将空间拓扑分析的结果与时间因果分析的结果融合为统一的数学表示，并具备前瞻性预测能力。
架构设计：基于 BaseFusionModule 抽象基类，FeatureVectorFusion 是默认实现。它采用最直接的特征拼接策略：从拓扑分析结果中提取6维空间特征（包括归一化聚类数、离群比例、环标志、全局重心坐标等），从因果分析结果中提取6维时间特征（包括因果密度、超级传播者比例、平均出度、平均入度、出入度比、情绪波动度等），拼接成一个12维的融合特征向量。同时，它使用向量自回归模型（VAR）对下一时间窗口的群体情绪状态进行预测。
预留的两个可替换方案中，DCTGNN（动态因果图神经网络）利用图神经网络在融合图中进行端到端的特征学习，能够捕捉更复杂的时空依赖关系；NeuralTemporalLogicFusion（神经时序逻辑融合）则将先验的时序逻辑规则嵌入到神经网络中，在保持可解释性的同时提升推理能力。
核心输出（FusionResult）：

feature_vector：12维融合特征向量，直接作为策略推理模块的输入。
fusion_graph：动态因果拓扑图，节点代表成员（颜色标记所属的拓扑簇，大小表示因果出度），边代表因果关系（标注滞后阶数与p值）。
time_series_forecast：对下一时间窗口所有成员情绪状态的预测值。

5. 策略推理模块 strategy
这一模块是整个框架"大脑"，它根据融合结果评估当前群体情绪所处的风险等级，并匹配最合适的干预策略。
架构设计：基于 BaseStrategyEngine 抽象基类，RuleBasedStrategyEngine 是默认实现。它通过融合特征向量与预定义的阈值进行比较，计算出一个0到1之间的综合风险评分，然后映射到五个风险等级之一：

SAFE（评分 < 0.2）：群体情绪稳定，执行持续监控即可。
MILD（评分 0.2 ~ 0.4）：检测到轻度风险，需要主动观察，但暂不需干预。
MODERATE（评分 0.4 ~ 0.6）：出现中度风险，推荐采取干预措施。
SEVERE（评分 0.6 ~ 0.8）：进入重度风险状态，必须采取干预措施。
CRITICAL（评分 ≥ 0.8）：群聊情绪处于危急状态，需要立即行动。

模块提供了两个可替换方案以适应不同场景。CausalRLEngine（因果强化学习引擎）将策略推理建模为马尔可夫决策过程，通过与环境的持续交互在线学习最优的干预时机和策略组合，能自适应地发现比规则引擎更优的策略。LLMStrategistEngine（大语言模型策略引擎）则将融合分析结果序列化为结构化文本，利用 LLM 的语义理解和常识推理能力来生成更细腻、更具上下文感知能力的策略建议。
策略引擎的典型模式匹配逻辑如下：
当检测到"单一大簇 + 高因果密度 + 存在超级传播者"的模式时，建议@该传播者单独沟通，并引入相反观点来打破情绪的单一流向。如果发现"两个以上分离簇 + 簇间因果边少"，说明成员存在明显的情感割裂，应当发起能够连接两簇成员的话题来促进融合。当"存在情绪环 + 中等因果密度"时，意味着群内出现了情绪上的矛盾循环，此时建议插入与当前话题正交的内容，例如从争论转为轻松话题来打断负向循环。如果"离群比例大于20%"，意味着有大量成员处于情绪孤立状态，发起全员投票或调查可以快速摸清他们的需求。当"全局重心在低唤醒且低愉悦"时，群体的整体状态是低迷且消极的，此时发布趣味挑战、红包或表情包大赛是有效的提振手段。

6. 管道编排模块 pipeline
管道编排模块是框架的"骨架"，通过 TenderPipeline 将上述五个分析阶段编排为一个完整的、按序执行的流程管道。
架构设计：TenderPipeline 在初始化时从 config.yaml 配置文件中加载完整配置，通过 _init_modules() 方法根据配置中的 engine 字段动态实例化各模块的具体实现。它维护了一个完整的分析历史字典，包括情绪时间序列历史、拓扑分析结果历史、因果分析结果历史、融合结果历史和策略决策历史，为后续的时间因果分析提供了必要的时序数据积累。
核心流程：每调用一次 analyze_window() 方法，管道会依次执行五个步骤：首先将原始消息转换为三维情绪向量，接着进行空间拓扑分析发现情绪的结构特征，然后利用积累的历史时间序列进行因果分析构建传染网络，随后将空间与时间特征融合并预测下一窗口的情绪动态，最后由策略引擎评估风险并推荐干预策略。整个过程从原始消息输入到一个完整的策略决策输出，一气呵成。
使用示例：
```
from tender.pipeline.orchestrator import TenderPipeline
from tender.pipeline.config_loader import load_config

# 加载配置
config = load_config("config.yaml")

# 初始化管道
pipeline = TenderPipeline(config)

# 分析一个时间窗口
decision = pipeline.analyze_window(
    member_messages={"user_1": [{"text": "今天真开心！", "timestamp": 1234567890}]},
    window_start=1234567880,
    window_end=1234567900,
)

print(f"风险等级: {decision.risk_level}")
print(f"建议操作: {decision.triggered_strategies}")
```

⚙️ 配置文件
config.yaml 是 Tender 的核心配置文件，支持覆盖所有模块的默认参数：
```
# Tender 完整配置示例
emotion_vectorizer:
  engine: "llm"                    # 可选: llm, neuro_symbolic, multimodal
  model_name: "deepseek"
  api_url: null                    # null 表示使用本地模型
  temperature: 0.1                 # LLM 采样温度（低温度保证一致性）
  batch_size: 16

topology_analysis:
  engine: "persistent_homology"
  min_cluster_size: 2
  h1_threshold_ratio: 0.3
  standardize: true

causal_analysis:
  engine: "granger"
  significance_level: 0.05
  max_lag: 5
  emotion_dimension: "composite"   # 可选: valence, arousal, focus, composite

fusion:
  engine: "feature_vector_fusion"
  forecast_method: "var"
  forecast_horizon: 1

strategy:
  outlier_ratio_mild: 0.2
  outlier_ratio_moderate: 0.4
  outlier_ratio_severe: 0.6
  causal_density_mild: 0.3
  causal_density_moderate: 0.6
```
📦 安装依赖
```
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
statsmodels>=0.14.0
networkx>=3.1
pyyaml>=6.0
pydantic>=2.0.0

# Topology analysis
hdbscan>=0.8.33          # 聚类
gudhi>=3.9.0             # 持续同调（备选）
# ripser>=0.6.4          # 持续同调（推荐，更快）

# Visualization (optional)
matplotlib>=3.7.0
plotly>=5.15.0

# LLM inference (optional)
# transformers>=4.36.0    # 本地模型
# requests>=2.31.0        # API 调用
```
📈 使用示例
示例 1：完整 Pipeline 分析
```
from tender.pipeline.orchestrator import TenderPipeline
from tender.pipeline.config_loader import load_config

# 加载配置
config = load_config("config.yaml")
pipeline = TenderPipeline(config)

# 模拟连续窗口分析
windows = [
    (100, 200, {"Alice": [...], "Bob": [...], "Carol": [...]}),
    (200, 300, {"Alice": [...], "Bob": [...], "Carol": [...]}),
    # ...
]

for start, end, messages in windows:
    decision = pipeline.analyze_window(messages, start, end)
    if decision.risk_level.value in ["moderate", "severe", "critical"]:
        print(f"[{start}-{end}] ⚠️  {decision.risk_level.value}: {decision.reasoning}")
        for strategy in decision.triggered_strategies:
            print(f"  建议: {strategy.name} → {strategy.actions}")

```
示例 2：独立使用拓扑分析
```
from tender.topology_analysis.persistent_homology import PersistentHomologyAnalyzer
from tender.emotion_vectorizer.base import EmotionVector

analyzer = PersistentHomologyAnalyzer({"min_cluster_size": 2})
vectors = {
    "A": EmotionVector(valence=0.8, arousal=0.6, focus=0.7),
    "B": EmotionVector(valence=0.7, arousal=0.5, focus=0.6),
    "C": EmotionVector(valence=-0.2, arousal=0.3, focus=0.4),
    "D": EmotionVector(valence=-0.8, arousal=0.7, focus=0.8),
}

result = analyzer.analyze(vectors, 100, 200)
print(f"聚类数: {result.cluster_count}, 环存在: {result.ring_exists}, 离群比例: {result.outlier_ratio:.2f}")
```

🗺️ 项目路线图
版本目标状态V0.1核心 Pipeline 原型，离线分析，批量输出报告 ✅ 已完成V0.5实时建议推送（每 5 分钟），群主可点击执行干预 🔄 开发中V1.0半自动干预（机器人代执行部分操作） 📋 规划中V2.0全自动闭环（强化学习优化干预策略）🔮 远期目标

📚 学术基础
Tender 的设计深受以下领域的启发：

情绪维度理论

Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology.
Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework for describing and measuring individual and group dynamics.

拓扑数据分析

Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological persistence and simplification. Discrete & Computational Geometry.
Ghrist, R. (2008). Barcodes: The persistent topology of data. Bulletin of the American Mathematical Society.

时间序列因果推断

Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica.
Sugihara, G., et al. (2012). Detecting causality in complex ecosystems. Science.

群体情绪动力学

Barsade, S. G. (2002). The ripple effect: Emotional contagion and its influence on group behavior. Administrative Science Quarterly.
Hatfield, E., Cacioppo, J. T., & Rapson, R. L. (1994). Emotional Contagion.

教育学引导策略

Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes.
Schwartz, D. L., Tsang, J. M., & Blair, K. P. (2016). The ABCs of How We Learn.

📄 许可证
```
Version 2.0, January 2004
Copyright [Year] [Copyright Owner]
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


