# Tender — 群聊情绪时空动力学分析与干预框架

> **用拓扑学刻画情绪的空间结构，用因果推断揭示情绪的时间流向。**
>
> — 让每一位群主都拥有一台情绪时空导航仪

Tender 是一个融合了持续同调与格兰杰因果检验的群体情绪动力学开源框架。它能够实时分析群聊中成员的情绪状态，监测情绪的“形状”与“流动”，并基于可替换的模块化架构，为群主提供精准的干预策略建议。

---

## 目录

- [项目简介与核心思想](#项目简介与核心思想)
- [核心特性](#核心特性)
- [整体架构](#整体架构)
- [快速开始](#快速开始)
- [Pipeline 工作流程](#pipeline-工作流程)
- [模块详解](#模块详解)
- [学术基础](#学术基础)
- [许可证](#许可证)

---

## 项目简介与核心思想

传统群聊管理依赖群主的主观经验，难以量化群体情绪的微妙变化。Tender 用一套可计算、可解释、可干预的方法论来解决这一问题，其核心在于**模块可替换**。

1. **空间结构**：通过持续同调分析成员情绪在三维空间中的拓扑形状，发现情绪簇、矛盾环和离群者。可替换方案包括持续拉普拉斯算子、拓扑梯度流。
2. **时间流向**：通过格兰杰因果检验构建成员之间的情绪传染网络，识别超级传播者与关键接收者。可替换方案包括收敛交叉映射、结构因果模型、LiNGAM。
3. **时空融合**：将空间特征与时间特征统一在动态因果拓扑图中，并预测下一窗口的群体情绪趋势。可替换方案包括动态因果图神经网络、神经时序逻辑。
4. **闭环干预**：基于融合特征向量匹配预定义的引导策略。可替换方案包括因果强化学习引擎、大语言模型策略引擎。
5. **共识优化**：可选的后处理过滤层，在不修改上游数学建模的前提下，根据共识度和互惠指数优化策略推荐，促进群体共同点的发现。

---

## 核心特性

- **拓扑情绪分析**：利用持续同调检测情绪点云中的聚类、分歧环和孤立节点，揭示群体情绪的隐形结构。
- **因果情绪推断**：基于格兰杰因果检验，量化成员之间情绪影响的方向、强度和滞后性。
- **时空融合表示**：将拓扑特征与因果特征拼接为12维融合特征向量，用于后续策略匹配。
- **时序预测**：使用向量自回归模型预测下一时间步的情绪状态，实现前瞻性干预。
- **插件式架构**：每个模块均基于抽象基类设计，支持通过配置文件一键切换算法实现。
- **可插拔共识优化**：可选的后处理共识化过滤层，在保留原数学建模的同时，为愿意促进和谐的群主提供更丰富的工具。

---

## 整体架构

整个框架由六个核心模块组成，通过管道编排器按序执行。数据流从原始消息开始，依次经过情绪向量化、空间拓扑分析、时间因果分析、时空融合、策略推理，最后输出策略决策。其中，第五步之后可选接一个共识化过滤层，用于在保持上游数学建模不变的前提下，从共识角度优化策略推荐。

每个模块均遵循插件式架构，通过配置文件中的 engine 字段即可切换不同的算法实现，无需修改任何上游或下游代码。

核心数据流：原始消息 → EmotionVector → TopologyResult + CausalResult → FusionResult → StrategyDecision → [可选] 共识化过滤层 → 优化后的策略决策

---

## 快速开始

### 环境要求

- Python 3.10+

### 安装与运行

1. 克隆仓库。
2. 安装依赖。
3. 编辑 config.yaml 配置文件，至少设置 emotion_vectorizer 的 model_name。
4. 运行入口示例：`python -m tender.pipeline.example`

---

## Pipeline 工作流程

Tender 的整个分析流程通过 TenderPipeline 编排，每调用一次 analyze_window 即可完成一个时间窗口的完整分析。

流程步骤：
1. 情绪向量化：将原始消息转换为三维情绪向量。
2. 空间拓扑分析：检测聚类、环和离群点。
3. 时间因果分析：检测情绪影响关系。
4. 时空融合：构建融合特征向量并预测。
5. 策略推理：评估风险并进行干预。
6. 共识化过滤：可选的后处理步骤，根据共识度优化策略推荐。

---

## 模块详解

### 1. 情绪向量化模块 emotion_vectorizer

这一模块负责将群聊中原始的文本消息转换为连续的三维情绪向量，是整个分析流程的入口。模块遵循插件式架构，基于抽象基类定义统一接口。

默认实现 LLMVectorizer 直接使用大语言模型从文本中端到端推断三个情绪维度的数值。可替换方案 NeuroSymbolicVectorizer 采用两步法：先提取结构化的事件三元组，再通过符号规则映射到情绪空间，可解释性更强。MultimodalVectorizer 则融合文本语义、表情符号行为、社交交互特征等多模态信号，适合具备多模态数据采集能力的环境。

核心输出是一个以成员ID为键、情绪向量为值的字典。

### 2. 空间拓扑分析模块 topology_analysis

这一模块的核心任务是对情绪点云进行拓扑结构分析，揭示群体情绪的隐形空间结构。默认实现 PersistentHomologyAnalyzer 通过持续同调计算情绪点云在不同尺度下的拓扑特征，并借助 HDBSCAN 自动发现情绪派系。

预留的可替换方案中，PersistentLaplacianAnalyzer 基于持续拉普拉斯算子提供更精细的流形结构信息；TopologicalGradientFlowAnalyzer 从拓扑梯度流的角度追踪情绪结构的动态演变。

核心输出 TopologyResult 包含四个关键字段：情绪派系数量、是否存在情绪矛盾环、离群成员比例、全局情绪重心。

### 3. 时间因果分析模块 causal_analysis

这一模块的目标是构建成员之间的有向因果网络，量化情绪的传染关系。默认实现 GrangerCausalityAnalyzer 利用格兰杰因果检验进行统计分析。

模块提供了三种可替换方案以适应不同的数据特性。ConvergentCrossMappingAnalyzer 适用于非线性系统；StructuralCausalModelAnalyzer 基于结构因果图谱理论处理更复杂的因果结构；PCLiNGAMAnalyzer 结合 PC 算法与线性非高斯无环模型，擅长在观测数据中发现因果方向。

核心输出 CausalResult 包含有向网络图、超级传播者列表和因果密度。

### 4. 时空融合模块 fusion

这一模块将空间拓扑分析结果与时间因果分析结果融合为统一的数学表示。默认实现 FeatureVectorFusion 采用特征拼接策略，将6维空间特征与6维时间特征拼接成12维融合特征向量，并使用向量自回归模型预测下一窗口的情绪状态。

预留的可替换方案中，DCTGNN 利用图神经网络捕捉更复杂的时空依赖关系；NeuralTemporalLogicFusion 将时序逻辑规则嵌入神经网络，在保持可解释性的同时提升推理能力。

核心输出 FusionResult 包含12维融合特征向量、动态因果拓扑图和下一窗口的情绪预测。

### 5. 策略推理模块 strategy

这一模块根据融合结果评估风险等级并匹配干预策略。默认实现 RuleBasedStrategyEngine 通过预定义阈值计算综合风险评分，映射到五个风险等级。

可替换方案 CausalRLEngine 将策略推理建模为马尔可夫决策过程，通过与环境的交互学习最优策略。LLMStrategistEngine 将融合结果序列化为结构化文本，利用大语言模型的语义理解能力生成更具上下文感知能力的策略建议。

策略引擎同时支持共识化过滤层作为后处理步骤。当启用时，共识化过滤层会计算共识分数和互惠指数，在不修改上游数学建模的前提下，对策略推荐进行优化，当检测到低共识或低互惠模式时，会在原有策略基础上加入促进共同点和双向交流的建议。

### 6. 管道编排模块 pipeline

管道编排模块通过 TenderPipeline 将上述分析阶段编排为完整的处理流程。它在初始化时加载配置文件并动态实例化各模块，维护完整的分析历史字典。

核心方法 analyze_window 接受成员消息字典和时间窗口参数，依次执行五个分析步骤，并根据配置决定是否执行共识化过滤，最终返回一个完整的策略决策结果。

---

## 学术基础

Tender 的设计深受以下领域的启发：

1. 情绪维度理论：Russell 的情感环状模型、Mehrabian 的愉悦-唤醒-支配框架。
2. 拓扑数据分析：Edelsbrunner 的持续同调理论、Ghrist 的拓扑数据条形码分析。
3. 时间序列因果推断：Granger 的因果检验方法、Sugihara 的收敛交叉映射。
4. 群体情绪动力学：Barsade 的情绪传染涟漪效应、Hatfield 的情绪传染理论。
5. 教育学引导策略：Vygotsky 的最近发展区理论、Schwartz 的学习科学框架。

---

## 许可证

**Apache License, Version 2.0**

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
