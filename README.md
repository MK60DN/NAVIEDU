# Tender: 拥抱异质性 —— 群体情绪时空动态分析框架

Tender 是一个面向群聊环境的群体情绪时空动态分析框架。它整合了拓扑数据分析与时间因果推断技术，能够揭示集体情绪状态的"形状"和流动规律，将隐性的群体动态模式转化为可量化、可推理、可干预的结构化信息。

它核心设计是"拥抱异质性"。我们认为，一个健康的群体并非由完全同构的个体组成，而是在承认并尊重差异的前提下，由不同类型成员共同维持的动态平衡系统。本框架的目的不是在群体内部强制共识，而是去理解这种复杂性，并帮助管理者在异质性中找到促进群体健康发展的最佳策略。

## 核心特性

*   **多维度分析**：从情绪、认知、拓扑、因果四个维度量化群体状态。
*   **异质性优先**：内置异质性分析模块，识别离群者、不匹配模式，并设计针对性的协调策略。
*   **可插拔引擎**：核心分析模块均支持通过配置文件动态切换不同的算法实现（如多种因果推断、融合策略）。
*   **全链路管道**：提供从原始消息输入到策略输出的完整、自动化分析流程。
*   **可解释性**：神经符号架构与结构因果模型等技术让AI分析过程透明、可审计。

## 架构总览

Tender 采用分层插件架构，自上而下分为：

### 策略与决策层 (Strategy & Decision Layer)

*   **策略推理模块 (strategy/)**：基于分析结果，输出风险等级与干预策略。
*   **异质性协调层 (Heterogeneity Coordination Layer)**：在策略中加入对群体异质性的考量，生成多目标优化策略。

### 核心分析层 (Core Analysis Layer)

*   **情绪向量化 (emotion\_vectorizer/)**：将文本消息转换为连续的三维情绪向量（愉悦度、唤醒度、专注度）。
*   **空间拓扑分析 (topology\_analysis/)**：分析群体情绪在空间上的聚类、环状结构及离群点。
*   **时间因果分析 (causal\_analysis/)**：分析成员间情绪影响的因果网络，识别"超级传播者"。
*   **时空融合 (fusion/)**：将拓扑与因果分析结果融合，预测群体情绪未来走势。
*   **认知状态分析 (cognition/)**：推断成员的认知负荷、理解水平与认知阶段。
*   **情绪-认知协同 (synergy/)**：分析群体情绪状态与认知发展阶段的匹配度。
*   **异质性分析 (heterogeneity/)**：量化群体在拓扑、因果、行为、认知维度的差异。
*   **个人-群体匹配检测 (mismatch/)**：识别个体的行为、情绪或认知是否与群体主流模式存在系统性差异。

### 基础设施层 (Infrastructure Layer)

*   **配置管理 (config\_loader.py)**：加载并验证YAML配置文件。
*   **通用工具 (utils/)**：提供通用功能支持。

## 快速开始

### 1. 环境准备

*   **Python**: >= 3.11
*   **依赖**: 使用 pip 安装。

### 2. 克隆仓库

```bash
git clone <your-repository-url>
cd tender
```

### 3. 安装依赖

Tender 的依赖分为核心依赖和新功能可选依赖，安装方式如下：

仅安装核心依赖（用于基础分析）：

```bash
pip install -r requirements.txt
```

安装所有依赖（包含深度学习、LLM、可视化等）：

```bash
pip install -e ".[dev,dl,llm,vis]"
# 或等效于完整安装
pip install -e .
```

可选依赖组包括：`dev` (开发)、`dl` (深度学习)、`llm` (大语言模型)、`tda` (拓扑数据分析)、`vis` (可视化)。

### 4. 配置

所有核心模块的配置参数都在 `config.yaml` 中管理。你可以根据需求修改此文件以切换各个分析模块的引擎和参数。

### 5. 运行示例

仓库提供多种使用场景的示例脚本，位于 `examples/` 目录下：

```bash
# 基础分析管道
python examples/basic_pipeline.py

# 全链路分析示例
python examples/full_pipeline.py

# 异质性分析示例
python examples/heterogeneity_analysis.py

# 不匹配检测示例
python examples/mismatch_detection.py

# 自愿隔离者案例分析
python examples/voluntary_isolate.py
```

## 模块详解与API参考

Tender 框架包含12个功能模块，每个模块都提供了清晰的公共接口。我们建议你查阅 `/docs/api_reference.md` 以获取详细的API文档（包括类、函数、参数与返回值说明）。

| 模块                  | 文档                      | 核心功能                 |
| ------------------- | ----------------------- | -------------------- |
| pipeline            | /docs/api\_reference.md | 管道编排，配置加载与验证         |
| emotion\_vectorizer | /docs/api\_reference.md | 文本消息到情绪向量的转换         |
| topology\_analysis  | /docs/api\_reference.md | 群体情绪的空间结构分析（聚类、环状结构） |
| causal\_analysis    | /docs/api\_reference.md | 情绪影响的因果网络构建与关键角色识别   |
| fusion              | /docs/api\_reference.md | 时空特征融合与未来情绪预测        |
| cognition           | /docs/api\_reference.md | 群体认知状态分析（认知负荷、理解水平等） |
| synergy             | /docs/api\_reference.md | 情绪-认知协同状态分析          |
| heterogeneity       | /docs/api\_reference.md | 多维度异质性量化分析，离群者分类     |
| mismatch            | /docs/api\_reference.md | 个人-群体不匹配检测与自洽性评估     |
| strategy            | /docs/api\_reference.md | 风险等级评估与策略推理          |
| visualization       | /docs/api\_reference.md | 数据可视化功能              |
| config              | /docs/api\_reference.md | 通用的配置加载与合并工具         |

## 设计原则与使用指南

在深入使用 Tender 之前，请理解其核心设计原则。这有助于你更好地解读分析结果并做出决策。

*   **异质性是一种信息，而非噪音**：每一种偏离主流的模式都在讲述一个独特的故事。
*   **自洽性压倒融入度**：一个自洽的、自愿不融入的个体不应被视为问题，其独特性可能是群体的宝贵资源。
*   **先理解，再干预**：在做出任何决策前，请先从拓扑、行为、认知等多个角度去理解"为什么会这样"。
*   **并行策略**：针对异质性高的群体，放弃寻找"一刀切"的策略，设计多条策略并行实施。

详细的哲学讨论请阅读 `/docs/philosophy.md`。异质性分析的具体策略请参考 `/docs/heterogeneity_guide.md`。策略推理的详细说明请参考 `/docs/strategy_guide.md`。情绪-认知协同的方法论请参考 `/docs/synergy_guide.md`。

## 开发指南

### 运行测试

本项目使用 pytest 进行测试。测试用例按模块组织在 `tests/` 目录下。

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/test_emotion_vectorizer.py
```

### 代码规范

*   **格式化**：项目使用 black 与 isort 进行代码格式化。
*   **命名**：遵循 PEP 8 规范。

### 贡献

我们欢迎任何形式的贡献，包括但不限于：

*   报告 Bug 或提出 Feature Request
*   提交 Pull Request 修复 Bug 或添加新功能
*   完善文档和测试用例

如果你有兴趣为 Tender 贡献代码，请先阅读 `CONTRIBUTING.md`（若存在）。

## 致谢

本项目的开发基于多项优秀的研究成果，包括但不限于：持续同调（Persistent Homology）、结构因果模型（SCM）、图卷积网络（GCN）、收敛交叉映射（CCM）等。向所有为该领域做出贡献的研究者表示感谢。

## 许可证

Tender 遵循 Apache License Version 2.0 许可证。
