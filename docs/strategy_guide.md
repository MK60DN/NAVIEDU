# 策略推理指南

> **Tender v2.0 — Embracing Heterogeneity**
>
> 本指南详细介绍如何使用 `strategy/` 模块基于融合结果和协同结果进行策略推理与干预决策。

## 1. 什么是策略推理？

**策略推理**是 Tender 管道的最后一步，它基于前面的情绪向量化、拓扑分析、因果分析、认知分析、融合和协同分析的结果，回答核心问题：

> **"现在应该做什么？"**

### 策略推理的三层架构

```text
第一层：风险评估
    ↓
第二层：策略选择 →  异质性协调层（🆕） → 最终策略输出
    ↑
第三层：认知约束（来自协同模块）
```

## 2. 风险管理体系

### 2.1 风险等级

Tender 定义了五个风险等级：

| 等级 | 值 | 颜色 | 含义 | 要求 |
|:---|:---|:---|:---|:---|
| SAFE | 0 | 🟢 绿色 | 安全 | 无操作 |
| LOW | 1 | 🔵 蓝色 | 低风险 | 观察 |
| MEDIUM | 2 | 🟡 黄色 | 中等风险 | 建议干预 |
| HIGH | 3 | 🟠 橙色 | 高风险 | 需要干预 |
| CRITICAL | 4 | 🔴 红色 | 危急 | 立即干预 |

### 2.2 风险评估函数

```python
def assess_risk_level(
    health_index: float,        # 群体健康度 (0-1)
    synergy_score: float,       # 情绪-认知协同度 (0-1)
    heterogeneity_index: float, # 群体异质性指数 (0-1)
    disengagement_ratio: float, # 脱离度 (0-1)
) -> RiskLevel:
    # 综合评分
    risk_score = (
        (1 - health_index) * 0.35 +
        (1 - synergy_score) * 0.25 +
        heterogeneity_index * 0.20 +
        disengagement_ratio * 0.20
    )

    if risk_score > 0.8:
        return RiskLevel.CRITICAL
    elif risk_score > 0.6:
        return RiskLevel.HIGH
    elif risk_score > 0.4:
        return RiskLevel.MEDIUM
    elif risk_score > 0.2:
        return RiskLevel.LOW
    else:
        return RiskLevel.SAFE
```

## 3. 干预策略

### 3.1 预定义策略

Tender 定义了 7 种预定义的干预策略：

| 策略 ID | 策略名称 | 风险等级 | 描述 |
|:---|:---|:---|:---|
| 0 | NONE | SAFE | 无操作 |
| 1 | OBSERVE | LOW | 持续观察，收集更多数据 |
| 2 | INFORMATION_SUPPORT | LOW-MEDIUM | 提供额外学习材料或信息 |
| 3 | COGNITIVE_SUPPORT_LOW | MEDIUM | 轻量级认知支持（如提示、引导） |
| 4 | COGNITIVE_SUPPORT_HIGH | MEDIUM-HIGH | 强化认知支持（如重新讲解、一对一辅导） |
| 5 | EMOTIONAL_INTERVENTION | HIGH | 情绪干预（如调节氛围、安抚） |
| 6 | EMERGENCY_INTERVENTION | CRITICAL | 紧急干预，需要人工介入 |

### 3.2 默认引擎：因果强化学习 (Causal RL)

该引擎将群体状态建模为马尔科夫决策过程 (MDP)，使用深度 Q 网络 (DQN) 学习最优策略。

```python
class CausalRLEngine:
    def __init__(self, config):
        self.state_dim = config.get("state_dim", 16)   # 状态空间维度
        self.action_dim = config.get("action_dim", 7)   # 动作空间维度
        self.gamma = config.get("gamma", 0.99)          # 折扣因子
        self.epsilon = config.get("epsilon_start", 1.0) # 探索率
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.memory = ReplayBuffer(config.get("buffer_size", 10000))
        self.use_double_q = config.get("use_double_q", True)

    def select_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 探索
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.argmax().item()  # 利用

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        # 双 Q 学习
        if self.use_double_q:
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
        else:
            next_q_values = self.target_network(next_states).max(dim=1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(self.q_network(states).gather(1, actions.unsqueeze(1)), targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 3.3 可选引擎：LLM 策略器

使用大语言模型进行策略推理：

```python
class LLMStrategistEngine:
    def __init__(self, config):
        self.model_name = config.get("model_name", "gpt-4o")
        self.fallback_strategy = config.get("fallback_strategy", "medium_risk")
        self.client = OpenAI()

    def reason(self, fusion_result, synergy_result, mismatch_metrics):
        # 1. 构建结构化文本摘要
        summary = self._build_summary(fusion_result, synergy_result, mismatch_metrics)

        # 2. 调用 LLM
        prompt = f"""
        你是一个群体情绪-认知分析专家。你需要根据以下分析结果，推荐最优干预策略。

        ## 当前群体状态
        {summary}

        ## 可选择的策略
        {self._strategy_descriptions}

        ## 输出格式
        请以 JSON 格式输出：
        {{
            "risk_level": "HIGH | MEDIUM | LOW | SAFE | CRITICAL",
            "selected_strategy_id": 0-6,
            "reasoning": "解释为什么选择这个策略",
            "target_members": ["member_id_1", ...],
            "specific_actions": ["action_1", ...]
        }}
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        try:
            return StrategyResult.from_dict(json.loads(response.choices[0].message.content))
        except Exception:
            # LLM 调用失败，使用默认策略
            return self._get_fallback_strategy()
```

## 4. 🆕 异质性协调层

异质性协调层替代了原 Tender 框架中的共识化过滤层。它不再追求"共识"，而是在承认异质性的前提下，对基础策略进行多目标优化调整。

### 4.1 核心逻辑

```python
class HeterogeneityCoordinationLayer:
    def coordinate(
        self,
        base_strategy: StrategyResult,
        heterogeneity_metrics: HeterogeneityMetrics,
        mismatch_metrics: Dict[str, MismatchMetrics],
    ) -> List[StrategyResult]:
        """
        根据群体异质性和个人-群体不匹配度，调整基础策略。
        返回可能包含多个并行策略（针对不同子群体）。
        """
        adjusted_strategies = []

        # 检查是否需要并行策略
        if heterogeneity_metrics.topological_richness > 0.7:
            # 高异质性 → 生成并行策略
            adjusted_strategies.extend(
                self._generate_parallel_strategies(
                    base_strategy,
                    heterogeneity_metrics,
                    mismatch_metrics
                )
            )
        else:
            # 低异质性 → 单一策略 + 局部调整
            adjusted_strategies.append(
                self._adjust_single_strategy(base_strategy, mismatch_metrics)
            )

        return adjusted_strategies

    def _generate_parallel_strategies(self, base, hetero, mismatches):
        strategies = []
        # 因果碎片化分析 → 针对不同子群体
        for cluster_id in hetero.cluster_ids:
            cluster_members = hetero.cluster_members[cluster_id]
            cluster_strategy = StrategyResult(
                risk_level=base.risk_level,
                target_members=cluster_members,
                action=self._adapt_action_for_cluster(base.action, cluster_id),
                confidence=base.confidence * 0.9,
            )
            strategies.append(cluster_strategy)

        # 离群者特殊处理
        outlier_ids = [mid for mid, m in mismatches.items()
                       if m.dynamic_distance > 0.7 and m.personal_self_consistency > 0.7]
        if outlier_ids:
            # 赋予独立角色
            strategies.append(
                StrategyResult(
                    risk_level="LOW",
                    target_members=outlier_ids,
                    action=self._get_standalone_action(),
                    confidence=0.95,
                )
            )
        return strategies
```

### 4.2 协调策略

| 策略 | 触发条件 | 行为 |
|:---|:---|:---|
| 尊重独立 | 个人自洽性高 + 不匹配度高 | 不干预，赋予独立空间 |
| 子群体分别处理 | 拓扑丰富度高 + 存在明显子群体 | 生成多个并行策略 |
| 局部调整 | 少数成员不匹配度高 + 自洽性低 | 在原策略基础上添加专属动作 |
| 全局调整 | 整体不匹配度高 | 修改全局策略风格 |

## 5. 快速开始

### 5.1 基础使用

```python
from tender.strategy.causal_rl import CausalRLEngine
from tender.strategy.heterogeneity_coordination_layer import HeterogeneityCoordinationLayer

# 初始化策略引擎
strategy_engine = CausalRLEngine(config["strategy"])
coordination_layer = HeterogeneityCoordinationLayer(config["strategy"]["heterogeneity_coordination"])

# 执行策略推理
base_strategy = strategy_engine.reason(
    fusion_result=result.fusion_result,
    synergy_result=result.synergy_result,
    mismatch_metrics=result.mismatch_metrics,
)

# 异质性协调
final_strategies = coordination_layer.coordinate(
    base_strategy=base_strategy,
    heterogeneity_metrics=result.heterogeneity_metrics,
    mismatch_metrics=result.mismatch_metrics,
)

# 输出最终策略
for strategy in final_strategies:
    print(f"风险等级: {strategy.risk_level}")
    print(f"目标成员: {strategy.target_members[:5]}...")
    print(f"推荐动作: {strategy.action}")
    print(f"置信度: {strategy.confidence:.2f}")
    if strategy.rationale:
        print(f"决策理由: {strategy.rationale}")
    print("---")
```

## 6. 配置参数

```yaml
strategy:
  engine: "causal_rl"                # causal_rl | llm_strategist
  state_dim: 16                      # 状态维度
  action_dim: 7                      # 策略数量

  causal_rl:
    learning_rate: 0.001
    gamma: 0.99                      # 折扣因子
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 0.995
    buffer_size: 10000
    batch_size: 64
    target_update_freq: 100
    use_double_q: true
    tau: 0.005

  llm_strategist:
    model_name: "gpt-4o"
    fallback_strategy: "medium_risk"

  heterogeneity_coordination:
    enabled: true
    min_consensus: 0.3
    max_retries: 3
    conflict_resolution: "adaptive"
    max_parallel_strategies: 5
```
