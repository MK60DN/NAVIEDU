"""
策略推理与干预模块

该模块根据时空融合结果进行态势识别和策略推理。
当检测到风险模式时，自动推荐干预策略或触发告警。

学术基础：
- 基于规则的策略引擎：将情绪动力学的定性知识转化为定量条件
- 多层次风险预警：从轻度到重度，提供梯度干预方案
- 策略效果反馈：通过后续窗口的分析结果评估干预效果

核心功能：
1. 态势识别：将融合特征映射到预定义的风险模式
2. 策略匹配：根据态势推荐干预策略
3. 自动驾驶：在特定条件下自动执行策略
4. 干预日志记录：记录所有干预操作供后续评估
"""

from tender.strategy.base import BaseStrategyEngine, RiskLevel
from tender.strategy.rule_based_strategy import RuleBasedStrategyEngine

__all__ = [
    "BaseStrategyEngine",
    "RiskLevel",
    "RuleBasedStrategyEngine",
]
