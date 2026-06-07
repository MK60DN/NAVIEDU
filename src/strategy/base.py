"""
策略推理与干预模块 - 抽象基类和数据结构

该模块定义了策略推理引擎的统一接口、风险等级和干预策略的数据结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from tender.fusion.base import FusionResult


class RiskLevel(Enum):
    """风险等级枚举"""
    SAFE = "safe"                      # 安全状态，无需干预
    MILD = "mild"                      # 轻度风险，建议观察
    MODERATE = "moderate"              # 中度风险，推荐干预
    SEVERE = "severe"                  # 重度风险，必须干预
    CRITICAL = "critical"              # 危急状态，立即干预


@dataclass
class InterventionStrategy:
    """
    干预策略数据结构

    Attributes:
        strategy_id: 策略唯一标识
        name: 策略名称
        description: 策略描述
        target_members: 目标成员ID列表（空列表表示全体成员）
        actions: 具体操作列表
        risk_level: 对应的风险等级
        priority: 优先级（1-10，数值越高越优先）
        expected_duration: 预期持续时间（秒）
    """
    strategy_id: str
    name: str
    description: str
    target_members: List[str]
    actions: List[str]
    risk_level: RiskLevel
    priority: int = 5
    expected_duration: int = 3600  # 默认1小时


@dataclass
class StrategyDecision:
    """
    策略决策结果

    Attributes:
        risk_level: 当前风险等级
        risk_score: 风险评分（0-1）
        triggered_strategies: 触发的策略列表
        fusion_result: 触发该决策的融合结果
        timestamp: 决策时间戳
        reasoning: 策略推理说明
        requires_human: 是否需要人工确认
    """
    risk_level: RiskLevel
    risk_score: float
    triggered_strategies: List[InterventionStrategy]
    fusion_result: FusionResult
    timestamp: float = 0.0
    reasoning: str = ""
    requires_human: bool = False


class BaseStrategyEngine(ABC):
    """
    策略推理引擎抽象基类

    所有具体实现必须继承此类并实现所有抽象方法。
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def assess_risk(self, fusion_result: FusionResult) -> StrategyDecision:
        """
        评估当前态势并做出决策

        Args:
            fusion_result: 融合分析结果

        Returns:
            StrategyDecision: 策略决策
        """
        pass

    @abstractmethod
    def get_available_strategies(self) -> List[InterventionStrategy]:
        """
        获取所有可用的干预策略

        Returns:
            List[InterventionStrategy]: 策略列表
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        pass
