"""
管道编排模块

该模块将所有的分析阶段（向量化、拓扑分析、因果分析、融合、策略）
编排为一个完整的处理管道（Pipeline），并支持配置驱动。
"""

from tender.pipeline.orchestrator import TenderPipeline
from tender.pipeline.config_loader import load_config, ConfigValidationError

__all__ = [
    "TenderPipeline",
    "load_config",
    "ConfigValidationError",
]
