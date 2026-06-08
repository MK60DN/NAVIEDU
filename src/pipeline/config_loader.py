"""
配置加载与验证模块

负责加载 YAML 配置文件并验证配置的完整性和合理性。
"""

import os
from typing import Dict, Any

import yaml


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


# 必需配置项列表
REQUIRED_SECTIONS = ["emotion_vectorizer", "topology_analysis", "causal_analysis", "fusion", "strategy"]
REQUIRED_FIELDS = {
    "emotion_vectorizer": ["engine"],
    "topology_analysis": ["engine"],
    "causal_analysis": ["engine"],
    "fusion": ["engine"],
    "strategy": ["engine"],
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载并验证配置文件

    Args:
        config_path: 配置文件路径（YAML格式）

    Returns:
        Dict[str, Any]: 验证后的配置字典

    Raises:
        ConfigValidationError: 配置验证失败
    """
    if not os.path.exists(config_path):
        raise ConfigValidationError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 验证必需部分
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ConfigValidationError(f"缺少必需配置段: {section}")

    # 验证必需字段
    for section, fields in REQUIRED_FIELDS.items():
        for field in fields:
            if field not in config[section]:
                raise ConfigValidationError(f"配置段 {section} 缺少必需字段: {field}")

    return config


def merge_configs(default: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个配置字典"""
    result = default.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
