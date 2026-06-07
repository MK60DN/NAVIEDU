"""
LLM 策略推理引擎（可替换方案）

该模块实现了基于大语言模型的策略推理方案。
它是 Tender 框架的替代策略引擎之一。

核心方法：
1. 将融合分析结果序列化为结构化的文本摘要
2. 使用 LLM 分析当前群体情绪态势
3. 引导 LLM 输出结构化的风险评估和策略建议
4. 解析 LLM 输出并映射到预定义的策略框架

学术基础：
- 上下文学习 (Brown et al., 2020): LLM 通过少量示例学习
- 思维链推理 (Wei et al., 2022): 通过逐步推理提升 LLM 决策质量
- 情绪智能理论 (Salovey & Mayer, 1990): 将情绪识别与推理结合

与基于规则的引擎的区别：
- 规则引擎：只能处理预设的条件组合，无法应对新出现的情绪模式
- LLM 引擎：具备语义理解能力，能感知微妙的情绪变化和复杂的社交动态
- 但需要 API 调用，延迟较高，成本也更高
"""

import time
import json
from typing import Dict, List, Any, Optional

import numpy as np

from tender.strategy.base import (
    BaseStrategyEngine,
    RiskLevel,
    InterventionStrategy,
    StrategyDecision,
)
from tender.fusion.base import FusionResult


class LLMStrategistEngine(BaseStrategyEngine):
    """
    基于大语言模型的策略推理引擎

    利用 LLM 的语义理解和推理能力，从群体情绪的文本描述中
    识别风险模式并生成个性化的干预策略建议。

    Args:
        config: 配置字典，包含以下字段：
            - llm_model: 模型名称（默认 "deepseek"）
            - llm_temperature: 采样温度（默认 0.3）
            - llm_max_tokens: 最大输出 token 数（默认 512）
            - llm_api_url: API 地址
            - llm_api_key: API 密钥
    """

    # 预定义的策略库（作为 LLM 输出的候选集）
    DEFAULT_STRATEGIES = [
        InterventionStrategy(
            strategy_id="observe",
            name="持续观察",
            description="当前群体情绪状态正常，持续监控即可",
            target_members=[],
            actions=["监控"],
            risk_level=RiskLevel.SAFE,
            priority=1,
        ),
        InterventionStrategy(
            strategy_id="mild_reminder",
            name="温和提醒",
            description="检测到轻度风险，建议发送鼓励信息或分享正面话题",
            target_members=[],
            actions=["发送鼓励信息", "分享正面案例"],
            risk_level=RiskLevel.MILD,
            priority=3,
        ),
        InterventionStrategy(
            strategy_id="positive_redirection",
            name="正面引导",
            description="引入轻松话题或趣味挑战来转移注意力",
            target_members=[],
            actions=["发起趣味投票", "分享搞笑内容"],
            risk_level=RiskLevel.MILD,
            priority=4,
        ),
        InterventionStrategy(
            strategy_id="moderate_mediation",
            name="调解介入",
            description="主动联系活跃成员，引导理性讨论",
            target_members=[],
            actions=["私聊活跃成员", "设置讨论规则"],
            risk_level=RiskLevel.MODERATE,
            priority=6,
        ),
        InterventionStrategy(
            strategy_id="cool_down",
            name="冷静机制",
            description="临时调整发言频率限制或开启全员禁言",
            target_members=[],
            actions=["开启慢速模式", "限制发言频率"],
            risk_level=RiskLevel.MODERATE,
            priority=7,
        ),
        InterventionStrategy(
            strategy_id="severe_intervention",
            name="强制介入",
            description="联系核心冲突成员进行私聊调解，必要时移除严重违规者",
            target_members=[],
            actions=["私聊冲突成员", "移除严重违规者"],
            risk_level=RiskLevel.SEVERE,
            priority=9,
        ),
        InterventionStrategy(
            strategy_id="emergency_shutdown",
            name="紧急关闭",
            description="临时关闭群聊或转移至备用群，等待情绪平复",
            target_members=[],
            actions=["临时关闭群聊", "全员通知安抚"],
            risk_level=RiskLevel.CRITICAL,
            priority=10,
        ),
    ]

    def __init__(self, config: Dict[str, Any]):
        self.llm_model = config.get("llm_model", "deepseek")
        self.temperature = config.get("llm_temperature", 0.3)
        self.max_tokens = config.get("llm_max_tokens", 512)
        self.api_url = config.get("llm_api_url")
        self.api_key = config.get("llm_api_key")

        # 初始化策略库
        self._strategies = config.get("custom_strategies", self.DEFAULT_STRATEGIES)

        # 创建策略 ID 到策略对象的映射
        self._strategy_map = {s.strategy_id: s for s in self._strategies}

    def _build_prompt(self, fusion_result: FusionResult) -> str:
        """
        构建 LLM 输入提示

        将融合分析结果序列化为结构化的文本摘要，
        引导 LLM 进行风险评估和策略推荐。

        Args:
            fusion_result: 融合分析结果

        Returns:
            str: 格式化提示
        """
        feat = fusion_result.feature_vector

        # 提取特征
        cluster_count = int(feat[0] * 20)  # 反归一化
        outlier_ratio = feat
        ring_exists = bool(feat[2](@ref)
        valence = feat
        arousal = feat
        focus = feat
        causal_density = feat
        super_spreader_ratio = feat

        # 构建结构化摘要
        prompt = f"""你是一个群聊情绪管理专家。请根据以下群体情绪的量化分析结果，
评估当前的风险状态并推荐最合适的干预策略。

## 群体情绪量化分析摘要

### 空间拓扑特征
- 情绪派系数: {cluster_count} 个
- 离群成员比例: {outlier_ratio:.1%}
- 是否存在情绪矛盾环: {'是' if ring_exists else '否'}
- 全局情绪重心: [愉悦度={valence:.2f}, 唤醒度={arousal:.2f}, 专注度={focus:.2f}]

### 时间因果特征
- 情绪传染密度: {causal_density:.1%}
- 超级传播者比例: {super_spreader_ratio:.1%}

### 关键风险指标解读
- 离群比例 {'过高 (≥40%)' if outlier_ratio >= 0.4 else '偏高 (20-40%)' if outlier_ratio >= 0.2 else '正常 (<20%)'}
- 情绪环 {'存在' if ring_exists else '未检测到'}
- 全局愉悦度 {'消极' if valence < 0 else '中性' if valence < 0.3 else '积极'}
- 全局唤醒度 {'高涨' if arousal > 0.7 else '适中' if arousal > 0.3 else '低迷'}
- 因果密度 {'密集' if causal_density > 0.6 else '适中' if causal_density > 0.3 else '稀疏'}

## 任务要求

请基于以上分析结果，输出以下 JSON 格式的决策（不要输出其他内容）：

{{
    "risk_level": "safe|mild|moderate|severe|critical",
    "risk_score": 0.0-1.0,
    "reasoning": "简要说明风险判断理由（50字以内）",
    "recommended_strategies": [
        {{
            "strategy_id": "策略ID",
            "custom_reason": "为什么这个策略适合当前情况"
        }}
    ],
    "target_members_hint": "可能需要注意的成员类型（如：离群者、超级传播者等）"
}}

## 可选策略列表
{self._format_strategies_for_prompt()}

只输出 JSON，不要任何解释。"""

        return prompt

    def _format_strategies_for_prompt(self) -> str:
        """格式化策略列表供 LLM 参考"""
        lines = []
        for s in self._strategies:
            lines.append(f"- {s.strategy_id}: {s.name} - {s.description}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        调用 LLM 进行策略推理

        Args:
            prompt: 输入提示

        Returns:
            Dict: 解析后的 LLM 输出

        Raises:
            RuntimeError: API 调用失败或解析失败
        """
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            content = response.json()["choices"]["message"]["content"]
            return self._parse_response(content)

        except Exception as e:
            raise RuntimeError(f"LLM 策略推理失败: {e}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 输出的 JSON

        Args:
            response: LLM 输出的原始字符串

        Returns:
            Dict: 解析后的决策信息
        """
        # 尝试提取 JSON
        try:
            # 查找 JSON 的开始和结束
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)

            raise ValueError("未找到 JSON 格式输出")

        except (json.JSONDecodeError, ValueError) as e:
            # 解析失败时的默认值
            print(f"警告：解析 LLM 策略输出失败: {e}")
            print(f"原始输出: {response[:200]}...")
            return {
                "risk_level": "moderate",
                "risk_score": 0.5,
                "reasoning": "无法解析 LLM 输出，使用默认中等风险",
                "recommended_strategies": [
                    {"strategy_id": "moderate_mediation", "custom_reason": "默认策略"}
                ],
                "target_members_hint": "全部成员",
            }

    def _map_risk_level(self, level_str: str) -> RiskLevel:
        """将字符串风险等级映射到枚举"""
        mapping = {
            "safe": RiskLevel.SAFE,
            "mild": RiskLevel.MILD,
            "moderate": RiskLevel.MODERATE,
            "severe": RiskLevel.SEVERE,
            "critical": RiskLevel.CRITICAL,
        }
        return mapping.get(level_str.lower(), RiskLevel.MODERATE)

    def assess_risk(self, fusion_result: FusionResult) -> StrategyDecision:
        """
        使用 LLM 评估风险并做出决策

        1. 构建包含融合分析结果的结构化提示
        2. 调用 LLM 生成风险评估和策略推荐
        3. 解析 LLM 输出并映射到决策数据结构

        Args:
            fusion_result: 融合分析结果

        Returns:
            StrategyDecision: 策略决策
        """
        # 1. 构建提示
        prompt = self._build_prompt(fusion_result)

        try:
            # 2. 调用 LLM
            llm_decision = self._call_llm(prompt)

            # 3. 解析结果
            risk_level = self._map_risk_level(
                llm_decision.get("risk_level", "moderate")
            )
            risk_score = float(llm_decision.get("risk_score", 0.5))
            reasoning = llm_decision.get("reasoning", "")

            # 4. 匹配推荐策略
            recommended = llm_decision.get("recommended_strategies", [])
            triggered_strategies = []
            for rec in recommended:
                strategy_id = rec.get("strategy_id", "")
                if strategy_id in self._strategy_map:
                    strategy = self._strategy_map[strategy_id]
                    # 更新风险等级匹配
                    strategy.risk_level = risk_level
                    triggered_strategies.append(strategy)

            # 如果 LLM 推荐了不存在的策略，添加默认策略
            if not triggered_strategies:
                triggered_strategies = [
                    self._strategy_map.get("moderate_mediation")
                    or self._strategies
                ]

        except RuntimeError as e:
            # API 失败时的回退策略
            print(f"警告：LLM 策略引擎失败 ({e})，使用默认策略")
            risk_level = RiskLevel.MODERATE
            risk_score = 0.5
            reasoning = f"LLM 调用失败，使用默认策略: {str(e)}"
            triggered_strategies = [s for s in self._strategies if s.risk_level == risk_level]

        return StrategyDecision(
            risk_level=risk_level,
            risk_score=risk_score,
            triggered_strategies=triggered_strategies,
            fusion_result=fusion_result,
            timestamp=time.time(),
            reasoning=reasoning,
            requires_human=(risk_level in [RiskLevel.SEVERE, RiskLevel.CRITICAL]),
        )

    def get_available_strategies(self) -> List[InterventionStrategy]:
        """获取所有可用的干预策略"""
        return self._strategies.copy()

    def get_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        return {
            "name": "LLMStrategistEngine",
            "description": "基于大语言模型的策略推理引擎",
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "num_strategies": len(self._strategies),
        }
