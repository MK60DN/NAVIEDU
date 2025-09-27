from .base_agent import DeepSeekBaseAgent
from typing import Dict, List, Any


class DeepSeekBalancingAgent(DeepSeekBaseAgent):
    """基于DeepSeek的平衡协调助手"""

    def __init__(self, agent_id: str, api_key: str, model: str = "deepseek-chat"):
        super().__init__(
            agent_id=agent_id,
            role="协调平衡助手",
            base_weight=1.0,  # 中性权重
            api_key=api_key,
            model=model
        )

        self.coordination_strategies = [
            "观点整合", "矛盾调和", "优先级排序", "综合分析"
        ]

    def _create_system_prompt(self) -> str:
        return """你是Navi的协调平衡助手，负责整合学习助手和批判思考助手的观点，为用户提供最终的综合建议。

你的核心职责：
1. 整合不同助手的观点和建议
2. 平衡学习效率与批判思维
3. 协调可能存在的观点冲突
4. 为用户提供最终的行动建议
5. 维护知识图谱的一致性和准确性

协调原则：
- 保持客观中立的立场
- 重视实证和逻辑
- 平衡不同观点的价值
- 优先考虑用户的最佳利益
- 确保建议的可行性

整合策略：
- 寻找观点的共同点
- 分析不同观点的适用场景
- 提供综合性的解决方案
- 考虑短期和长期的影响
- 保持开放和灵活的态度

风格要求：
- 客观、理性、平衡
- 简洁明了、切中要害
- 具有指导性和可操作性
- 尊重不同观点的价值

请综合各方面信息，为用户提供最佳的学习建议和知识整合方案。"""

    async def synthesize_responses(self,
                                   learning_response: Dict,
                                   questioning_response: Dict,
                                   user_context: Dict) -> Dict:
        """综合学习助手和质疑助手的响应"""

        synthesis_prompt = f"""
        请综合分析以下内容：

        学习助手的建议：
        {learning_response.get('content', '')}

        批判思考助手的质疑：
        {questioning_response.get('content', '')}

        用户情况：
        - 当前问题：{user_context.get('current_question', '')}
        - 学习目标：{user_context.get('learning_goals', [])}

        请提供：
        1. 综合性的学习建议
        2. 需要重点注意的问题
        3. 具体的行动步骤
        4. 知识图谱更新建议
        """

        context = {
            "learning_content": learning_response,
            "questioning_content": questioning_response,
            "user_context": user_context
        }

        return await self.generate_response(synthesis_prompt, context)

    def _assess_response_quality(self, response: str) -> float:
        """评估平衡助手响应质量"""
        quality_score = super()._assess_response_quality(response)

        # 平衡助手特有的质量指标
        balance_markers = [
            "综合", "平衡", "整合", "建议", "考虑", "另外", "同时", "总体", "综上", "因此"
        ]

        balance_score = sum(1 for marker in balance_markers if marker in response) / len(balance_markers)

        # 检查结构化程度（是否有明确的建议步骤）
        structure_markers = ["1.", "2.", "3.", "首先", "其次", "最后", "总结"]
        structure_score = min(sum(1 for marker in structure_markers if marker in response) / 4, 1.0)

        return (quality_score + balance_score + structure_score) / 3

    async def update_knowledge_graph_decision(self,
                                              user_input: str,
                                              learning_output: str,
                                              questioning_output: str) -> Dict:
        """决定如何更新知识图谱"""

        decision_prompt = f"""
        基于以下信息，决定如何更新用户的知识图谱：

        用户输入：{user_input}
        学习助手输出：{learning_output[:300]}...
        批判思考输出：{questioning_output[:300]}...

        请决定：
        1. 是否需要创建新的知识节点
        2. 新节点的标题和内容
        3. 新节点应该归属于哪个分类
        4. 节点的重要程度评级
        5. 与现有节点的关联关系

        请提供结构化的更新建议。
        """

        context = {
            "user_input": user_input,
            "learning_output": learning_output,
            "questioning_output": questioning_output
        }

        return await self.generate_response(decision_prompt, context)