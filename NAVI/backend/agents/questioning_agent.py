from .base_agent import DeepSeekBaseAgent
from typing import Dict, List


class DeepSeekQuestioningAgent(DeepSeekBaseAgent):
    """基于DeepSeek的质疑思考助手"""

    def __init__(self, agent_id: str, api_key: str, model: str = "deepseek-chat"):
        super().__init__(
            agent_id=agent_id,
            role="批判思考助手",
            base_weight=1.414,  # 根号2，体现理性思维
            api_key=api_key,
            model=model
        )

        self.questioning_strategies = [
            "苏格拉底式提问", "批判性分析", "逆向思维", "假设反驳"
        ]

        self.thinking_frameworks = [
            "5W1H分析", "SWOT分析", "因果关系", "类比推理"
        ]

    def _create_system_prompt(self) -> str:
        return """你是Navi的批判思考助手，专注于培养用户的批判性思维和深度思考能力。

你的核心职责：
1. 对用户的观点和学习内容提出有价值的质疑
2. 引导用户进行更深层次的思考
3. 帮助用户发现盲点和逻辑漏洞
4. 提供不同角度的视角和观点
5. 培养用户的独立思考能力

质疑思考原则：
- 善意的质疑，而非攻击性批评
- 基于逻辑和事实的分析
- 鼓励多角度思考问题
- 帮助用户建立批判性思维框架
- 质疑的同时提供建设性建议

提问策略：
- 使用开放性问题引导思考
- 挑战假设和既定观念
- 探索因果关系和相关性
- 寻找反例和特殊情况
- 引导用户思考更深层的含义

风格要求：
- 友善但具有挑战性
- 逻辑清晰、论证有力
- 启发性强，避免直接给出答案
- 尊重用户观点的同时提出质疑

请通过巧妙的提问和质疑，帮助用户培养批判性思维，深化对问题的理解。"""

    def _enhance_user_input(self, user_input: str, context: Dict) -> str:
        enhanced = super()._enhance_user_input(user_input, context)

        # 添加学习上下文（来自学习助手的内容）- 修复None检查
        if 'learning_context' in context and context['learning_context'] is not None:
            learning_content = context['learning_context']
            if learning_content:
                enhanced += f"\n\n学习助手刚才的内容："
                for msg in learning_content[-2:]:  # 最近两条
                    if isinstance(msg, dict) and msg.get('sender') == 'assistant':
                        enhanced += f"- {msg.get('content', '')[:200]}..."

        # 添加质疑策略提示
        enhanced += f"\n\n请运用以下思考框架进行质疑分析：{', '.join(self.thinking_frameworks)}"
        enhanced += f"\n建议的质疑策略：{', '.join(self.questioning_strategies)}"
        enhanced += "\n重点关注逻辑漏洞、假设前提、反例情况和深层含义。"

        return enhanced

    def _assess_response_quality(self, response: str) -> float:
        """评估质疑助手响应质量"""
        quality_score = super()._assess_response_quality(response)

        # 质疑助手特有的质量指标
        questioning_markers = [
            "为什么", "如果", "是否", "真的", "一定", "假设", "考虑", "思考", "质疑", "反思"
        ]

        questioning_score = sum(1 for marker in questioning_markers if marker in response) / len(questioning_markers)

        # 检查是否包含问题（以问号结尾的句子）
        question_count = response.count('？') + response.count('?')
        question_score = min(question_count / 3, 1.0)  # 期望3个问题左右

        return (quality_score + questioning_score + question_score) / 3

    async def generate_critical_questions(self, topic: str, content: str) -> Dict:
        """生成批判性问题"""
        critical_prompt = f"""
        针对主题"{topic}"和内容"{content[:500]}..."，

        请生成一系列批判性思考问题：
        1. 挑战基本假设的问题
        2. 探索因果关系的问题  
        3. 寻找反例的问题
        4. 深入含义的问题
        5. 替代观点的问题

        每个问题都应该能够促进深度思考。
        """

        context = {"topic": topic, "content": content}
        return await self.generate_response(critical_prompt, context)