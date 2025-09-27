from .base_agent import DeepSeekBaseAgent
from typing import Dict, List


class DeepSeekLearningAgent(DeepSeekBaseAgent):
    """基于DeepSeek的学习助手"""

    def __init__(self, agent_id: str, api_key: str, model: str = "deepseek-chat"):
        super().__init__(
            agent_id=agent_id,
            role="学习辅导助手",
            base_weight=1.618,  # 黄金比例
            api_key=api_key,
            model=model
        )

        self.learning_strategies = [
            "渐进式学习", "支架式教学", "示例引导", "概念分解"
        ]

        self.knowledge_domains = [
            "数学", "科学", "语言", "历史", "艺术", "技术"
        ]

    def _create_system_prompt(self) -> str:
        return """你是Navi的学习辅导助手，专注于系统化的知识传授和学习指导。

你的核心职责：
1. 提供结构化、准确的知识解释
2. 设计渐进式学习路径
3. 识别和填补知识缺口
4. 提供具体的示例和应用场景
5. 根据用户水平调整教学深度

教学指导原则：
- 始终从基础概念开始，逐步深入
- 使用具体例子说明抽象概念
- 关注知识的实际应用价值
- 适当时提供总结和复习要点
- 鼓励用户主动思考和提问

风格要求：
- 清晰、系统、有耐心
- 鼓励性、支持性语气
- 注重逻辑性和连贯性
- 避免过于技术化的术语

请确保你的回答既有深度又易于理解，帮助用户建立扎实的知识体系。根据用户的学习目标和当前水平，选择最合适的教学策略。"""

    def _enhance_user_input(self, user_input: str, context: Dict) -> str:
        enhanced = super()._enhance_user_input(user_input, context)

        # 安全检查并添加学习进度信息
        if context and 'learning_progress' in context:
            progress = context['learning_progress']
            if progress:
                enhanced += f"\n当前学习进度: {progress.get('current_topic', '未知')}, "
                enhanced += f"掌握程度: {progress.get('mastery_level', 0)}/1.0"

        # 安全检查并添加知识图谱上下文
        if context and 'knowledge_graph' in context and context['knowledge_graph'] is not None:
            graph = context['knowledge_graph']
            enhanced += f"\n\n用户知识图谱信息："
            enhanced += f"- 根节点: {graph.get('title', '未知')}"
            if 'children' in graph and isinstance(graph['children'], list) and len(graph['children']) > 0:
                enhanced += f"- 已有知识点: {len(graph['children'])}个"

        # 添加学习策略提示
        enhanced += f"\n\n请根据用户当前水平选择合适的教学策略：{', '.join(self.learning_strategies)}"
        enhanced += "\n重点关注概念的深度理解和实际应用。"

        return enhanced

    def _assess_response_quality(self, response: str) -> float:
        """评估学习助手响应质量"""
        quality_score = super()._assess_response_quality(response)

        # 学习助手特有的质量指标
        educational_markers = [
            "例如", "比如", "首先", "其次", "总结", "重点", "关键", "应用", "练习"
        ]

        educational_score = sum(1 for marker in educational_markers if marker in response) / len(educational_markers)

        return (quality_score + educational_score) / 2

    async def create_learning_path(self, topic: str, user_level: str) -> Dict:
        """创建学习路径"""
        learning_path_prompt = f"""
        为用户创建关于"{topic}"的学习路径。
        用户当前水平：{user_level}

        请提供：
        1. 学习目标
        2. 先决知识
        3. 学习步骤（按难度递增）
        4. 实践建议
        5. 检验方法
        """

        context = {"topic": topic, "level": user_level}
        return await self.generate_response(learning_path_prompt, context)