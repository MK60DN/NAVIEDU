import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from backend.agents.learning_agent import DeepSeekLearningAgent
from backend.agents.questioning_agent import DeepSeekQuestioningAgent
from backend.agents.balancing_agent import DeepSeekBalancingAgent


class TestAgents:
    @pytest.fixture
    def learning_agent(self):
        return DeepSeekLearningAgent("test_learning", "test_api_key")

    @pytest.fixture
    def questioning_agent(self):
        return DeepSeekQuestioningAgent("test_questioning", "test_api_key")

    @pytest.fixture
    def balancing_agent(self):
        return DeepSeekBalancingAgent("test_balancing", "test_api_key")

    @pytest.mark.asyncio
    async def test_learning_agent_response(self, learning_agent):
        """测试学习智能体响应"""
        with patch.object(learning_agent, 'call_deepseek_api') as mock_api:
            mock_api.return_value = {
                'choices': [{'message': {'content': '这是学习助手的回应'}}]
            }

            response = await learning_agent.generate_response(
                "什么是机器学习？",
                {"conversation_history": []}
            )

            assert response['role'] == '学习辅导助手'
            assert '学习助手' in response['content']
            mock_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_questioning_agent_response(self, questioning_agent):
        """测试质疑智能体响应"""
        with patch.object(questioning_agent, 'call_deepseek_api') as mock_api:
            mock_api.return_value = {
                'choices': [{'message': {'content': '让我们质疑一下这个观点'}}]
            }

            response = await questioning_agent.generate_response(
                "人工智能是万能的",
                {"learning_context": []}
            )

            assert response['role'] == '批判思考助手'
            assert '质疑' in response['content']

    def test_agent_weight_initialization(self, learning_agent, questioning_agent, balancing_agent):
        """测试智能体权重初始化"""
        assert learning_agent.base_weight == 1.618  # 黄金比例
        assert questioning_agent.base_weight == 1.414  # 根号2
        assert balancing_agent.base_weight == 1.0  # 中性权重

    def test_temperature_settings(self, learning_agent, questioning_agent, balancing_agent):
        """测试温度参数设置"""
        assert learning_agent._get_temperature() == 0.3  # 较低，更确定性
        assert questioning_agent._get_temperature() == 0.7  # 较高，更创造性
        assert balancing_agent._get_temperature() == 0.5  # 中等