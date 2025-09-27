import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
import numpy as np
import time
from abc import ABC, abstractmethod


class DeepSeekBaseAgent(ABC):
    """基于DeepSeek的智能体基类"""

    def __init__(self,
                 agent_id: str,
                 role: str,
                 base_weight: float,
                 api_key: str,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):

        self.agent_id = agent_id
        self.role = role
        self.base_weight = base_weight
        self.current_weight = base_weight
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # 智能体个性配置
        self.system_prompt = self._create_system_prompt()
        self.conversation_history = []
        self.knowledge_base = {}

        # API调用统计
        self.api_call_count = 0
        self.total_tokens = 0
        self.last_response_time = 0

    @abstractmethod
    def _create_system_prompt(self) -> str:
        """创建系统提示词（子类必须重写）"""
        pass

    async def call_deepseek_api(self, messages: List[Dict]) -> Dict:
        """调用DeepSeek API"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self._get_temperature(),
            "max_tokens": self._get_max_tokens(),
            "stream": False
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.api_call_count += 1
                        self.total_tokens += data.get('usage', {}).get('total_tokens', 0)
                        self.last_response_time = time.time() - start_time
                        return data
                    else:
                        error_text = await response.text()
                        raise Exception(f"API调用失败: {response.status} - {error_text}")
        except Exception as e:
            print(f"DeepSeek API调用错误: {e}")
            return self._get_fallback_response()

    def _get_temperature(self) -> float:
        """根据智能体角色调整温度参数"""
        temperature_map = {
            "learning_agent": 0.3,  # 较低温度，更确定性
            "questioning_agent": 0.7,  # 较高温度，更多创造性
            "balancing_agent": 0.5,  # 中等温度
        }
        return temperature_map.get(self.agent_id, 0.5)

    def _get_max_tokens(self) -> int:
        """根据角色调整最大token数"""
        token_map = {
            "learning_agent": 2000,  # 详细解释需要更多token
            "questioning_agent": 1500,  # 问题通常较短
            "balancing_agent": 1000,  # 协调信息较短
        }
        return token_map.get(self.agent_id, 1500)

    async def generate_response(self, user_input: str, context: Dict) -> Dict:
        """生成智能体响应"""
        messages = self._build_message_sequence(user_input, context)

        response_data = await self.call_deepseek_api(messages)

        if 'choices' in response_data and len(response_data['choices']) > 0:
            content = response_data['choices'][0]['message']['content']
            return self._process_response(content, context)
        else:
            return self._get_fallback_response()

    def _build_message_sequence(self, user_input: str, context: Dict) -> List[Dict]:
        """构建消息序列"""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        # 添加上下文信息
        if context.get('conversation_history'):
            # 添加最近几轮对话历史
            history = context['conversation_history'][-3:]  # 最近3轮
            for turn in history:
                if isinstance(turn, dict):
                    messages.append({"role": "user", "content": turn.get('content', '')})
                    if 'response' in turn:
                        messages.append({"role": "assistant", "content": turn['response']})

        # 添加当前用户输入
        enhanced_input = self._enhance_user_input(user_input, context)
        messages.append({"role": "user", "content": enhanced_input})

        return messages

    def _enhance_user_input(self, user_input: str, context: Dict) -> str:
        """增强用户输入，添加上下文信息"""
        enhanced = f"用户输入: {user_input}\n\n"

        # 添加用户知识水平信息
        if context and 'knowledge_level' in context:
            enhanced += f"用户当前知识水平: {context['knowledge_level']}/1.0\n"

        # 添加学习目标信息
        if context and 'learning_goals' in context:
            enhanced += f"用户学习目标: {', '.join(context['learning_goals'])}\n"

        # 添加智能体特定指令
        enhanced += f"\n请以{self.role}的身份回应，专注于你的核心职责。"

        return enhanced

    def _process_response(self, raw_response: str, context: Dict) -> Dict:
        """处理原始API响应"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "content": raw_response,
            "timestamp": time.time(),
            "weight": self.current_weight,
            "metadata": {
                "tokens_used": context.get('tokens_used', 0),
                "response_quality": self._assess_response_quality(raw_response),
                "relevance_score": self._calculate_relevance(raw_response, context),
                "response_time": self.last_response_time
            }
        }

    def _assess_response_quality(self, response: str) -> float:
        """评估响应质量"""
        if not response:
            return 0.0

        # 简单的质量评估指标
        length_score = min(len(response) / 500, 1.0)  # 长度适中
        structure_score = 1.0 if any(marker in response for marker in ['。', '？', '！', '\n']) else 0.5

        return (length_score + structure_score) / 2

    def _calculate_relevance(self, response: str, context: Dict) -> float:
        """计算响应相关性"""
        if not response or not context:
            return 0.5

        # 简单的相关性计算（实际应用中可能需要更复杂的算法）
        return 0.8  # 默认相关性

    def _get_fallback_response(self) -> Dict:
        """获取后备响应"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "content": f"抱歉，我暂时无法处理您的请求。请稍后重试。",
            "timestamp": time.time(),
            "weight": self.current_weight,
            "metadata": {
                "is_fallback": True,
                "response_quality": 0.3,
                "relevance_score": 0.3
            }
        }

    def update_weight(self, feedback_score: float):
        """根据反馈更新权重"""
        alpha = 0.1  # 学习率
        self.current_weight = self.current_weight * (1 - alpha) + feedback_score * alpha

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "api_call_count": self.api_call_count,
            "total_tokens": self.total_tokens,
            "current_weight": self.current_weight,
            "avg_response_time": self.last_response_time
        }