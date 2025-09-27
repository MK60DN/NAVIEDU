import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)


class DeepSeekService:
    """DeepSeek API服务封装"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.request_count = 0
        self.total_tokens = 0

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def chat_completion(self,
                              messages: List[Dict],
                              model: str = "deepseek-chat",
                              temperature: float = 0.7,
                              max_tokens: int = 2000,
                              stream: bool = False) -> Dict:
        """
        调用DeepSeek聊天完成API

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否使用流式输出

        Returns:
            API响应数据
        """

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        start_time = time.time()

        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()

                    # 更新统计信息
                    self.request_count += 1
                    usage = data.get('usage', {})
                    self.total_tokens += usage.get('total_tokens', 0)

                    # 记录响应时间
                    response_time = time.time() - start_time

                    logger.info(f"DeepSeek API调用成功 - 响应时间: {response_time:.2f}s, "
                                f"Token使用: {usage.get('total_tokens', 0)}")

                    return {
                        "success": True,
                        "data": data,
                        "response_time": response_time,
                        "tokens_used": usage.get('total_tokens', 0)
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API错误: {response.status} - {error_text}")
                    return {
                        "success": False,
                        "error": f"API错误: {response.status}",
                        "details": error_text
                    }

        except asyncio.TimeoutError:
            logger.error("DeepSeek API调用超时")
            return {
                "success": False,
                "error": "请求超时",
                "details": "API调用超时，请稍后重试"
            }
        except Exception as e:
            logger.error(f"DeepSeek API调用异常: {str(e)}")
            return {
                "success": False,
                "error": "网络错误",
                "details": str(e)
            }

    async def stream_chat_completion(self,
                                     messages: List[Dict],
                                     model: str = "deepseek-chat",
                                     temperature: float = 0.7,
                                     max_tokens: int = 2000):
        """
        流式聊天完成

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数

        Yields:
            流式响应数据块
        """

        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                yield data
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    yield {
                        "error": f"API错误: {response.status}",
                        "details": error_text
                    }

        except Exception as e:
            yield {
                "error": "网络错误",
                "details": str(e)
            }

    def get_stats(self) -> Dict:
        """获取API使用统计"""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / max(self.request_count, 1)
        }

    async def test_connection(self) -> bool:
        """测试API连接"""
        test_messages = [
            {"role": "user", "content": "Hello"}
        ]

        result = await self.chat_completion(test_messages, max_tokens=10)
        return result.get("success", False)