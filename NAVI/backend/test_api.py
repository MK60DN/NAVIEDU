import os
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()


async def test_api():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    print(f"Testing API Key: {api_key[:10]}...")

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            print(f"Status: {response.status}")
            result = await response.text()
            print(f"Response: {result}")


asyncio.run(test_api())