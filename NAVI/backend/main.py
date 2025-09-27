import os
import sys

# 设置环境变量强制使用UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 重新配置标准输出编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from dotenv import load_dotenv
# ... 其他导入代码
import os
from dotenv import load_dotenv

# 在其他导入之前加载环境变量
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

from agents.learning_agent import DeepSeekLearningAgent
from agents.questioning_agent import DeepSeekQuestioningAgent
from agents.balancing_agent import DeepSeekBalancingAgent
from services.knowledge_service import KnowledgeGraphService

app = FastAPI(title="Navi API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
learning_agent = None
questioning_agent = None
balancing_agent = None
knowledge_service = KnowledgeGraphService()


class ChatRequest(BaseModel):
    message: str
    context: Optional[List[Dict]] = []
    knowledge_graph: Optional[Dict] = None


class ChatResponse(BaseModel):
    content: str
    type: str
    metadata: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    global learning_agent, questioning_agent, balancing_agent

    # 从环境变量读取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("警告：未找到 DEEPSEEK_API_KEY 环境变量")
        api_key = "test_key"
    else:
        print(f"API Key 加载成功: {api_key[:10]}...")

    learning_agent = DeepSeekLearningAgent("learning_agent", api_key)
    questioning_agent = DeepSeekQuestioningAgent("questioning_agent", api_key)
    balancing_agent = DeepSeekBalancingAgent("balancing_agent", api_key)

    print("所有智能体初始化完成")


@app.get("/")
async def root():
    return {"message": "Navi API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Navi API"}


@app.post("/api/learning", response_model=ChatResponse)
async def learning_chat(request: ChatRequest):
    try:
        print(f"[DEBUG] 收到学习请求: {request.message}")
        print(f"[DEBUG] learning_agent 是否存在: {learning_agent is not None}")

        if learning_agent is None:
            print("[ERROR] learning_agent 未初始化!")
            raise HTTPException(status_code=500, detail="学习智能体未初始化")

        context = {
            'conversation_history': request.context,
            'knowledge_graph': request.knowledge_graph
        }

        print(f"[DEBUG] 调用 learning_agent.generate_response...")
        response = await learning_agent.generate_response(request.message, context)
        print(f"[DEBUG] 智能体响应: {response}")

        if response is None:
            print("[ERROR] 智能体返回None!")
            raise HTTPException(status_code=500, detail="智能体响应为空")

        return ChatResponse(
            content=response.get('content', '响应内容为空'),
            type="learning",
            metadata=response.get('metadata', {})
        )
    except Exception as e:
        print(f"[ERROR] 学习API错误: {str(e)}")
        print(f"[ERROR] 错误类型: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/questioning", response_model=ChatResponse)
async def questioning_chat(request: ChatRequest):
    try:
        context = {
            'conversation_history': request.context,
            'learning_context': request.context
        }

        response = await questioning_agent.generate_response(request.message, context)

        return ChatResponse(
            content=response['content'],
            type="questioning",
            metadata=response.get('metadata', {})
        )
    except Exception as e:
        print(f"质疑API错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def general_chat(request: ChatRequest):
    try:
        # 使用平衡智能体进行一般对话
        context = {
            'conversation_history': request.context
        }

        response = await balancing_agent.generate_response(request.message, context)

        return ChatResponse(
            content=response['content'],
            type="chat",
            metadata=response.get('metadata', {})
        )
    except Exception as e:
        print(f"对话API错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/knowledge/update")
async def update_knowledge_graph(knowledge_data: Dict):
    try:
        updated_graph = knowledge_service.update_graph(knowledge_data)
        return {"status": "success", "graph": updated_graph}
    except Exception as e:
        print(f"知识图谱更新错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    try:
        # 这里可以添加知识图谱统计逻辑
        return {
            "total_nodes": 0,
            "total_connections": 0,
            "last_updated": None
        }
    except Exception as e:
        print(f"获取知识图谱统计错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/status")
async def get_agents_status():
    try:
        status = {
            "learning_agent": {
                "initialized": learning_agent is not None,
                "stats": learning_agent.get_stats() if learning_agent else None
            },
            "questioning_agent": {
                "initialized": questioning_agent is not None,
                "stats": questioning_agent.get_stats() if questioning_agent else None
            },
            "balancing_agent": {
                "initialized": balancing_agent is not None,
                "stats": balancing_agent.get_stats() if balancing_agent else None
            }
        }
        return status
    except Exception as e:
        print(f"获取智能体状态错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("启动Navi API服务器...")
    print("API文档地址: http://localhost:8000/docs")
    print("健康检查地址: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)