from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import asyncio

from app.database import get_db
from app.services.ai_tutor_service import DeepSeekAITutorService
from app.models.user import User
from app.api.deps import get_current_user
from app.config import settings
import logging

logger = logging.getLogger("ai_tutor_api")

router = APIRouter()

# 初始化AI导师服务
ai_tutor_service = None


def get_ai_tutor_service():
    global ai_tutor_service
    if ai_tutor_service is None:
        ai_tutor_service = DeepSeekAITutorService(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD,
            deepseek_key=settings.DEEPSEEK_API_KEY
        )
    return ai_tutor_service


class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="用户消息")
    conversation_history: List[Dict[str, Any]] = Field(default=[], description="对话历史")


class ContributionData(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="概念名称")
    description: str = Field(..., min_length=10, max_length=1000, description="详细描述")
    difficulty: str = Field(default="中级", description="难度级别")
    category: str = Field(default="编程", description="分类")
    estimated_time: str = Field(default="30分钟", description="预估学习时间")
    prerequisites: str = Field(default="", description="前置知识")


class QuickSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键词")


@router.post("/chat")
async def chat_with_ai_tutor(
        chat_data: ChatMessage,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """与AI学习导师对话"""
    try:
        service = get_ai_tutor_service()

        # 处理用户消息
        response = await service.process_user_message(
            user_id=current_user.id,
            message=chat_data.message,
            conversation_history=chat_data.conversation_history
        )

        # 后台任务：记录对话
        background_tasks.add_task(
            log_conversation,
            current_user.id,
            chat_data.message,
            response
        )

        return {
            "success": True,
            "data": response,
            "user_id": current_user.id,
            "timestamp": int(time.time())
        }

    except Exception as e:
        logger.error(f"AI导师对话失败 - 用户{current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "AI导师暂时无法响应，请稍后重试",
                "error_type": "service_error"
            }
        )


@router.post("/contribute")
async def contribute_knowledge(
        contribution: ContributionData,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """用户贡献知识点"""
    try:
        service = get_ai_tutor_service()

        result = await service.add_knowledge_contribution(
            user_id=current_user.id,
            concept_data=contribution.dict()
        )

        if result["success"]:
            # 后台任务：奖励代币
            background_tasks.add_task(
                reward_contribution_tokens,
                current_user.id,
                contribution.name,
                10  # 奖励10个代币
            )

        return {
            "success": result["success"],
            "data": result,
            "message": result["message"]
        }

    except Exception as e:
        logger.error(f"知识贡献失败 - 用户{current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "贡献提交失败，请稍后重试",
                "error_type": "contribution_error"
            }
        )


@router.post("/search")
async def search_knowledge(
        search_data: QuickSearchRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """快速搜索知识点"""
    try:
        service = get_ai_tutor_service()

        knowledge_points = service._search_neo4j_knowledge([search_data.query])

        return {
            "success": True,
            "data": {
                "knowledge_points": knowledge_points,
                "total": len(knowledge_points),
                "query": search_data.query
            }
        }

    except Exception as e:
        logger.error(f"知识搜索失败 - 用户{current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "搜索失败，请稍后重试",
                "error_type": "search_error"
            }
        )


@router.get("/learning-path/{start_topic}/{end_topic}")
async def get_learning_path(
        start_topic: str,
        end_topic: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """获取学习路径"""
    try:
        service = get_ai_tutor_service()

        paths = service._calculate_learning_path(start_topic, end_topic)

        return {
            "success": True,
            "data": {
                "paths": paths,
                "start_topic": start_topic,
                "end_topic": end_topic,
                "path_count": len(paths)
            }
        }

    except Exception as e:
        logger.error(f"学习路径计算失败 - 用户{current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "路径计算失败，请稍后重试",
                "error_type": "path_error"
            }
        )


@router.get("/stats")
async def get_ai_tutor_stats(
        current_user: User = Depends(get_current_user)
):
    """获取AI导师使用统计"""
    try:
        service = get_ai_tutor_service()
        stats = service.get_usage_stats()

        return {
            "success": True,
            "data": stats
        }

    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return {
            "success": False,
            "data": {"message": "统计数据暂时无法获取"}
        }


@router.post("/feedback")
async def submit_feedback(
        feedback_data: Dict[str, Any],
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """提交用户反馈"""
    try:
        # 这里应该保存用户反馈到数据库
        logger.info(f"用户{current_user.id}反馈: {feedback_data}")

        return {
            "success": True,
            "message": "感谢您的反馈！我们会持续改进AI导师的服务质量。"
        }

    except Exception as e:
        logger.error(f"反馈提交失败: {e}")
        raise HTTPException(status_code=500, detail="反馈提交失败")


# 后台任务函数
async def log_conversation(user_id: int, user_message: str, ai_response: Dict[str, Any]):
    """记录对话日志"""
    try:
        # 这里应该保存到数据库
        logger.info(f"对话记录 - 用户{user_id}: {user_message[:50]}...")
    except Exception as e:
        logger.error(f"对话记录失败: {e}")


async def reward_contribution_tokens(user_id: int, concept_name: str, token_amount: int):
    """奖励贡献代币"""
    try:
        # 这里应该调用代币服务增加用户代币
        logger.info(f"代币奖励 - 用户{user_id}获得{token_amount}个代币，因为贡献了'{concept_name}'")
    except Exception as e:
        logger.error(f"代币奖励失败: {e}")