import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import aiofiles

logger = logging.getLogger(__name__)


class StorageService:
    """存储服务 - 处理数据持久化"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.ensure_data_directory()

    def ensure_data_directory(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    async def save_session(self, user_id: str, session_data: Dict) -> bool:
        """保存用户会话数据"""
        try:
            file_path = os.path.join(self.data_dir, f"session_{user_id}.json")

            # 添加时间戳
            session_data["updated_at"] = datetime.now().isoformat()

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(session_data, ensure_ascii=False, indent=2))

            logger.info(f"保存会话数据: {user_id}")
            return True

        except Exception as e:
            logger.error(f"保存会话数据失败: {e}")
            return False

    async def load_session(self, user_id: str) -> Optional[Dict]:
        """加载用户会话数据"""
        try:
            file_path = os.path.join(self.data_dir, f"session_{user_id}.json")

            if not os.path.exists(file_path):
                return None

            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)

        except Exception as e:
            logger.error(f"加载会话数据失败: {e}")
            return None

    async def save_knowledge_graph(self, user_id: str, graph_data: Dict) -> bool:
        """保存知识图谱数据"""
        try:
            file_path = os.path.join(self.data_dir, f"knowledge_graph_{user_id}.json")

            # 添加版本信息和时间戳
            graph_data["version"] = "1.0.0"
            graph_data["updated_at"] = datetime.now().isoformat()

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(graph_data, ensure_ascii=False, indent=2))

            logger.info(f"保存知识图谱: {user_id}")
            return True

        except Exception as e:
            logger.error(f"保存知识图谱失败: {e}")
            return False

    async def load_knowledge_graph(self, user_id: str) -> Optional[Dict]:
        """加载知识图谱数据"""
        try:
            file_path = os.path.join(self.data_dir, f"knowledge_graph_{user_id}.json")

            if not os.path.exists(file_path):
                return None

            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)

        except Exception as e:
            logger.error(f"加载知识图谱失败: {e}")
            return None

    async def backup_user_data(self, user_id: str) -> bool:
        """备份用户数据"""
        try:
            backup_dir = os.path.join(self.data_dir, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"backup_{user_id}_{timestamp}.json")

            # 收集所有用户数据
            session_data = await self.load_session(user_id)
            knowledge_graph = await self.load_knowledge_graph(user_id)

            backup_data = {
                "user_id": user_id,
                "backup_time": datetime.now().isoformat(),
                "session_data": session_data,
                "knowledge_graph": knowledge_graph
            }

            async with aiofiles.open(backup_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(backup_data, ensure_ascii=False, indent=2))

            logger.info(f"备份用户数据: {user_id} -> {backup_file}")
            return True

        except Exception as e:
            logger.error(f"备份用户数据失败: {e}")
            return False

    async def cleanup_old_data(self, days: int = 30) -> int:
        """清理过期数据"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0

            for filename in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, filename)

                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                    if file_time < cutoff_date:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"删除过期文件: {filename}")

            return deleted_count

        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
            return 0

    async def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        try:
            total_size = 0
            file_count = 0

            for filename in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "data_directory": self.data_dir
            }

        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}