from typing import List, Dict, Any, Optional, Tuple
import httpx
import json
import asyncio
from neo4j import GraphDatabase
from sqlalchemy.orm import Session
from app.models.user import User
from app.database import get_db
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_tutor")


class DeepSeekAITutorService:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, deepseek_key: str):
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.api_key = deepseek_key
        self.api_base = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 监控统计
        self.call_count = 0
        self.total_tokens = 0
        self.error_count = 0

    async def call_deepseek_api(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.7):
        """调用DeepSeek API"""
        start_time = time.time()
        self.call_count += 1

        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "stream": False
            }

            try:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"]

                # 统计Token使用
                input_tokens = sum(len(msg["content"]) for msg in messages) / 1.5
                output_tokens = len(content) / 1.5
                self.total_tokens += input_tokens + output_tokens

                duration = time.time() - start_time
                logger.info(f"DeepSeek API调用成功 - 耗时: {duration:.2f}s, 预估Token: {input_tokens + output_tokens:.0f}")

                return content

            except Exception as e:
                self.error_count += 1
                logger.error(f"DeepSeek API调用失败: {e}")
                return "抱歉，AI导师暂时无法响应，请稍后重试。"

    async def process_user_message(self, user_id: int, message: str, conversation_history: List[Dict]) -> Dict[
        str, Any]:
        """处理用户消息的主入口"""
        try:
            # 1. 分析用户意图
            intent = await self._analyze_intent(message, conversation_history)
            logger.info(f"用户{user_id}意图分析: {intent}")

            # 2. 根据意图执行相应操作
            if intent["type"] == "SEARCH":
                return await self._handle_knowledge_search(user_id, message, intent)
            elif intent["type"] == "PATH":
                return await self._handle_path_planning(user_id, message, intent)
            elif intent["type"] == "LEARN":
                return await self._handle_learning_assistance(user_id, message, intent)
            elif intent["type"] == "CONTRIBUTE":
                return await self._handle_contribution(user_id, message, intent)
            else:
                return await self._handle_general_chat(user_id, message, intent)

        except Exception as e:
            logger.error(f"处理用户消息失败: {e}")
            return {
                "type": "error",
                "content": "抱歉，系统出现了一些问题，请稍后重试。",
                "data": {}
            }

    async def _analyze_intent(self, message: str, history: List[Dict]) -> Dict[str, Any]:
        """使用DeepSeek分析用户意图"""
        system_prompt = """你是专业的学习意图分析专家。请分析用户输入的学习相关意图：

意图类型定义：
- SEARCH: 查询知识点（如"什么是Python装饰器"、"解释递归算法"）
- PATH: 学习路径规划（如"我想系统学习Python"、"零基础学编程"）
- LEARN: 学习辅导请求（如"我不理解这个概念"、"这里为什么这样写"）
- CONTRIBUTE: 提到新概念（如"asyncio怎么使用"、"FastAPI路由设计"）
- CHAT: 一般对话（如问候、感谢、闲聊）

请严格按以下JSON格式返回，不要添加任何解释：
{
  "type": "意图类型",
  "keywords": ["关键词1", "关键词2"],
  "confidence": 置信度数值,
  "reason": "判断理由"
}"""

        # 优化对话历史
        recent_history = self._optimize_history(history)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析这个输入的意图：{message}"}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=200, temperature=0.3)

        try:
            # 提取JSON内容
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            result = json.loads(json_str.strip())
            return result
        except Exception as e:
            logger.warning(f"意图分析JSON解析失败: {e}, 原始响应: {response}")
            return self._fallback_intent_analysis(message)

    def _fallback_intent_analysis(self, message: str) -> Dict[str, Any]:
        """降级的意图分析"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["是什么", "什么是", "解释", "讲解", "介绍", "定义"]):
            keywords = [w for w in message.split() if len(w) > 1][:3]
            return {"type": "SEARCH", "keywords": keywords, "confidence": 0.7, "reason": "关键词匹配-搜索"}
        elif any(word in message_lower for word in ["学习路径", "怎么学", "如何学习", "系统学习", "规划", "零基础"]):
            return {"type": "PATH", "keywords": [message], "confidence": 0.8, "reason": "关键词匹配-路径"}
        elif any(word in message_lower for word in ["不理解", "不懂", "帮我", "教我", "为什么", "怎么回事"]):
            return {"type": "LEARN", "keywords": [message], "confidence": 0.7, "reason": "关键词匹配-辅导"}
        else:
            return {"type": "CHAT", "keywords": [], "confidence": 0.5, "reason": "默认分类"}

    def _optimize_history(self, history: List[Dict], max_messages: int = 10) -> List[Dict]:
        """优化对话历史，控制Token数量"""
        if len(history) <= max_messages:
            return history
        return history[-max_messages:]

    async def _handle_knowledge_search(self, user_id: int, message: str, intent: Dict) -> Dict[str, Any]:
        """处理知识点搜索"""
        knowledge_points = self._search_neo4j_knowledge(intent["keywords"])

        if knowledge_points:
            response = await self._generate_knowledge_display(knowledge_points, message)
            return {
                "type": "knowledge_search",
                "content": response,
                "data": {
                    "knowledge_points": knowledge_points,
                    "has_results": True
                },
                "next_action": "wait_for_selection"
            }
        else:
            # 没找到知识点，可能是新的贡献点
            return await self._handle_contribution(user_id, message, intent)

    def _search_neo4j_knowledge(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """在Neo4j中搜索知识点"""
        if not keywords:
            return []

        try:
            with self.neo4j_driver.session() as session:
                # 构造搜索条件
                search_conditions = []
                for keyword in keywords[:3]:  # 限制搜索关键词数量
                    search_conditions.append(f"n.name CONTAINS '{keyword}' OR n.description CONTAINS '{keyword}'")

                if not search_conditions:
                    return []

                keyword_condition = " OR ".join(search_conditions)

                cypher_query = f"""
                MATCH (n:Knowledge)
                WHERE {keyword_condition}
                OPTIONAL MATCH (n)-[r]-(related:Knowledge)
                RETURN n.name as name, n.description as description, 
                       n.difficulty as difficulty, n.category as category,
                       collect(DISTINCT related.name) as related_topics,
                       n.estimated_time as estimated_time,
                       n.prerequisites as prerequisites
                ORDER BY 
                    CASE 
                        WHEN n.name CONTAINS '{keywords[0]}' THEN 1
                        WHEN n.description CONTAINS '{keywords[0]}' THEN 2
                        ELSE 3
                    END
                LIMIT 10
                """

                result = session.run(cypher_query)
                knowledge_points = []

                for record in result:
                    knowledge_points.append({
                        "name": record["name"],
                        "description": record["description"] or "暂无描述",
                        "difficulty": record.get("difficulty", "中级"),
                        "category": record.get("category", "编程基础"),
                        "related_topics": [t for t in (record["related_topics"] or []) if t],
                        "estimated_time": record.get("estimated_time", "30分钟"),
                        "prerequisites": record.get("prerequisites", "")
                    })

                return knowledge_points

        except Exception as e:
            logger.error(f"Neo4j查询失败: {e}")
            return []

    async def _generate_knowledge_display(self, knowledge_points: List[Dict], user_query: str) -> str:
        """生成知识点展示内容"""
        system_prompt = """你是资深的编程导师，擅长用通俗易懂的中文解释技术概念。

用户询问了知识点，我找到了相关内容。请你用亲切、专业的语气：
1. 简洁总结找到的知识点（突出最相关的）
2. 按学习的重要性和难度排序展示  
3. 询问用户具体想了解哪个方向
4. 了解用户的学习目标和基础水平

回复要求：
- 语言自然流畅，就像面对面交流
- 避免过于技术化的表述
- 适当使用emoji增加亲和力（🔍📚💡等）
- 引导用户主动思考
- 控制在200字以内"""

        knowledge_summary = "\n".join([
            f"• {kp['name']}: {kp['description']} (难度: {kp['difficulty']}, 时长: {kp['estimated_time']})"
            for kp in knowledge_points[:5]  # 限制显示数量
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户询问: {user_query}\n\n找到的知识点:\n{knowledge_summary}"}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=300, temperature=0.8)
        return response

    async def _handle_path_planning(self, user_id: int, message: str, intent: Dict) -> Dict[str, Any]:
        """处理学习路径规划"""
        # 从消息中解析学习目标
        learning_goal = await self._parse_learning_goals(message, intent)

        # 计算学习路径
        learning_paths = self._calculate_learning_path(learning_goal.get("start", "编程基础"),
                                                       learning_goal.get("end", learning_goal.get("topic", "Python基础")))

        if learning_paths:
            response = await self._generate_path_recommendation(learning_paths, message, learning_goal)
            return {
                "type": "learning_path",
                "content": response,
                "data": {
                    "paths": learning_paths,
                    "learning_goal": learning_goal
                },
                "next_action": "start_learning"
            }
        else:
            return {
                "type": "path_planning",
                "content": "我正在为您量身定制学习路径！请告诉我：\n\n1. 您的编程基础如何？（零基础/有基础/较熟练）\n2. 您的学习目标是什么？（如：做网站、数据分析、找工作）\n3. 每天可以投入多长时间学习？\n\n这样我就能为您规划出最适合的学习路径！ 🎯",
                "data": {"need_more_info": True},
                "next_action": "collect_goal_info"
            }

    async def _parse_learning_goals(self, message: str, intent: Dict) -> Dict[str, str]:
        """解析学习目标"""
        system_prompt = """从用户的学习请求中提取关键信息：

请提取：
1. 学习主题（如Python、数据结构、Web开发）
2. 起始水平（零基础、有基础、进阶等）
3. 具体目标（如找工作、做项目、考试等）

返回JSON格式：
{
  "topic": "学习主题",
  "level": "起始水平", 
  "goal": "具体目标"
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"分析这个学习请求：{message}"}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=150, temperature=0.3)

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            return json.loads(json_str.strip())
        except:
            # 降级处理
            return {
                "topic": intent["keywords"][0] if intent["keywords"] else "编程",
                "level": "基础",
                "goal": "系统学习"
            }

    def _calculate_learning_path(self, start_topic: str, end_topic: str) -> List[Dict[str, Any]]:
        """计算最短学习路径"""
        try:
            with self.neo4j_driver.session() as session:
                # 使用更灵活的路径查询
                cypher_query = f"""
                MATCH (start:Knowledge)
                WHERE start.name CONTAINS '{start_topic}' OR start.category CONTAINS '{start_topic}'
                WITH start
                MATCH (end:Knowledge)
                WHERE end.name CONTAINS '{end_topic}' OR end.category CONTAINS '{end_topic}'
                WITH start, end
                MATCH path = shortestPath((start)-[:PREREQUISITE*1..5]-(end))
                RETURN nodes(path) as learning_path,
                       length(path) as path_length
                ORDER BY path_length
                LIMIT 3

                UNION

                // 如果找不到直接路径，返回相关的学习序列
                MATCH (n:Knowledge)
                WHERE n.name CONTAINS '{end_topic}' OR n.category CONTAINS '{end_topic}'
                OPTIONAL MATCH (prereq:Knowledge)-[:PREREQUISITE]->(n)
                RETURN [prereq, n] as learning_path, 1 as path_length
                ORDER BY n.difficulty
                LIMIT 2
                """

                result = session.run(cypher_query)
                paths = []

                for record in result:
                    if not record["learning_path"]:
                        continue

                    path_nodes = []
                    for i, node in enumerate(record["learning_path"]):
                        if node:  # 确保节点存在
                            path_nodes.append({
                                "step": i + 1,
                                "name": node["name"],
                                "description": node.get("description", ""),
                                "difficulty": node.get("difficulty", "中级"),
                                "estimated_time": node.get("estimated_time", "30分钟"),
                                "category": node.get("category", "编程"),
                                "prerequisites": node.get("prerequisites", "")
                            })

                    if path_nodes:  # 只添加非空路径
                        paths.append({
                            "nodes": path_nodes,
                            "total_length": len(path_nodes),
                            "estimated_total_time": self._calculate_total_time(path_nodes),
                            "difficulty_level": self._calculate_avg_difficulty(path_nodes)
                        })

                return paths[:3]  # 返回最多3条路径

        except Exception as e:
            logger.error(f"学习路径计算失败: {e}")
            return []

    def _calculate_total_time(self, nodes: List[Dict]) -> str:
        """计算总学习时间"""
        total_minutes = 0
        for node in nodes:
            time_str = node.get("estimated_time", "30分钟")
            # 简单解析时间
            if "小时" in time_str:
                hours = int(''.join(filter(str.isdigit, time_str.split("小时")[0])))
                total_minutes += hours * 60
            elif "分钟" in time_str:
                minutes = int(''.join(filter(str.isdigit, time_str.split("分钟")[0])))
                total_minutes += minutes
            else:
                total_minutes += 30  # 默认30分钟

        if total_minutes >= 60:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}小时{minutes}分钟" if minutes > 0 else f"{hours}小时"
        else:
            return f"{total_minutes}分钟"

    def _calculate_avg_difficulty(self, nodes: List[Dict]) -> str:
        """计算平均难度"""
        difficulty_scores = {"入门": 1, "初级": 2, "中级": 3, "高级": 4, "专家": 5}

        scores = []
        for node in nodes:
            difficulty = node.get("difficulty", "中级")
            scores.append(difficulty_scores.get(difficulty, 3))

        if not scores:
            return "中级"

        avg_score = sum(scores) / len(scores)

        for difficulty, score in difficulty_scores.items():
            if abs(avg_score - score) < 0.5:
                return difficulty

        return "中级"

    async def _generate_path_recommendation(self, learning_paths: List[Dict], message: str, learning_goal: Dict) -> str:
        """生成学习路径推荐"""
        system_prompt = """你是经验丰富的学习规划师，善于设计循序渐进的学习路径。

请用清晰易懂的方式：
1. 介绍推荐的学习路径（选择最适合的一条重点介绍）
2. 解释每个步骤的重要性和学习要点
3. 给出实用的学习建议和注意事项
4. 鼓励用户，让他们对学习充满信心

要求：
- 使用清晰的结构（可以用数字编号）
- 每个步骤都要说明具体学什么
- 提供实际的时间估算
- 语气积极正面，适当使用emoji
- 控制在300字以内"""

        paths_summary = ""
        for i, path in enumerate(learning_paths[:2]):  # 最多展示2条路径
            paths_summary += f"\n路径{i + 1} (总时长: {path['estimated_total_time']}, 难度: {path['difficulty_level']}):\n"
            for step in path['nodes'][:5]:  # 每条路径最多展示5步
                paths_summary += f"  {step['step']}. {step['name']} ({step['estimated_time']})\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户请求: {message}\n学习目标: {learning_goal}\n\n推荐路径:{paths_summary}"}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=400, temperature=0.8)
        return response

    async def _handle_learning_assistance(self, user_id: int, message: str, intent: Dict) -> Dict[str, Any]:
        """处理学习辅导"""
        # 获取用户当前学习上下文
        current_topic = await self._get_user_current_topic(user_id)

        system_prompt = f"""你是耐心的编程老师，擅长启发式教学。

当前学习主题：{current_topic or "编程学习"}
用户问题：{message}

教学原则：
1. 先理解用户的困惑点在哪里
2. 用生动的比喻和实例来解释
3. 循序渐进，由浅入深
4. 提出1-2个启发性问题让用户思考
5. 鼓励用户动手实践

回复要求：
- 解释要通俗易懂，避免过于技术化
- 多用实际的代码例子（如果适用）
- 适时提问检验理解程度
- 保持鼓励和耐心的态度
- 适当使用emoji增加亲和力
- 控制在250字以内"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=400, temperature=0.7)

        return {
            "type": "learning_assistance",
            "content": response,
            "data": {
                "current_topic": current_topic,
                "suggestions": await self._get_learning_suggestions(message)
            },
            "next_action": "continue_learning"
        }

    async def _get_user_current_topic(self, user_id: int) -> Optional[str]:
        """获取用户当前学习主题"""
        # 这里应该查询数据库获取用户的学习进度
        # 暂时返回默认值
        return "Python编程"

    async def _get_learning_suggestions(self, message: str) -> List[str]:
        """根据用户问题生成学习建议"""
        suggestions = [
            "可以尝试动手写一个简单的例子",
            "建议复习相关的基础概念",
            "可以在线搜索更多实际应用案例"
        ]
        return suggestions

    async def _handle_contribution(self, user_id: int, message: str, intent: Dict) -> Dict[str, Any]:
        """处理知识贡献"""
        # 检查是否包含新概念
        new_concepts = self._identify_new_concepts(message, intent["keywords"])

        if new_concepts:
            system_prompt = f"""用户提到了一个新概念：{new_concepts[0]}

请友好地：
1. 确认这是有价值的知识点
2. 询问用户是否愿意帮助完善这个内容  
3. 简单说明需要提供的信息：概念定义、应用场景、学习建议
4. 让用户感觉在参与有意义的知识共建
5. 提及会有代币奖励

话术要求：
- 自然友好，不要打断学习流程
- 突出贡献的价值和意义
- 适当使用emoji
- 控制在150字以内"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]

            response = await self.call_deepseek_api(messages, max_tokens=250, temperature=0.8)

            return {
                "type": "contribution_request",
                "content": response,
                "data": {
                    "new_concepts": new_concepts,
                    "contribution_requested": True
                },
                "next_action": "collect_contribution"
            }
        else:
            # 没有新概念，转为一般对话
            return await self._handle_general_chat(user_id, message, intent)

    def _identify_new_concepts(self, message: str, keywords: List[str]) -> List[str]:
        """识别消息中的新概念"""
        new_concepts = []

        try:
            with self.neo4j_driver.session() as session:
                # 检查关键词是否在知识图谱中
                for keyword in keywords[:3]:  # 限制检查数量
                    if len(keyword) < 2:  # 跳过过短的关键词
                        continue

                    result = session.run(
                        "MATCH (n:Knowledge) WHERE n.name CONTAINS $name OR n.description CONTAINS $name RETURN count(n) as count",
                        name=keyword
                    )

                    record = result.single()
                    if record and record["count"] == 0:
                        new_concepts.append(keyword)

                return new_concepts[:2]  # 最多返回2个新概念

        except Exception as e:
            logger.error(f"新概念识别失败: {e}")
            return []

    async def _handle_general_chat(self, user_id: int, message: str, intent: Dict) -> Dict[str, Any]:
        """处理一般对话"""
        system_prompt = """你是友好的AI学习助手，名叫"智学"。保持自然对话的同时，适时引导用户进行学习。

特点：
- 亲切友好，富有耐心
- 对学习话题很感兴趣
- 善于鼓励和motivate用户
- 会适时推荐学习内容
- 使用适当的emoji

回复要求：
- 简洁友好，不要太长
- 如果合适，可以推荐学习内容
- 保持积极正面的态度
- 控制在100字以内"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        response = await self.call_deepseek_api(messages, max_tokens=200, temperature=0.8)

        return {
            "type": "general_chat",
            "content": response,
            "data": {
                "chat_suggestions": [
                    "想学点什么新技能吗？",
                    "有什么编程问题可以问我",
                    "可以帮你规划学习路径哦"
                ]
            },
            "next_action": "continue_chat"
        }

    async def add_knowledge_contribution(self, user_id: int, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加用户贡献的知识点"""
        try:
            with self.neo4j_driver.session() as session:
                # 创建新的知识节点
                cypher_query = """
                CREATE (n:Knowledge {
                    name: $name,
                    description: $description,
                    difficulty: $difficulty,
                    category: $category,
                    created_by: $user_id,
                    created_at: datetime(),
                    status: 'pending_review',
                    estimated_time: $estimated_time,
                    prerequisites: $prerequisites
                })
                RETURN n.name as created_node
                """

                result = session.run(
                    cypher_query,
                    name=concept_data["name"],
                    description=concept_data["description"],
                    difficulty=concept_data.get("difficulty", "中级"),
                    category=concept_data.get("category", "编程"),
                    estimated_time=concept_data.get("estimated_time", "30分钟"),
                    prerequisites=concept_data.get("prerequisites", ""),
                    user_id=user_id
                )

                record = result.single()
                created_node = record["created_node"] if record else concept_data["name"]

                return {
                    "success": True,
                    "created_node": created_node,
                    "message": f"🎉 感谢您的贡献！'{created_node}'已添加到知识图谱中，正在等待审核。\n\n您已获得10个$PYTHON代币奖励！继续贡献更多有价值的内容吧！"
                }

        except Exception as e:
            logger.error(f"添加知识贡献失败: {e}")
            return {
                "success": False,
                "message": "添加失败，请稍后重试。"
            }

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "total_calls": self.call_count,
            "estimated_tokens": int(self.total_tokens),
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1) * 100
        }

    def close(self):
        """关闭数据库连接"""
        if self.neo4j_driver:
            self.neo4j_driver.close()