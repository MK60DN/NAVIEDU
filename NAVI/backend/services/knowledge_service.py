import json
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """知识图谱服务"""

    def __init__(self):
        self.graph_cache = {}

    def create_default_graph(self) -> Dict:
        """创建默认知识图谱"""
        return {
            "id": "root",
            "title": "我的知识库",
            "content": "个人知识图谱根节点",
            "type": "system",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "children": [
                {
                    "id": "learning_notes",
                    "title": "学习笔记",
                    "content": "学习过程中的知识点记录",
                    "type": "system",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "children": []
                },
                {
                    "id": "questions",
                    "title": "问题思考",
                    "content": "质疑和批判性思考记录",
                    "type": "system",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "children": []
                }
            ]
        }

    def add_learning_node(self,
                          graph: Dict,
                          user_question: str,
                          learning_response: str,
                          questioning_response: Optional[str] = None) -> Dict:
        """添加学习节点到知识图谱"""

        # 创建新的学习节点
        new_node = {
            "id": f"learning_{uuid.uuid4().hex[:8]}",
            "title": self._extract_title(user_question),
            "content": self._create_learning_content(user_question, learning_response, questioning_response),
            "type": "learning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "user_question": user_question,
                "learning_response": learning_response[:500],  # 截断以节省空间
                "questioning_response": questioning_response[:500] if questioning_response else None,
                "keywords": self._extract_keywords(user_question + " " + learning_response)
            },
            "children": []
        }

        # 确定插入位置
        target_parent = self._find_best_parent(graph, new_node)

        # 插入节点
        updated_graph = self._insert_node(graph, target_parent["id"], new_node)

        logger.info(f"添加学习节点: {new_node['title']} 到父节点: {target_parent['title']}")

        return updated_graph

    def add_questioning_node(self,
                             graph: Dict,
                             user_question: str,
                             questioning_response: str) -> Dict:
        """添加质疑节点到知识图谱"""

        new_node = {
            "id": f"questioning_{uuid.uuid4().hex[:8]}",
            "title": f"质疑: {self._extract_title(user_question)}",
            "content": self._create_questioning_content(user_question, questioning_response),
            "type": "questioning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {
                "user_question": user_question,
                "questioning_response": questioning_response[:500],
                "keywords": self._extract_keywords(user_question + " " + questioning_response)
            },
            "children": []
        }

        # 质疑节点通常添加到"问题思考"分类下
        questions_parent = self._find_node_by_id(graph, "questions")
        if not questions_parent:
            questions_parent = self._find_node_by_title(graph, "问题思考")

        if questions_parent:
            updated_graph = self._insert_node(graph, questions_parent["id"], new_node)
        else:
            # 如果找不到问题思考节点，添加到根节点
            updated_graph = self._insert_node(graph, "root", new_node)

        logger.info(f"添加质疑节点: {new_node['title']}")

        return updated_graph

    def update_node(self, graph: Dict, node_id: str, updates: Dict) -> Dict:
        """更新知识图谱节点"""

        def update_recursive(node):
            if node["id"] == node_id:
                node.update(updates)
                node["updated_at"] = datetime.now().isoformat()
                return True

            if "children" in node:
                for child in node["children"]:
                    if update_recursive(child):
                        return True
            return False

        updated_graph = json.loads(json.dumps(graph))  # 深拷贝

        if update_recursive(updated_graph):
            logger.info(f"更新节点: {node_id}")
            return updated_graph
        else:
            logger.warning(f"未找到节点: {node_id}")
            return graph

    def delete_node(self, graph: Dict, node_id: str) -> Dict:
        """删除知识图谱节点"""

        def delete_recursive(node):
            if "children" in node:
                node["children"] = [
                    child for child in node["children"]
                    if child["id"] != node_id
                ]
                for child in node["children"]:
                    delete_recursive(child)

        updated_graph = json.loads(json.dumps(graph))  # 深拷贝
        delete_recursive(updated_graph)

        logger.info(f"删除节点: {node_id}")
        return updated_graph

    def search_nodes(self, graph: Dict, query: str) -> List[Dict]:
        """搜索知识图谱节点"""
        results = []
        query_lower = query.lower()

        def search_recursive(node):
            # 检查标题和内容
            if (query_lower in node.get("title", "").lower() or
                    query_lower in node.get("content", "").lower()):
                results.append({
                    "node": node,
                    "relevance": self._calculate_relevance(node, query)
                })

            # 检查关键词
            if "metadata" in node and "keywords" in node["metadata"]:
                keywords = node["metadata"]["keywords"]
                if any(query_lower in keyword.lower() for keyword in keywords):
                    results.append({
                        "node": node,
                        "relevance": self._calculate_relevance(node, query)
                    })

            if "children" in node:
                for child in node["children"]:
                    search_recursive(child)

        search_recursive(graph)

        # 按相关性排序
        results.sort(key=lambda x: x["relevance"], reverse=True)

        return [result["node"] for result in results]

    def get_graph_stats(self, graph: Dict) -> Dict:
        """获取知识图谱统计信息"""
        stats = {
            "total_nodes": 0,
            "learning_nodes": 0,
            "questioning_nodes": 0,
            "manual_nodes": 0,
            "max_depth": 0,
            "created_at": graph.get("created_at"),
            "updated_at": graph.get("updated_at")
        }

        def count_recursive(node, depth=0):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            node_type = node.get("type", "manual")
            if node_type == "learning":
                stats["learning_nodes"] += 1
            elif node_type == "questioning":
                stats["questioning_nodes"] += 1
            else:
                stats["manual_nodes"] += 1

            if "children" in node:
                for child in node["children"]:
                    count_recursive(child, depth + 1)

        count_recursive(graph)
        return stats

    def _extract_title(self, text: str, max_length: int = 30) -> str:
        """从文本中提取标题"""
        # 简单的标题提取逻辑
        title = text.strip()
        if len(title) > max_length:
            title = title[:max_length] + "..."
        return title

    def _create_learning_content(self, question: str, learning_response: str,
                                 questioning_response: Optional[str]) -> str:
        """创建学习节点内容"""
        content = f"问题: {question}\n\n"
        content += f"学习指导:\n{learning_response[:300]}...\n\n"

        if questioning_response:
            content += f"批判思考:\n{questioning_response[:200]}...\n\n"

        content += f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return content

    def _create_questioning_content(self, question: str, questioning_response: str) -> str:
        """创建质疑节点内容"""
        content = f"原问题: {question}\n\n"
        content += f"批判性思考:\n{questioning_response[:400]}...\n\n"
        content += f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return content

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        import re

        # 移除标点符号，分割单词
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', text.lower())

        # 过滤短词和停用词
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要',
                     '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

        filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]

        # 简单的词频统计
        word_count = {}
        for word in filtered_words:
            word_count[word] = word_count.get(word, 0) + 1

        # 按频率排序，返回前N个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]

    def _find_best_parent(self, graph: Dict, new_node: Dict) -> Dict:
        """为新节点找到最佳父节点"""
        # 简单的匹配逻辑：根据关键词相似度
        new_keywords = set(new_node.get("metadata", {}).get("keywords", []))

        best_parent = graph  # 默认父节点是根节点
        best_score = 0

        def find_recursive(node):
            nonlocal best_parent, best_score

            if node.get("type") == "system" and node.get("id") in ["learning_notes"]:
                # 学习节点倾向于放在学习笔记下
                if new_node.get("type") == "learning":
                    return node

            # 计算关键词相似度
            node_keywords = set(node.get("metadata", {}).get("keywords", []))
            common_keywords = new_keywords.intersection(node_keywords)
            score = len(common_keywords)

            if score > best_score:
                best_score = score
                best_parent = node

            if "children" in node:
                for child in node["children"]:
                    result = find_recursive(child)
                    if result:
                        return result

            return None

        result = find_recursive(graph)
        return result if result else best_parent

    def _find_node_by_id(self, graph: Dict, node_id: str) -> Optional[Dict]:
        """根据ID查找节点"""
        if graph.get("id") == node_id:
            return graph

        if "children" in graph:
            for child in graph["children"]:
                result = self._find_node_by_id(child, node_id)
                if result:
                    return result

        return None

    def _find_node_by_title(self, graph: Dict, title: str) -> Optional[Dict]:
        """根据标题查找节点"""
        if graph.get("title") == title:
            return graph

        if "children" in graph:
            for child in graph["children"]:
                result = self._find_node_by_title(child, title)
                if result:
                    return result

        return None

    def _insert_node(self, graph: Dict, parent_id: str, new_node: Dict) -> Dict:
        """在指定父节点下插入新节点"""
        updated_graph = json.loads(json.dumps(graph))  # 深拷贝

        def insert_recursive(node):
            if node["id"] == parent_id:
                if "children" not in node:
                    node["children"] = []
                node["children"].append(new_node)
                node["updated_at"] = datetime.now().isoformat()
                return True

            if "children" in node:
                for child in node["children"]:
                    if insert_recursive(child):
                        return True
            return False

        insert_recursive(updated_graph)
        return updated_graph

    def _calculate_relevance(self, node: Dict, query: str) -> float:
        """计算节点与查询的相关性"""
        relevance = 0.0
        query_lower = query.lower()

        # 标题匹配
        title = node.get("title", "").lower()
        if query_lower in title:
            relevance += 0.5

        # 内容匹配
        content = node.get("content", "").lower()
        if query_lower in content:
            relevance += 0.3

        # 关键词匹配
        keywords = node.get("metadata", {}).get("keywords", [])
        for keyword in keywords:
            if query_lower in keyword.lower():
                relevance += 0.2

        return min(relevance, 1.0)