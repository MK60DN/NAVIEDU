# ====== backend/scripts/init_neo4j.py ======
# Neo4j知识图谱初始化脚本

import os
import sys
from neo4j import GraphDatabase
from typing import Dict, List, Any
import json
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jInitializer:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"连接到Neo4j: {uri}")

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库（慎用！）"""
        with self.driver.session() as session:
            logger.warning("正在清空数据库...")
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("数据库已清空")

    def create_constraints_and_indexes(self):
        """创建约束和索引"""
        with self.driver.session() as session:
            constraints = [
                # 知识点名称唯一
                "CREATE CONSTRAINT knowledge_name_unique IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.name IS UNIQUE",

                # 用户ID唯一
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",

                # 类别名称唯一
                "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE"
            ]

            indexes = [
                # 知识点搜索索引
                "CREATE INDEX knowledge_name_index IF NOT EXISTS FOR (k:Knowledge) ON (k.name)",
                "CREATE INDEX knowledge_description_index IF NOT EXISTS FOR (k:Knowledge) ON (k.description)",
                "CREATE INDEX knowledge_category_index IF NOT EXISTS FOR (k:Knowledge) ON (k.category)",
                "CREATE INDEX knowledge_difficulty_index IF NOT EXISTS FOR (k:Knowledge) ON (k.difficulty)",

                # 全文搜索索引
                "CREATE FULLTEXT INDEX knowledge_search IF NOT EXISTS FOR (k:Knowledge) ON EACH [k.name, k.description, k.tags]"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"约束创建成功: {constraint.split()[2]}")
                except Exception as e:
                    logger.warning(f"约束创建失败或已存在: {e}")

            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"索引创建成功: {index.split()[2]}")
                except Exception as e:
                    logger.warning(f"索引创建失败或已存在: {e}")

    def create_initial_knowledge_graph(self):
        """创建初始知识图谱"""
        # Python编程知识图谱
        python_knowledge = self.get_python_knowledge_data()

        # 编程基础知识图谱
        programming_basics = self.get_programming_basics_data()

        # Web开发知识图谱
        web_development = self.get_web_development_data()

        # 数据科学知识图谱
        data_science = self.get_data_science_data()

        # 创建所有知识点
        all_knowledge = python_knowledge + programming_basics + web_development + data_science

        with self.driver.session() as session:
            # 批量创建知识点
            for knowledge in all_knowledge:
                self.create_knowledge_node(session, knowledge)

            # 创建关系
            self.create_knowledge_relationships(session, all_knowledge)

    def get_python_knowledge_data(self) -> List[Dict]:
        """Python编程知识数据"""
        return [
            {
                "name": "Python基础语法",
                "description": "Python编程语言的基本语法规则，包括变量、数据类型、运算符等",
                "category": "编程语言",
                "difficulty": "入门",
                "estimated_time": "2小时",
                "tags": ["Python", "语法", "基础"],
                "prerequisites": [],
                "content_type": "theory",
                "learning_objectives": ["掌握Python基本语法", "理解变量和数据类型", "学会使用运算符"]
            },
            {
                "name": "Python数据类型",
                "description": "Python中的基本数据类型：整数、浮点数、字符串、布尔值等",
                "category": "编程语言",
                "difficulty": "入门",
                "estimated_time": "1.5小时",
                "tags": ["Python", "数据类型", "基础"],
                "prerequisites": ["Python基础语法"],
                "content_type": "theory",
                "learning_objectives": ["理解各种数据类型", "掌握类型转换", "学会数据类型判断"]
            },
            {
                "name": "Python控制结构",
                "description": "条件语句(if/elif/else)和循环语句(for/while)的使用",
                "category": "编程语言",
                "difficulty": "初级",
                "estimated_time": "2.5小时",
                "tags": ["Python", "控制结构", "条件", "循环"],
                "prerequisites": ["Python数据类型"],
                "content_type": "practice",
                "learning_objectives": ["掌握条件判断", "学会循环控制", "理解程序流程"]
            },
            {
                "name": "Python函数",
                "description": "函数的定义、调用、参数传递、返回值以及作用域概念",
                "category": "编程语言",
                "difficulty": "初级",
                "estimated_time": "3小时",
                "tags": ["Python", "函数", "参数", "作用域"],
                "prerequisites": ["Python控制结构"],
                "content_type": "practice",
                "learning_objectives": ["掌握函数定义和调用", "理解参数传递机制", "学会使用作用域"]
            },
            {
                "name": "Python数据结构",
                "description": "列表、元组、字典、集合等内置数据结构的使用",
                "category": "编程语言",
                "difficulty": "初级",
                "estimated_time": "4小时",
                "tags": ["Python", "数据结构", "列表", "字典"],
                "prerequisites": ["Python函数"],
                "content_type": "practice",
                "learning_objectives": ["掌握各种数据结构", "学会数据操作方法", "理解数据结构选择"]
            },
            {
                "name": "Python面向对象编程",
                "description": "类和对象、继承、封装、多态等面向对象编程概念",
                "category": "编程语言",
                "difficulty": "中级",
                "estimated_time": "5小时",
                "tags": ["Python", "面向对象", "类", "对象", "继承"],
                "prerequisites": ["Python数据结构"],
                "content_type": "theory_practice",
                "learning_objectives": ["理解OOP概念", "掌握类的定义和使用", "学会继承和多态"]
            },
            {
                "name": "Python异常处理",
                "description": "try-except语句、异常类型、自定义异常等异常处理机制",
                "category": "编程语言",
                "difficulty": "中级",
                "estimated_time": "2小时",
                "tags": ["Python", "异常处理", "调试"],
                "prerequisites": ["Python面向对象编程"],
                "content_type": "practice",
                "learning_objectives": ["掌握异常处理语法", "理解常见异常类型", "学会程序调试"]
            },
            {
                "name": "Python文件操作",
                "description": "文件读写、路径操作、CSV/JSON文件处理等",
                "category": "编程语言",
                "difficulty": "中级",
                "estimated_time": "2.5小时",
                "tags": ["Python", "文件操作", "IO"],
                "prerequisites": ["Python异常处理"],
                "content_type": "practice",
                "learning_objectives": ["掌握文件读写操作", "学会处理不同格式文件", "理解路径操作"]
            },
            {
                "name": "Python模块和包",
                "description": "模块导入、包的创建和使用、Python标准库介绍",
                "category": "编程语言",
                "difficulty": "中级",
                "estimated_time": "3小时",
                "tags": ["Python", "模块", "包", "标准库"],
                "prerequisites": ["Python文件操作"],
                "content_type": "theory_practice",
                "learning_objectives": ["理解模块和包的概念", "掌握导入机制", "学会使用标准库"]
            },
            {
                "name": "Python装饰器",
                "description": "装饰器的概念、实现原理、常用装饰器的使用",
                "category": "编程语言",
                "difficulty": "高级",
                "estimated_time": "3小时",
                "tags": ["Python", "装饰器", "高级特性"],
                "prerequisites": ["Python模块和包"],
                "content_type": "theory_practice",
                "learning_objectives": ["理解装饰器原理", "掌握装饰器语法", "学会创建自定义装饰器"]
            }
        ]

    def get_programming_basics_data(self) -> List[Dict]:
        """编程基础知识数据"""
        return [
            {
                "name": "编程思维",
                "description": "计算思维、问题分解、算法设计等编程基本思维方式",
                "category": "编程基础",
                "difficulty": "入门",
                "estimated_time": "1小时",
                "tags": ["编程思维", "算法", "问题解决"],
                "prerequisites": [],
                "content_type": "theory",
                "learning_objectives": ["培养编程思维", "学会问题分解", "理解算法概念"]
            },
            {
                "name": "数据结构基础",
                "description": "数组、链表、栈、队列等基本数据结构的概念和应用",
                "category": "数据结构与算法",
                "difficulty": "初级",
                "estimated_time": "4小时",
                "tags": ["数据结构", "数组", "链表", "栈", "队列"],
                "prerequisites": ["编程思维"],
                "content_type": "theory_practice",
                "learning_objectives": ["理解基本数据结构", "掌握数据结构选择", "学会数据结构操作"]
            },
            {
                "name": "算法基础",
                "description": "排序算法、搜索算法、时间复杂度和空间复杂度分析",
                "category": "数据结构与算法",
                "difficulty": "中级",
                "estimated_time": "6小时",
                "tags": ["算法", "排序", "搜索", "复杂度"],
                "prerequisites": ["数据结构基础"],
                "content_type": "theory_practice",
                "learning_objectives": ["掌握常用算法", "理解算法复杂度", "学会算法分析"]
            },
            {
                "name": "版本控制Git",
                "description": "Git版本控制系统的使用，包括基本命令、分支管理等",
                "category": "开发工具",
                "difficulty": "初级",
                "estimated_time": "3小时",
                "tags": ["Git", "版本控制", "协作"],
                "prerequisites": ["编程思维"],
                "content_type": "practice",
                "learning_objectives": ["掌握Git基本操作", "理解版本控制概念", "学会团队协作"]
            },
            {
                "name": "调试技能",
                "description": "程序调试方法、调试工具使用、错误定位和修复技巧",
                "category": "编程基础",
                "difficulty": "中级",
                "estimated_time": "2小时",
                "tags": ["调试", "错误处理", "开发技能"],
                "prerequisites": ["版本控制Git"],
                "content_type": "practice",
                "learning_objectives": ["掌握调试方法", "学会错误定位", "提高问题解决能力"]
            }
        ]

    def get_web_development_data(self) -> List[Dict]:
        """Web开发知识数据"""
        return [
            {
                "name": "HTML基础",
                "description": "HTML标签、文档结构、语义化标记等网页结构基础",
                "category": "前端开发",
                "difficulty": "入门",
                "estimated_time": "2小时",
                "tags": ["HTML", "网页结构", "前端"],
                "prerequisites": [],
                "content_type": "practice",
                "learning_objectives": ["掌握HTML基本标签", "理解文档结构", "学会语义化标记"]
            },
            {
                "name": "CSS基础",
                "description": "CSS选择器、样式属性、盒模型、布局等网页样式设计",
                "category": "前端开发",
                "difficulty": "初级",
                "estimated_time": "3小时",
                "tags": ["CSS", "样式设计", "布局"],
                "prerequisites": ["HTML基础"],
                "content_type": "practice",
                "learning_objectives": ["掌握CSS基本语法", "理解盒模型", "学会页面布局"]
            },
            {
                "name": "JavaScript基础",
                "description": "JavaScript语法、DOM操作、事件处理等前端交互编程",
                "category": "前端开发",
                "difficulty": "初级",
                "estimated_time": "4小时",
                "tags": ["JavaScript", "DOM", "事件", "交互"],
                "prerequisites": ["CSS基础"],
                "content_type": "practice",
                "learning_objectives": ["掌握JavaScript语法", "学会DOM操作", "理解事件处理"]
            },
            {
                "name": "响应式设计",
                "description": "移动端适配、媒体查询、弹性布局等响应式网页设计",
                "category": "前端开发",
                "difficulty": "中级",
                "estimated_time": "3小时",
                "tags": ["响应式", "移动端", "媒体查询"],
                "prerequisites": ["JavaScript基础"],
                "content_type": "practice",
                "learning_objectives": ["掌握响应式设计原理", "学会移动端适配", "理解弹性布局"]
            },
            {
                "name": "前端框架Vue.js",
                "description": "Vue.js框架基础、组件开发、数据绑定等现代前端开发",
                "category": "前端开发",
                "difficulty": "中级",
                "estimated_time": "5小时",
                "tags": ["Vue.js", "框架", "组件", "MVVM"],
                "prerequisites": ["响应式设计"],
                "content_type": "practice",
                "learning_objectives": ["掌握Vue.js基础", "学会组件开发", "理解MVVM模式"]
            },
            {
                "name": "RESTful API",
                "description": "REST架构风格、HTTP方法、API设计原则等后端接口设计",
                "category": "后端开发",
                "difficulty": "中级",
                "estimated_time": "2小时",
                "tags": ["REST", "API", "HTTP"],
                "prerequisites": ["前端框架Vue.js"],
                "content_type": "theory",
                "learning_objectives": ["理解REST架构", "掌握API设计", "学会HTTP协议"]
            },
            {
                "name": "数据库基础",
                "description": "关系型数据库、SQL语言、数据库设计等数据存储基础",
                "category": "后端开发",
                "difficulty": "初级",
                "estimated_time": "4小时",
                "tags": ["数据库", "SQL", "关系型"],
                "prerequisites": ["RESTful API"],
                "content_type": "practice",
                "learning_objectives": ["掌握SQL语言", "理解数据库设计", "学会数据操作"]
            }
        ]

    def get_data_science_data(self) -> List[Dict]:
        """数据科学知识数据"""
        return [
            {
                "name": "数据科学概述",
                "description": "数据科学的定义、应用领域、基本流程等入门知识",
                "category": "数据科学",
                "difficulty": "入门",
                "estimated_time": "1小时",
                "tags": ["数据科学", "概述", "应用"],
                "prerequisites": ["Python基础语法"],
                "content_type": "theory",
                "learning_objectives": ["了解数据科学领域", "理解数据科学流程", "认识应用场景"]
            },
            {
                "name": "NumPy数值计算",
                "description": "NumPy库的使用，数组操作、数学运算、线性代数等",
                "category": "数据科学",
                "difficulty": "初级",
                "estimated_time": "3小时",
                "tags": ["NumPy", "数值计算", "数组"],
                "prerequisites": ["数据科学概述"],
                "content_type": "practice",
                "learning_objectives": ["掌握NumPy基本操作", "学会数组计算", "理解向量化运算"]
            },
            {
                "name": "Pandas数据处理",
                "description": "Pandas库的使用，数据清洗、转换、分析等数据处理技能",
                "category": "数据科学",
                "difficulty": "中级",
                "estimated_time": "4小时",
                "tags": ["Pandas", "数据处理", "数据分析"],
                "prerequisites": ["NumPy数值计算"],
                "content_type": "practice",
                "learning_objectives": ["掌握Pandas数据操作", "学会数据清洗", "理解数据分析方法"]
            },
            {
                "name": "数据可视化",
                "description": "使用Matplotlib、Seaborn等库进行数据可视化",
                "category": "数据科学",
                "difficulty": "中级",
                "estimated_time": "3小时",
                "tags": ["数据可视化", "Matplotlib", "图表"],
                "prerequisites": ["Pandas数据处理"],
                "content_type": "practice",
                "learning_objectives": ["掌握数据可视化技能", "学会制作各种图表", "理解可视化原则"]
            }
        ]

    def create_knowledge_node(self, session, knowledge: Dict):
        """创建知识点节点"""
        cypher = """
        CREATE (k:Knowledge {
            name: $name,
            description: $description,
            category: $category,
            difficulty: $difficulty,
            estimated_time: $estimated_time,
            tags: $tags,
            content_type: $content_type,
            learning_objectives: $learning_objectives,
            created_at: datetime(),
            status: 'active'
        })
        """

        try:
            session.run(cypher, **knowledge)
            logger.info(f"知识点创建成功: {knowledge['name']}")
        except Exception as e:
            logger.error(f"知识点创建失败 {knowledge['name']}: {e}")

    def create_knowledge_relationships(self, session, all_knowledge: List[Dict]):
        """创建知识点之间的关系"""
        # 创建先决条件关系
        for knowledge in all_knowledge:
            for prerequisite in knowledge.get('prerequisites', []):
                cypher = """
                MATCH (prerequisite:Knowledge {name: $prerequisite_name})
                MATCH (knowledge:Knowledge {name: $knowledge_name})
                CREATE (prerequisite)-[:PREREQUISITE]->(knowledge)
                """

                try:
                    session.run(cypher,
                                prerequisite_name=prerequisite,
                                knowledge_name=knowledge['name'])
                    logger.info(f"关系创建成功: {prerequisite} -> {knowledge['name']}")
                except Exception as e:
                    logger.error(f"关系创建失败 {prerequisite} -> {knowledge['name']}: {e}")

        # 创建类别关系
        categories = set(k['category'] for k in all_knowledge)
        for category in categories:
            # 创建类别节点
            try:
                session.run("CREATE (c:Category {name: $name})", name=category)
                logger.info(f"类别创建成功: {category}")
            except Exception as e:
                logger.warning(f"类别创建失败或已存在 {category}: {e}")

            # 连接知识点到类别
            cypher = """
            MATCH (c:Category {name: $category})
            MATCH (k:Knowledge {category: $category})
            CREATE (k)-[:BELONGS_TO]->(c)
            """

            try:
                session.run(cypher, category=category)
                logger.info(f"类别关系创建成功: -> {category}")
            except Exception as e:
                logger.error(f"类别关系创建失败 -> {category}: {e}")

    def create_sample_user_progress(self):
        """创建示例用户学习进度"""
        with self.driver.session() as session:
            # 创建示例用户
            sample_users = [
                {"user_id": 1, "username": "学习者小明", "level": "初级"},
                {"user_id": 2, "username": "程序员小红", "level": "中级"},
            ]

            for user in sample_users:
                cypher = """
                CREATE (u:User {
                    user_id: $user_id,
                    username: $username,
                    level: $level,
                    created_at: datetime(),
                    total_learning_time: 0,
                    completed_topics: 0
                })
                """

                try:
                    session.run(cypher, **user)
                    logger.info(f"示例用户创建成功: {user['username']}")
                except Exception as e:
                    logger.error(f"示例用户创建失败: {e}")

            # 创建学习记录
            learning_records = [
                {"user_id": 1, "knowledge": "Python基础语法", "status": "completed"},
                {"user_id": 1, "knowledge": "Python数据类型", "status": "completed"},
                {"user_id": 1, "knowledge": "Python控制结构", "status": "in_progress"},
                {"user_id": 2, "knowledge": "Python函数", "status": "completed"},
                {"user_id": 2, "knowledge": "Python数据结构", "status": "in_progress"},
            ]

            for record in learning_records:
                cypher = """
                MATCH (u:User {user_id: $user_id})
                MATCH (k:Knowledge {name: $knowledge})
                CREATE (u)-[:LEARNED {
                    status: $status,
                    started_at: datetime(),
                    progress: CASE $status 
                        WHEN 'completed' THEN 100 
                        WHEN 'in_progress' THEN 50 
                        ELSE 0 
                    END
                }]->(k)
                """

                try:
                    session.run(cypher, **record)
                    logger.info(f"学习记录创建成功: {record['user_id']} -> {record['knowledge']}")
                except Exception as e:
                    logger.error(f"学习记录创建失败: {e}")

    def verify_graph_creation(self):
        """验证图谱创建结果"""
        with self.driver.session() as session:
            # 统计节点数量
            stats_queries = [
                ("知识点数量", "MATCH (k:Knowledge) RETURN count(k) as count"),
                ("类别数量", "MATCH (c:Category) RETURN count(c) as count"),
                ("用户数量", "MATCH (u:User) RETURN count(u) as count"),
                ("前置关系数量", "MATCH ()-[r:PREREQUISITE]->() RETURN count(r) as count"),
                ("学习记录数量", "MATCH ()-[r:LEARNED]->() RETURN count(r) as count")
            ]

            logger.info("=== 图谱创建结果统计 ===")
            for name, query in stats_queries:
                try:
                    result = session.run(query)
                    count = result.single()["count"]
                    logger.info(f"{name}: {count}")
                except Exception as e:
                    logger.error(f"{name}统计失败: {e}")

            # 检查图谱连通性
            logger.info("=== 图谱连通性检查 ===")
            connectivity_query = """
            MATCH path = (start:Knowledge)-[:PREREQUISITE*]->(end:Knowledge)
            WHERE NOT (start)<-[:PREREQUISITE]-()
            RETURN start.name as start_point, 
                   length(path) as path_length,
                   end.name as end_point
            ORDER BY path_length DESC
            LIMIT 5
            """

            try:
                result = session.run(connectivity_query)
                logger.info("最长学习路径:")
                for record in result:
                    logger.info(f"  {record['start_point']} -> {record['end_point']} (步骤: {record['path_length']})")
            except Exception as e:
                logger.error(f"连通性检查失败: {e}")


def main():
    """主函数"""
    try:
        # 初始化数据库连接
        neo4j_init = Neo4jInitializer(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD
        )

        logger.info("开始初始化Neo4j知识图谱...")

        # 询问是否清空数据库
        if len(sys.argv) > 1 and sys.argv[1] == '--clear':
            confirm = input("⚠️  确定要清空数据库吗？这将删除所有数据！(yes/no): ")
            if confirm.lower() == 'yes':
                neo4j_init.clear_database()
            else:
                logger.info("取消清空数据库")

        # 创建约束和索引
        neo4j_init.create_constraints_and_indexes()

        # 创建知识图谱
        neo4j_init.create_initial_knowledge_graph()

        # 创建示例用户数据
        neo4j_init.create_sample_user_progress()

        # 验证创建结果
        neo4j_init.verify_graph_creation()

        logger.info("✅ Neo4j知识图谱初始化完成！")

    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return 1
    finally:
        neo4j_init.close()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

# ====== backend/scripts/test_knowledge_graph.py ======
# 知识图谱测试脚本

import os
import sys
import asyncio
from neo4j import GraphDatabase

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ai_tutor_service import DeepSeekAITutorService
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_knowledge_search():
    """测试知识点搜索"""
    service = DeepSeekAITutorService(
        neo4j_uri=settings.NEO4J_URI,
        neo4j_user=settings.NEO4J_USER,
        neo4j_password=settings.NEO4J_PASSWORD,
        deepseek_key=settings.DEEPSEEK_API_KEY
    )

    test_queries = [
        "Python",
        "函数",
        "数据结构",
        "网页开发",
        "数据科学"
    ]

    logger.info("=== 测试知识点搜索 ===")
    for query in test_queries:
        try:
            results = service._search_neo4j_knowledge([query])
            logger.info(f"搜索 '{query}': 找到 {len(results)} 个结果")
            for result in results[:3]:  # 只显示前3个
                logger.info(f"  - {result['name']} ({result['difficulty']})")
        except Exception as e:
            logger.error(f"搜索 '{query}' 失败: {e}")

    service.close()


def test_learning_paths():
    """测试学习路径计算"""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    test_paths = [
        ("Python基础语法", "Python装饰器"),
        ("HTML基础", "前端框架Vue.js"),
        ("编程思维", "算法基础"),
        ("数据科学概述", "数据可视化")
    ]

    logger.info("=== 测试学习路径计算 ===")
    with driver.session() as session:
        for start, end in test_paths:
            try:
                cypher = """
                MATCH (start:Knowledge {name: $start})
                MATCH (end:Knowledge {name: $end})
                MATCH path = shortestPath((start)-[:PREREQUISITE*]->(end))
                RETURN nodes(path) as learning_path, length(path) as path_length
                """

                result = session.run(cypher, start=start, end=end)
                record = result.single()

                if record:
                    path_length = record["path_length"]
                    path_names = [node["name"] for node in record["learning_path"]]
                    logger.info(f"路径 '{start}' -> '{end}': {path_length} 步")
                    logger.info(f"  路径: {' -> '.join(path_names)}")
                else:
                    logger.warning(f"未找到路径 '{start}' -> '{end}'")

            except Exception as e:
                logger.error(f"路径计算失败 '{start}' -> '{end}': {e}")

    driver.close()


def test_graph_queries():
    """测试各种图数据库查询"""
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )

    queries = [
        # 查找入门级知识点
        ("入门级知识点", """
            MATCH (k:Knowledge {difficulty: '入门'})
            RETURN k.name, k.category
            ORDER BY k.category, k.name
        """),

        # 查找没有前置条件的知识点（起始点）
        ("起始知识点", """
            MATCH (k:Knowledge)
            WHERE NOT (k)<-[:PREREQUISITE]-()
            RETURN k.name, k.category, k.difficulty
            ORDER BY k.category
        """),

        # 查找最复杂的知识点（有很多前置条件）
        ("复杂知识点", """
            MATCH (k:Knowledge)<-[:PREREQUISITE*]-(prereq)
            WITH k, count(DISTINCT prereq) as prereq_count
            WHERE prereq_count > 0
            RETURN k.name, k.category, prereq_count
            ORDER BY prereq_count DESC
            LIMIT 5
        """),

        # 按类别统计知识点
        ("类别统计", """
            MATCH (k:Knowledge)
            WITH k.category as category, count(k) as count
            RETURN category, count
            ORDER BY count DESC
        """),

        # 查找孤立的知识点（没有任何关系）
        ("孤立知识点", """
            MATCH (k:Knowledge)
            WHERE NOT (k)-[:PREREQUISITE]-() AND NOT (k)<-[:PREREQUISITE]-()
            RETURN k.name, k.category
        """)
    ]

    logger.info("=== 测试图数据库查询 ===")
    with driver.session() as session:
        for name, query in queries:
            try:
                result = session.run(query)
                records = list(result)
                logger.info(f"{name}: {len(records)} 条记录")

                for record in records[:5]:  # 只显示前5条
                    values = [str(v) for v in record.values()]
                    logger.info(f"  {' | '.join(values)}")

                if len(records) > 5:
                    logger.info(f"  ... 还有 {len(records) - 5} 条记录")

            except Exception as e:
                logger.error(f"{name} 查询失败: {e}")

    driver.close()


async def main():
    """主测试函数"""
    logger.info("开始测试Neo4j知识图谱...")

    try:
        # 测试知识点搜索
        await test_knowledge_search()

        # 测试学习路径计算
        test_learning_paths()

        # 测试各种查询
        test_graph_queries()

        logger.info("✅ 所有测试完成！")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

# ====== backend/scripts/backup_restore.py ======
# 数据库备份和恢复脚本

import os
import sys
import json
import gzip
from datetime import datetime
from neo4j import GraphDatabase
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jBackupRestore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)

    def close(self):
        self.driver.close()

    def backup_graph(self, filename: str = None):
        """备份整个知识图谱"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_graph_backup_{timestamp}.json.gz"

        filepath = os.path.join(self.backup_dir, filename)

        logger.info(f"开始备份知识图谱到: {filepath}")

        with self.driver.session() as session:
            # 备份所有节点
            nodes_query = """
            MATCH (n)
            RETURN labels(n) as labels, properties(n) as properties
            """

            # 备份所有关系
            relationships_query = """
            MATCH (a)-[r]->(b)
            RETURN labels(a) as start_labels, properties(a) as start_props,
                   type(r) as rel_type, properties(r) as rel_props,
                   labels(b) as end_labels, properties(b) as end_props
            """

            backup_data = {
                "backup_time": datetime.now().isoformat(),
                "nodes": [],
                "relationships": []
            }

            # 获取节点数据
            logger.info("备份节点数据...")
            result = session.run(nodes_query)
            for record in result:
                backup_data["nodes"].append({
                    "labels": record["labels"],
                    "properties": record["properties"]
                })

            logger.info(f"备份了 {len(backup_data['nodes'])} 个节点")

            # 获取关系数据
            logger.info("备份关系数据...")
            result = session.run(relationships_query)
            for record in result:
                backup_data["relationships"].append({
                    "start_labels": record["start_labels"],
                    "start_props": record["start_props"],
                    "rel_type": record["rel_type"],
                    "rel_props": record["rel_props"],
                    "end_labels": record["end_labels"],
                    "end_props": record["end_props"]
                })

            logger.info(f"备份了 {len(backup_data['relationships'])} 个关系")

        # 压缩保存
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"✅ 备份完成: {filepath}")
        return filepath

    def restore_graph(self, filepath: str, clear_existing: bool = False):
        """从备份恢复知识图谱"""
        if not os.path.exists(filepath):
            logger.error(f"备份文件不存在: {filepath}")
            return False

        logger.info(f"开始从备份恢复: {filepath}")

        # 读取备份数据
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
        except Exception as e:
            logger.error(f"读取备份文件失败: {e}")
            return False

        with self.driver.session() as session:
            # 清空现有数据（如果需要）
            if clear_existing:
                logger.warning("清空现有数据...")
                session.run("MATCH (n) DETACH DELETE n")

            # 恢复节点
            logger.info(f"恢复 {len(backup_data['nodes'])} 个节点...")
            for node_data in backup_data['nodes']:
                labels = ":".join(node_data['labels'])
                properties = node_data['properties']

                # 构造CREATE语句
                props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
                cypher = f"CREATE (n:{labels} {{{props_str}}})"

                try:
                    session.run(cypher, **properties)
                except Exception as e:
                    logger.warning(f"节点恢复失败: {e}")

            # 恢复关系
            logger.info(f"恢复 {len(backup_data['relationships'])} 个关系...")
            for rel_data in backup_data['relationships']:
                start_labels = ":".join(rel_data['start_labels'])
                end_labels = ":".join(rel_data['end_labels'])
                rel_type = rel_data['rel_type']

                start_props = rel_data['start_props']
                end_props = rel_data['end_props']
                rel_props = rel_data.get('rel_props', {})

                # 构造MATCH和CREATE语句
                start_match = " AND ".join([f"start.{k} = ${k}" for k in start_props.keys()])
                end_match = " AND ".join([f"end.{k} = ${k}" for k in end_props.keys()])

                if rel_props:
                    rel_props_str = "{" + ", ".join([f"{k}: ${k}" for k in rel_props.keys()]) + "}"
                    cypher = f"""
                    MATCH (start:{start_labels}) WHERE {start_match}
                    MATCH (end:{end_labels}) WHERE {end_match}
                    CREATE (start)-[r:{rel_type} {rel_props_str}]->(end)
                    """
                else:
                    cypher = f"""
                    MATCH (start:{start_labels}) WHERE {start_match}
                    MATCH (end:{end_labels}) WHERE {end_match}
                    CREATE (start)-[r:{rel_type}]->(end)
                    """

                try:
                    params = {**start_props, **end_props, **rel_props}
                    session.run(cypher, **params)
                except Exception as e:
                    logger.warning(f"关系恢复失败: {e}")

        logger.info("✅ 恢复完成")
        return True

    def list_backups(self):
        """列出所有备份文件"""
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.startswith("knowledge_graph_backup_"):
                filepath = os.path.join(self.backup_dir, filename)
                stat = os.stat(filepath)
                backups.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime)
                })

        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups


def main():
    """主函数"""
    backup_restore = Neo4jBackupRestore()

    try:
        if len(sys.argv) < 2:
            print("用法:")
            print("  python backup_restore.py backup [filename]")
            print("  python backup_restore.py restore <filename> [--clear]")
            print("  python backup_restore.py list")
            return 1

        command = sys.argv[1]

        if command == "backup":
            filename = sys.argv[2] if len(sys.argv) > 2 else None
            filepath = backup_restore.backup_graph(filename)
            print(f"备份已保存到: {filepath}")

        elif command == "restore":
            if len(sys.argv) < 3:
                print("错误: 请指定备份文件")
                return 1

            filepath = sys.argv[2]
            clear_existing = "--clear" in sys.argv

            if clear_existing:
                confirm = input("⚠️  确定要清空现有数据并恢复备份吗？(yes/no): ")
                if confirm.lower() != 'yes':
                    print("取消恢复")
                    return 0

            success = backup_restore.restore_graph(filepath, clear_existing)
            if success:
                print("恢复成功")
            else:
                print("恢复失败")
                return 1

        elif command == "list":
            backups = backup_restore.list_backups()
            if backups:
                print("可用的备份文件:")
                for backup in backups:
                    size_mb = backup["size"] / (1024 * 1024)
                    print(f"  {backup['filename']}")
                    print(f"    大小: {size_mb:.2f} MB")
                    print(f"    创建时间: {backup['created']}")
                    print()
            else:
                print("没有找到备份文件")

        else:
            print(f"未知命令: {command}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    finally:
        backup_restore.close()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)