#!/usr/bin/env python
"""
数据库初始化脚本
运行: python init_db.py
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import User, Token, Capsule
from app.services.auth_service import get_password_hash
from app.config import settings
import uuid


def init_database():
    """初始化数据库并创建默认数据"""

    # 创建所有表
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)

    # 创建会话
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # 检查是否已初始化
        if db.query(User).filter(User.username == "admin").first():
            print("数据库已初始化")
            return

        # 创建管理员用户
        admin = User(
            id=str(uuid.uuid4()),
            username="admin",
            email="admin@edupath.com",
            hashed_password=get_password_hash("admin123"),
            is_admin=True,
            e_coin_balance=1000.0,
            python_token_balance=1000.0
        )
        db.add(admin)

        # 创建Python代币
        python_token = Token(
            id=str(uuid.uuid4()),
            name="Python Token",
            symbol="PYTHON",
            price=1.0,
            total_supply=1000000.0
        )
        db.add(python_token)

        # 创建示例胶囊
        capsules = [
            {
                "title": "Python变量和数据类型",
                "description": "了解Python中的基本数据类型",
                "content": "Python中有多种数据类型，包括整数(int)、浮点数(float)、字符串(str)、布尔值(bool)等。变量是存储数据的容器，在Python中不需要声明变量类型。",
                "code": """# 整数
age = 25
# 浮点数  
price = 19.99
# 字符串
name = "Alice"
# 布尔值
is_student = True""",
                "category": "基础语法"
            },
            {
                "title": "Python列表",
                "description": "学习Python中的列表数据结构",
                "content": "列表是Python中最常用的数据结构之一，可以存储多个元素，元素可以是不同类型。列表是可变的，可以添加、删除、修改元素。",
                "code": """# 创建列表
fruits = ["apple", "banana", "orange"]
# 访问元素
print(fruits[0])  # apple
# 添加元素
fruits.append("grape")
# 删除元素
fruits.remove("banana")""",
                "category": "数据结构"
            },
            {
                "title": "Python函数",
                "description": "掌握Python函数的定义和使用",
                "content": "函数是组织好的、可重复使用的代码块。函数能提高应用的模块性和代码的重复利用率。",
                "code": """# 定义函数
def greet(name):
    return f"Hello, {name}!"

# 调用函数
message = greet("World")
print(message)  # Hello, World!""",
                "category": "函数编程"
            }
        ]

        for capsule_data in capsules:
            capsule = Capsule(
                id=str(uuid.uuid4()),
                title=capsule_data["title"],
                description=capsule_data["description"],
                content=capsule_data["content"],
                code=capsule_data["code"],
                category=capsule_data["category"],
                author_id=admin.id,
                status="approved"
            )
            db.add(capsule)

        db.commit()
        print("数据库初始化成功！")
        print("管理员账号: admin / admin123")

    except Exception as e:
        print(f"初始化失败: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_database()