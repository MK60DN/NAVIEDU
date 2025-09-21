# backend/init_mvp_db.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app.models import User, KnowledgeCapsule, UserToken, LearningProgress
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def init_database():
    """初始化MVP数据库"""
    print("🔧 正在初始化EduPath MVP数据库...")

    # 创建所有表
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建完成")

    # 创建会话
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # 检查是否已有管理员用户
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            # 创建管理员用户
            hashed_password = pwd_context.hash("admin123")
            admin_user = User(
                username="admin",
                email="admin@edupath.com",
                hashed_password=hashed_password,
                is_superuser=True,
                is_active=True
            )
            db.add(admin_user)
            db.flush()  # 获取ID

            # 给管理员初始代币
            admin_tokens = [
                UserToken(user_id=admin_user.id, token_type="E_COIN", balance=1000),
                UserToken(user_id=admin_user.id, token_type="PYTHON_TOKEN", balance=100)
            ]
            for token in admin_tokens:
                db.add(token)

            print("✅ 管理员用户创建成功: admin / admin123")
        else:
            print("ℹ️  管理员用户已存在")

        # 检查是否已有示例知识胶囊
        capsule_count = db.query(KnowledgeCapsule).count()
        if capsule_count == 0:
            # 创建示例知识胶囊
            sample_capsules = [
                KnowledgeCapsule(
                    name="Python基础",
                    description="Python编程语言基础概念",
                    content="Python是一种高级编程语言，以其简洁明了的语法而著称。它广泛应用于Web开发、数据分析、人工智能等领域。Python的设计哲学强调代码的可读性，使用有意义的缩进来定义代码块。",
                    code_example="# 这是Python注释\nprint('Hello, World!')\n\n# 变量赋值\nname = 'Python'\nprint(f'我正在学习 {name}')\n\n# Python的特点\nprint('Python是:')\nprint('- 易学易用')\nprint('- 功能强大')\nprint('- 社区活跃')",
                    difficulty="初级",
                    category="Python",
                    estimated_time="1小时",
                    prerequisites="",
                    status="published"
                ),

                KnowledgeCapsule(
                    name="变量与数据类型",
                    description="学习Python中的变量和基本数据类型",
                    content="在Python中，变量是存储数据的容器。Python有多种数据类型：整数(int)、浮点数(float)、字符串(str)、布尔值(bool)等。Python是动态类型语言，不需要显式声明变量类型。",
                    code_example="# 不同的数据类型\nname = 'Alice'      # 字符串 (str)\nage = 25           # 整数 (int)\nheight = 1.68      # 浮点数 (float)\nis_student = True  # 布尔值 (bool)\n\n# 打印变量信息\nprint(f'姓名: {name} (类型: {type(name).__name__})')\nprint(f'年龄: {age} (类型: {type(age).__name__})')\nprint(f'身高: {height}m (类型: {type(height).__name__})')\nprint(f'是学生: {is_student} (类型: {type(is_student).__name__})')",
                    difficulty="初级",
                    category="Python",
                    estimated_time="45分钟",
                    prerequisites="Python基础",
                    parent_id=1,
                    status="published"
                ),

                KnowledgeCapsule(
                    name="条件语句",
                    description="学习if-else条件判断",
                    content="条件语句用于根据不同情况执行不同的代码。Python使用if、elif、else关键字来实现条件判断。缩进在Python中非常重要，用于定义代码块的范围。",
                    code_example="# 简单条件判断\nage = 18\n\nif age >= 18:\n    print('你是成年人')\nelse:\n    print('你是未成年人')\n\n# 多重条件\nscore = 85\n\nif score >= 90:\n    grade = '优秀'\nelif score >= 80:\n    grade = '良好'\nelif score >= 70:\n    grade = '中等'\nelif score >= 60:\n    grade = '及格'\nelse:\n    grade = '不及格'\n\nprint(f'你的成绩是: {grade}')\n\n# 逻辑运算符\nusername = 'admin'\npassword = '123456'\n\nif username == 'admin' and password == '123456':\n    print('登录成功')\nelse:\n    print('用户名或密码错误')",
                    difficulty="初级",
                    category="Python",
                    estimated_time="30分钟",
                    prerequisites="变量与数据类型",
                    parent_id=2,
                    status="published"
                ),

                KnowledgeCapsule(
                    name="循环语句",
                    description="学习for和while循环",
                    content="循环语句用于重复执行代码块。Python提供两种主要的循环类型：for循环用于遍历序列，while循环用于条件循环。",
                    code_example="# for循环 - 遍历列表\nfruits = ['苹果', '香蕉', '橙子']\nfor fruit in fruits:\n    print(f'我喜欢吃{fruit}')\n\n# for循环 - 使用range\nprint('倒计时:')\nfor i in range(5, 0, -1):\n    print(i)\nprint('发射!')\n\n# while循环\ncount = 0\nwhile count < 3:\n    print(f'这是第 {count + 1} 次循环')\n    count += 1\n\n# 列表推导式 (高级用法)\nsquares = [x**2 for x in range(1, 6)]\nprint(f'1到5的平方: {squares}')",
                    difficulty="初级",
                    category="Python",
                    estimated_time="40分钟",
                    prerequisites="条件语句",
                    parent_id=3,
                    status="published"
                ),

                KnowledgeCapsule(
                    name="函数定义",
                    description="学习如何定义和使用函数",
                    content="函数是组织好的、可重复使用的代码块。使用def关键字定义函数，可以接收参数并返回值。函数有助于代码复用和模块化设计。",
                    code_example="# 简单函数定义\ndef greet(name):\n    \"\"\"问候函数\"\"\"\n    return f'你好, {name}!'\n\n# 调用函数\nmessage = greet('Alice')\nprint(message)\n\n# 带默认参数的函数\ndef introduce(name, age=18, city='北京'):\n    return f'我叫{name}，今年{age}岁，来自{city}'\n\nprint(introduce('小明'))\nprint(introduce('小红', 20))\nprint(introduce('小李', 22, '上海'))\n\n# 返回多个值\ndef calculate(a, b):\n    \"\"\"计算两个数的和、差、积、商\"\"\"\n    return a + b, a - b, a * b, a / b\n\nsum_val, diff, product, quotient = calculate(10, 3)\nprint(f'和: {sum_val}, 差: {diff}, 积: {product}, 商: {quotient:.2f}')",
                    difficulty="中级",
                    category="Python",
                    estimated_time="50分钟",
                    prerequisites="循环语句",
                    parent_id=4,
                    status="published"
                )
            ]

            for capsule in sample_capsules:
                db.add(capsule)

            print("✅ 示例知识胶囊创建成功")
        else:
            print("ℹ️  知识胶囊已存在")

        # 创建示例普通用户
        test_user = db.query(User).filter(User.username == "testuser").first()
        if not test_user:
            hashed_password = pwd_context.hash("test123")
            test_user = User(
                username="testuser",
                email="test@edupath.com",
                hashed_password=hashed_password,
                is_superuser=False,
                is_active=True
            )
            db.add(test_user)
            db.flush()

            # 给测试用户初始代币
            test_tokens = [
                UserToken(user_id=test_user.id, token_type="E_COIN", balance=50),
                UserToken(user_id=test_user.id, token_type="PYTHON_TOKEN", balance=10)
            ]
            for token in test_tokens:
                db.add(token)

            print("✅ 测试用户创建成功: testuser / test123")
        else:
            print("ℹ️  测试用户已存在")

        # 提交所有更改
        db.commit()
        print("\n🎉 EduPath MVP数据库初始化完成!")
        print("=" * 50)
        print("📊 数据库信息:")
        print(f"👥 用户数量: {db.query(User).count()}")
        print(f"📚 知识胶囊数量: {db.query(KnowledgeCapsule).count()}")
        print(f"💰 代币记录数量: {db.query(UserToken).count()}")
        print("=" * 50)
        print("🔑 登录信息:")
        print("管理员: admin / admin123")
        print("测试用户: testuser / test123")
        print("=" * 50)

    except Exception as e:
        db.rollback()
        print(f"❌ 数据库初始化失败: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_database()