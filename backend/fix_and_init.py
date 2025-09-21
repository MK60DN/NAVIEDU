import os
import sys

print("修复并初始化数据库...")

# 安装必要的包
print("1. 检查依赖...")
try:
    import sqlalchemy
    import passlib
    print("✓ 依赖已安装")
except ImportError:
    print("安装依赖...")
    os.system("pip install sqlalchemy passlib[bcrypt]")

# 创建必要的目录和文件
print("2. 创建文件结构...")
os.makedirs("app/models", exist_ok=True)

# 创建 __init__.py 文件
with open("app/__init__.py", "w") as f:
    f.write("# App module\n")

with open("app/models/__init__.py", "w") as f:
    f.write("""
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    e_coin_balance = Column(Float, default=0.0)
    python_token_balance = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
""")

# 简单的初始化脚本
print("3. 初始化数据库...")
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    e_coin_balance = Column(Float, default=0.0)
    python_token_balance = Column(Float, default=0.0)

# 创建数据库
engine = create_engine("sqlite:///./edupath.db")
Base.metadata.create_all(engine)

# 创建会话
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# 创建密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 创建管理员用户
admin = User(
    username="admin",
    email="admin@edupath.com",
    hashed_password=pwd_context.hash("admin123"),
    is_admin=True,
    e_coin_balance=1000.0,
    python_token_balance=1000.0
)

db.add(admin)
db.commit()
db.close()

print("=" * 50)
print("数据库初始化成功！")
print("管理员账号: admin / admin123")
print("数据库文件: edupath.db")
print("=" * 50)