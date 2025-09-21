import os
import sys

print("修复 app 模块...")

# 创建 app 目录
os.makedirs("app", exist_ok=True)
os.makedirs("app/models", exist_ok=True)
os.makedirs("app/api", exist_ok=True)

# 创建 app/__init__.py
with open("app/__init__.py", "w", encoding="utf-8") as f:
    f.write('''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

def create_app():
    app = FastAPI(title="EduPath API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 模拟用户数据
    users = {
        "admin": {
            "password": "admin123",
            "email": "admin@edupath.com",
            "is_admin": True
        }
    }

    @app.get("/")
    def read_root():
        return {"message": "EduPath API is running"}

    @app.post("/api/auth/login")
    def login(request: LoginRequest):
        if request.username in users and users[request.username]["password"] == request.password:
            return {
                "token": "fake-jwt-token-123",
                "user": {
                    "id": "1",
                    "username": request.username,
                    "email": users[request.username]["email"],
                    "is_admin": users[request.username].get("is_admin", False),
                    "e_coin_balance": 100.0,
                    "python_token_balance": 50.0
                }
            }
        return {"error": "Invalid credentials"}, 401

    @app.post("/api/auth/register")
    def register(request: RegisterRequest):
        if request.username in users:
            return {"error": "Username already exists"}, 400

        users[request.username] = {
            "password": request.password,
            "email": request.email,
            "is_admin": False
        }

        return {
            "id": "2",
            "username": request.username,
            "email": request.email,
            "is_admin": False,
            "e_coin_balance": 10.0,
            "python_token_balance": 0.0
        }

    @app.get("/api/auth/me")
    def get_me():
        return {
            "id": "1",
            "username": "admin",
            "email": "admin@edupath.com",
            "is_admin": True,
            "e_coin_balance": 100.0,
            "python_token_balance": 50.0
        }

    return app
''')

print("✓ app/__init__.py 创建完成")

# 创建空的 __init__.py 文件
with open("app/models/__init__.py", "w") as f:
    f.write("# Models module")

with open("app/api/__init__.py", "w") as f:
    f.write("# API module")

print("✓ 目录结构创建完成")
print("现在可以运行: py main.py")