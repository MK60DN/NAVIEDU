import uvicorn
import sys
import os
import socket
from contextlib import closing

print("Starting EduPath Backend...")

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_port(host, port):
    """检查端口是否可用"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) != 0


def find_available_port(start_port=8000, max_port=8010):
    """找到可用端口"""
    for port in range(start_port, max_port + 1):
        if check_port('localhost', port):
            return port
    return None


try:
    from app import create_app

    print("✓ App module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import app module: {e}")
    print("Creating minimal app module...")

    # 创建app文件夹
    os.makedirs("app", exist_ok=True)

    # 创建最小的__init__.py
    with open("app/__init__.py", "w", encoding="utf-8") as f:
        f.write("""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    app = FastAPI(title="EduPath API", version="1.0.0")

    # CORS设置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def read_root():
        return {"message": "EduPath API is running", "status": "success"}

    @app.get("/api/health")
    def health_check():
        return {"status": "healthy", "service": "EduPath API"}

    # 模拟登录接口
    @app.post("/api/auth/login")
    def login():
        return {
            "access_token": "mock_token_12345",
            "token_type": "bearer",
            "user": {
                "id": 1,
                "username": "admin",
                "email": "admin@edupath.com",
                "is_admin": True
            }
        }

    # 模拟注册接口
    @app.post("/api/auth/register")
    def register():
        return {"message": "注册成功", "success": True}

    # 模拟用户信息接口
    @app.get("/api/auth/me")
    def get_current_user():
        return {
            "id": 1,
            "username": "admin",
            "email": "admin@edupath.com",
            "is_admin": True
        }

    return app
""")

    from app import create_app

    print("✓ App module created and imported")

try:
    app = create_app()
    print("✓ App created successfully")

    # 查找可用端口
    available_port = find_available_port()

    if available_port is None:
        print("✗ No available ports found in range 8000-8010")
        print("Please manually stop processes using these ports or restart your computer.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("=" * 50)
    print("EduPath Backend Server")
    if available_port != 8000:
        print(f"⚠️  Port 8000 is busy, using port {available_port}")
    print(f"API URL: http://localhost:{available_port}")
    print(f"API Docs: http://localhost:{available_port}/docs")
    print("=" * 50)

    # 如果使用了非8000端口，提示用户更新前端配置
    if available_port != 8000:
        print(f"🔧 请更新前端配置文件 web/js/config.js 中的API_BASE_URL:")
        print(f"   API_BASE_URL: 'http://localhost:{available_port}/api'")
        print("=" * 50)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=available_port,
        reload=False,  # 禁用reload避免端口冲突
        log_level="info"
    )

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\n可能的解决方案:")
    print("1. 关闭其他占用8000端口的程序")
    print("2. 重启计算机")
    print("3. 使用不同的端口")
    print("\nPress Enter to exit...")
    input()
    sys.exit(1)