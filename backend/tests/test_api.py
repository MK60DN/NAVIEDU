import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import create_app
from app.database import Base, get_db
from app.services.auth_service import get_password_hash

# 创建测试数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app = create_app()
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


class TestAuth:
    def test_register(self):
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpass123"
            }
        )
        assert response.status_code == 200
        assert response.json()["username"] == "testuser"

    def test_login(self):
        # 先注册
        client.post(
            "/api/auth/register",
            json={
                "username": "logintest",
                "email": "login@example.com",
                "password": "testpass123"
            }
        )

        # 然后登录
        response = client.post(
            "/api/auth/login",
            json={
                "username": "logintest",
                "password": "testpass123"
            }
        )
        assert response.status_code == 200
        assert "token" in response.json()

    def test_invalid_login(self):
        response = client.post(
            "/api/auth/login",
            json={
                "username": "nonexistent",
                "password": "wrongpass"
            }
        )
        assert response.status_code == 401


class TestCapsules:
    def setup_method(self):
        # 创建测试用户并获取token
        response = client.post(
            "/api/auth/register",
            json={
                "username": "capsuletest",
                "email": "capsule@example.com",
                "password": "testpass123"
            }
        )

        login_response = client.post(
            "/api/auth/login",
            json={
                "username": "capsuletest",
                "password": "testpass123"
            }
        )
        self.token = login_response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_create_capsule(self):
        response = client.post(
            "/api/capsules/",
            json={
                "title": "Test Capsule",
                "description": "Test Description",
                "content": "Test Content",
                "category": "基础语法"
            },
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["title"] == "Test Capsule"

    def test_get_capsules(self):
        response = client.get("/api/capsules/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestWallet:
    def setup_method(self):
        # 创建测试用户并获取token
        response = client.post(
            "/api/auth/register",
            json={
                "username": "wallettest",
                "email": "wallet@example.com",
                "password": "testpass123"
            }
        )

        login_response = client.post(
            "/api/auth/login",
            json={
                "username": "wallettest",
                "password": "testpass123"
            }
        )
        self.token = login_response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_get_balance(self):
        response = client.get("/api/wallet/balance", headers=self.headers)
        assert response.status_code == 200
        assert "eCoin" in response.json()
        assert "pythonToken" in response.json()

    def test_recharge(self):
        response = client.post(
            "/api/wallet/recharge",
            json={
                "amount": 100,
                "payment_method": "wechat"
            },
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["new_balance"] == 100