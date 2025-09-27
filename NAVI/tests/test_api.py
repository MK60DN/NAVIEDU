import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

class TestAPI:
    def test_health_check(self):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_learning_endpoint(self):
        """测试学习API端点"""
        response = client.post("/api/learning", json={
            "message": "什么是Python？",
            "context": []
        })
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["type"] == "learning"

    def test_questioning_endpoint(self):
        """测试质疑API端点"""
        response = client.post("/api/questioning", json={
            "message": "Python是最好的编程语言",
            "context": []
        })
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert data["type"] == "questioning"

    def test_invalid_request(self):
        """测试无效请求"""
        response = client.post("/api/learning", json={})
        assert response.status_code == 422  # 验证错误

    def test_knowledge_update(self):
        """测试知识图谱更新"""
        response = client.post("/api/knowledge/update", json={
            "user_id": "test_user",
            "graph_data": {
                "id": "root",
                "title": "测试知识库",
                "children": []
            }
        })
        assert response.status_code == 200