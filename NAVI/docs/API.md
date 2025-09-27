# Navi API 文档

## 概述

Navi API 提供了与智能学习助手交互的RESTful接口，支持学习辅导、批判思考和知识图谱管理功能。

## 基础信息

- 基础URL: `http://localhost:8000`
- 数据格式: JSON
- 认证方式: API Key (在请求头中)

## 端点列表

### 1. 学习辅导 API

#### POST /api/learning

与学习助手进行对话，获取系统化的知识指导。

**请求体:**
```json
{
  "message": "请解释量子力学的基本概念",
  "context": [
    {
      "content": "之前的对话内容",
      "sender": "user"
    }
  ],
  "knowledge_graph": {
    "id": "root",
    "title": "我的知识库",
    "children": []
  }
}
```

**响应:**
```json
{
  "content": "量子力学是描述微观粒子行为的物理理论...",
  "type": "learning",
  "metadata": {
    "tokens_used": 150,
    "response_time": 2.3,
    "quality_score": 0.85
  }
}
```

### 2. 批判思考 API

#### POST /api/questioning

与质疑助手进行对话，获取批判性思维指导。

**请求体:**
```json
{
  "message": "量子纠缠真的能实现瞬时通信吗？",
  "context": [
    {
      "content": "相关学习内容",
      "sender": "assistant"
    }
  ]
}
```

**响应:**
```json
{
  "content": "这是一个很好的质疑。让我们深入思考...",
  "type": "questioning",
  "metadata": {
    "tokens_used": 120,
    "question_count": 3,
    "critical_thinking_score": 0.9
  }
}
```

### 3. 通用对话 API

#### POST /api/chat

进行一般性对话交流。

**请求体:**
```json
{
  "message": "今天天气真好",
  "context": []
}
```

**响应:**
```json
{
  "content": "是的，好天气总是让人心情愉悦...",
  "type": "chat",
  "metadata": {
    "tokens_used": 80,
    "sentiment": "positive"
  }
}