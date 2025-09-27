更新用户的知识图谱。

**请求体:**
```json
{
  "user_id": "user123",
  "graph_data": {
    "id": "root",
    "title": "我的知识库",
    "children": [
      {
        "id": "physics",
        "title": "物理学",
        "content": "物理学基础知识",
        "children": []
      }
    ]
  }
}
```

**响应:**
```json
{
  "status": "success",
  "message": "知识图谱更新成功",
  "graph": {
    "updated_nodes": 1,
    "total_nodes": 15
  }
}
```

## 错误代码

| 状态码 | 说明 | 示例响应 |
|--------|------|----------|
| 400 | 请求参数错误 | `{"error": "缺少必需参数: message"}` |
| 401 | 认证失败 | `{"error": "无效的API密钥"}` |
| 429 | 请求过于频繁 | `{"error": "请求速率限制"}` |
| 500 | 服务器内部错误 | `{"error": "DeepSeek API调用失败"}` |

## 使用示例

### Python 示例

```python
import requests

def call_learning_api(message):
    url = "http://localhost:8000/api/learning"
    data = {
        "message": message,
        "context": []
    }
    
    response = requests.post(url, json=data)
    return response.json()

# 使用示例
result = call_learning_api("什么是机器学习？")
print(result["content"])
```

### JavaScript 示例

```javascript
async function callLearningAPI(message) {
  const response = await fetch('http://localhost:8000/api/learning', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      context: []
    })
  });
  
  return await response.json();
}

// 使用示例
callLearningAPI("解释深度学习的原理").then(result => {
  console.log(result.content);
});
```

## 速率限制

- 每个用户每分钟最多100次请求
- 每个用户每小时最多1000次请求
- 超出限制将返回429状态码

## 注意事项

1. 所有请求都需要包含有效的Content-Type头
2. message字段不能为空且长度不超过2000字符
3. context数组最多包含10条历史消息
4. API响应时间通常在1-5秒之间
```

### docs/SETUP.md
```markdown
# Navi 项目安装指南

## 系统要求

- Node.js 16.0+ 
- Python 3.8+
- npm 或 yarn
- Git

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/navi.git
cd navi
```

### 2. 前端安装

```bash
# 安装依赖
npm install

# 启动开发服务器
npm start
```

前端将在 http://localhost:3000 运行

### 3. 后端安装

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动后端服务
python main.py
```

后端将在 http://localhost:8000 运行

### 4. 配置 DeepSeek API

1. 在 `backend` 目录创建 `.env` 文件
2. 添加您的 DeepSeek API 密钥：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 详细安装步骤

### 前端配置

1. **安装 Node.js**
   - 访问 [Node.js 官网](https://nodejs.org/) 下载安装
   - 验证安装: `node --version` 和 `npm --version`

2. **安装项目依赖**
   ```bash
   npm install
   # 或使用 yarn
   yarn install
   ```

3. **环境变量配置**
   创建 `.env` 文件：
   ```env
   REACT_APP_API_URL=http://localhost:8000
   REACT_APP_APP_NAME=Navi
   ```

4. **启动开发服务器**
   ```bash
   npm start
   ```

### 后端配置

1. **安装 Python**
   - 下载 Python 3.8+ 版本
   - 确保 pip 已安装

2. **创建虚拟环境**
   ```bash
   python -m venv navi-backend
   source navi-backend/bin/activate  # Linux/macOS
   # 或
   navi-backend\Scripts\activate  # Windows
   ```

3. **安装 Python 依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**
   创建 `.env` 文件：
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key
   DEEPSEEK_BASE_URL=https://api.deepseek.com
   API_HOST=0.0.0.0
   API_PORT=8000
   LOG_LEVEL=INFO
   DATA_DIR=./data
   ```

5. **启动后端服务**
   ```bash
   python main.py
   # 或使用 uvicorn
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 生产环境部署

### 前端部署

1. **构建生产版本**
   ```bash
   npm run build
   ```

2. **使用 Nginx 部署**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       root /path/to/navi/build;
       
       location / {
           try_files $uri $uri/ /index.html;
       }
       
       location /api {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### 后端部署

1. **使用 Gunicorn**
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **使用 Docker**
   ```dockerfile
   FROM python:3.8-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8000
   
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

## 常见问题

### Q: 前端启动时出现依赖错误
A: 尝试删除 `node_modules` 文件夹和 `package-lock.json`，然后重新运行 `npm install`

### Q: 后端 API 调用失败
A: 检查 DeepSeek API 密钥是否正确配置，网络连接是否正常

### Q: 知识图谱数据丢失
A: 检查浏览器的本地存储是否被清理，确保应用有写入权限

### Q: 语音功能不工作
A: 确保浏览器支持 Web Speech API，并且已授予麦克风权限

## 开发工具推荐

- **代码编辑器**: VS Code, WebStorm
- **API 测试**: Postman, Insomnia  
- **数据库管理**: DB Browser for SQLite
- **版本控制**: Git, GitHub Desktop

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情