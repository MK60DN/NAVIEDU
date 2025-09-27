#!/bin/bash

echo "🧭 启动 Navi 开发服务器"
echo "======================="

# 检查配置文件
if [ ! -f "backend/.env" ]; then
    echo "❌ 配置文件不存在，请先运行 ./scripts/setup.sh"
    exit 1
fi

# 检查 API Key
if ! grep -q "DEEPSEEK_API_KEY=sk-" backend/.env; then
    echo "⚠️ 警告: DeepSeek API Key 可能未配置"
    echo "请在 backend/.env 文件中设置 DEEPSEEK_API_KEY"
fi

echo "🚀 启动后端服务器..."

# 启动后端
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!

cd ..

# 等待后端启动
sleep 3

echo "🎨 启动前端服务器..."

# 启动前端
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ 服务启动完成！"
echo ""
echo "📱 前端地址: http://localhost:3000"
echo "⚡ 后端地址: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"

# 捕获退出信号
trap 'echo ""; echo "🛑 停止服务..."; kill $BACKEND_PID $FRONTEND_PID; exit' INT

# 等待进程结束
wait