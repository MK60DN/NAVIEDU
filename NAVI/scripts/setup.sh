#!/bin/bash

echo "🧭 Navi 项目安装脚本"
echo "===================="

# 检查依赖
echo "检查系统依赖..."

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js 16.0+"
    exit 1
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 未安装，请先安装 Python 3.8+"
    exit 1
fi

echo "✅ 系统依赖检查完成"

# 安装前端依赖
echo "📦 安装前端依赖..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ 前端依赖安装失败"
    exit 1
fi

echo "✅ 前端依赖安装完成"

# 安装后端依赖
echo "📦 安装后端依赖..."
cd backend

# 创建虚拟环境
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 后端依赖安装失败"
    exit 1
fi

echo "✅ 后端依赖安装完成"

# 创建配置文件
echo "⚙️ 创建配置文件..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 请编辑 backend/.env 文件，添加您的 DeepSeek API Key"
fi

# 创建数据目录
mkdir -p data
mkdir -p data/backups
mkdir -p logs

echo "✅ 配置文件创建完成"

cd ..

echo ""
echo "🎉 安装完成！"
echo ""
echo "下一步："
echo "1. 编辑 backend/.env 文件，添加 DeepSeek API Key"
echo "2. 运行 npm run dev 启动开发服务器"
echo ""
echo "开始您的智能学习之旅吧！🚀"