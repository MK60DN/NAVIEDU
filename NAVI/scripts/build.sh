#!/bin/bash

echo "🏗️ 构建 Navi 生产版本"
echo "===================="

# 清理之前的构建
echo "🧹 清理旧文件..."
rm -rf build/
rm -rf dist/

# 构建前端
echo "📦 构建前端..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ 前端构建失败"
    exit 1
fi

echo "✅ 前端构建完成"

# 打包后端
echo "📦 打包后端..."
cd backend

# 创建分发目录
mkdir -p ../dist/backend

# 复制必要文件
cp -r *.py ../dist/backend/
cp -r agents/ ../dist/backend/
cp -r services/ ../dist/backend/
cp requirements.txt ../dist/backend/
cp .env.example ../dist/backend/

echo "✅ 后端打包完成"

cd ..

# 复制前端构建文件
cp -r build/ dist/frontend/

# 创建启动脚本
cat > dist/start.sh << 'EOF'
#!/bin/bash
echo "🧭 启动 Navi 生产服务器"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 未安装"
    exit 1
fi

cd backend

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
EOF

chmod +x dist/start.sh

echo ""
echo "🎉 构建完成！"
echo ""
echo "📂 构建文件位置: ./dist/"
echo "🚀 运行: cd dist && ./start.sh"