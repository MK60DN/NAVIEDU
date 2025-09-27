# 🧭 Navi - 智能学习助手

<div align="center">

![Navi Logo](https://via.placeholder.com/200x200/000000/FBBF24?text=NAVI)

**基于 DeepSeek 的多智能体协作学习平台**

[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-API-FBBF24?style=flat-square)](https://deepseek.com/)

[🚀 快速开始](#快速开始) • [📖 文档](#文档) • [🎯 特性](#特性) • [🤝 贡献](#贡献)

</div>

## ✨ 项目介绍

Navi 是一个革新性的智能学习助手，通过多智能体协作架构，为用户提供个性化的学习体验。三个专业智能体分工协作：

- 🎓 **学习辅导助手** - 提供系统化知识传授和学习指导
- 🤔 **批判思考助手** - 培养批判性思维，提出深度问题
- ⚖️ **协调平衡助手** - 整合观点，构建个人知识图谱

## 🎯 核心特性

### 🧠 多智能体协作
- **智能分工** - 三个专业智能体各司其职，协同工作
- **动态权重** - 根据反馈调整智能体权重，优化学习效果
- **知识融合** - 自动整合不同观点，形成完整知识体系

### 📱 现代化界面
- **双模式设计** - 学习模式（双栏布局）+ 对话模式（自由交流）
- **无缝切换** - 同一问题可在不同模式间自由切换
- **黑黄配色** - DeepSeek 风格的现代化界面设计

### 🌐 多模态交互
- **文本输入** - 支持富文本编辑和 Markdown
- **语音交互** - 语音识别 + 语音合成
- **图片上传** - 支持图片分析和理解

### 🗺️ 知识图谱
- **自动构建** - 基于对话内容自动生成知识节点
- **可视化展示** - 树形结构展示个人知识体系
- **智能关联** - 自动发现知识点之间的关联关系

### 💾 本地优先
- **离线可用** - 核心功能支持离线使用
- **数据安全** - 所有数据存储在本地，保护隐私
- **快速响应** - 本地缓存确保流畅体验

## 🚀 快速开始

### 环境要求

- **Node.js** 16.0+
- **Python** 3.8+
- **DeepSeek API Key** ([获取地址](https://platform.deepseek.com/))

### 一键安装

```bash
# 克隆项目
git clone https://github.com/your-username/navi.git
cd navi

# 安装前端依赖
npm install

# 安装后端依赖
cd backend
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，添加您的 DeepSeek API Key

# 启动开发服务
npm run dev  # 同时启动前端和后端
```

### 分别启动

**前端开发服务器:**
```bash
npm start
# 访问 http://localhost:3000
```

**后端API服务器:**
```bash
cd backend
python main.py
# API地址 http://localhost:8000
```

## 📖 使用指南

### 学习模式

1. **双栏协作** - 左侧学习辅导，右侧批判思考
2. **深度学习** - 系统化知识传授 + 批判性思维训练
3. **知识沉淀** - 自动构建个人知识图谱

### 对话模式

1. **自由交流** - 轻松的日常对话交互
2. **快速问答** - 获得即时回应和帮助
3. **模式切换** - 随时切换到学习模式深入探讨

### 知识图谱

1. **自动生成** - 基于学习内容自动创建知识节点
2. **手动编辑** - 支持手动添加、编辑、删除节点
3. **智能搜索** - 快速找到相关知识点

## 🏗️ 项目架构

```
navi/
├── 📱 前端 (React)
│   ├── 🎨 用户界面组件
│   ├── 🔧 自定义 Hooks
│   ├── 🌐 API 服务层
│   └── 💾 本地存储管理
├── ⚡ 后端 (FastAPI)
│   ├── 🤖 多智能体系统
│   ├── 🔗 DeepSeek API 集成
│   ├── 🗂️ 知识图谱服务
│   └── 💿 数据存储服务
└── 📚 文档和配置
    ├── 📋 API 文档
    ├── 🛠️ 部署指南
    └── 🏛️ 架构说明
```

## 🛠️ 技术栈

### 前端技术
- **React 18** - 用户界面框架
- **React Hooks** - 状态管理和副作用处理
- **Tailwind CSS** - 原子化CSS框架
- **Axios** - HTTP客户端
- **Lucide React** - 图标库

### 后端技术
- **FastAPI** - 现代Python Web框架
- **aiohttp** - 异步HTTP客户端
- **Pydantic** - 数据验证和设置管理
- **Uvicorn** - ASGI服务器

### AI集成
- **DeepSeek API** - 大语言模型服务
- **多智能体架构** - 协作式AI系统
- **知识图谱** - 结构化知识表示

## 📊 项目状态

- ✅ **多智能体协作系统** - 已完成
- ✅ **双模式界面设计** - 已完成  
- ✅ **知识图谱构建** - 已完成
- ✅ **本地数据存储** - 已完成
- ✅ **多模态交互** - 已完成
- 🚧 **语音功能优化** - 开发中
- 🚧 **移动端适配** - 开发中
- 📋 **多语言支持** - 计划中
- 📋 **协作功能** - 计划中

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

1. **🐛 报告问题** - 在 Issues 中报告 bug
2. **💡 功能建议** - 提出新功能想法
3. **📝 改进文档** - 完善项目文档
4. **💻 代码贡献** - 提交 Pull Request

### 开发流程

```bash
# 1. Fork 项目
# 2. 创建功能分支
git checkout -b feature/amazing-feature

# 3. 提交更改
git commit -m 'Add amazing feature'

# 4. 推送到分支
git push origin feature/amazing-feature

# 5. 创建 Pull Request
```

### 代码规范

- **前端**: ESLint + Prettier
- **后端**: Black + isort + mypy
- **提交**: Conventional Commits

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) - 详情请查看 LICENSE 文件

## 🙏 致谢

- [DeepSeek](https://deepseek.com/) - 提供强大的AI能力
- [React](https://reactjs.org/) - 优秀的前端框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Python框架
- 所有为项目做出贡献的开发者们

## 📞 联系我们

- **项目主页**: [GitHub Repository](https://github.com/your-username/navi)
- **问题反馈**: [GitHub Issues](https://github.com/your-username/navi/issues)
- **功能建议**: [GitHub Discussions](https://github.com/your-username/navi/discussions)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请考虑给它一个 Star！**

Made with ❤️ by Navi Team

</div>