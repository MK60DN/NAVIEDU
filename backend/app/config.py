# backend/app/config.py
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # 基础配置
    DATABASE_URL: str = "sqlite:///./edupath.db"
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # API配置
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "EduPath"
    PROJECT_VERSION: str = "1.0.0"

    # 跨域配置
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # AI导师相关配置（可选）
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "your_neo4j_password"
    DEEPSEEK_API_KEY: str = ""

    # DeepSeek API配置
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_MAX_TOKENS: int = 2000
    DEEPSEEK_TEMPERATURE: float = 0.7

    # AI导师功能开关
    AI_TUTOR_ENABLED: bool = False  # MVP阶段默认关闭
    KNOWLEDGE_GRAPH_ENABLED: bool = False  # MVP阶段默认关闭
    CONTRIBUTION_ENABLED: bool = True
    AUTO_REWARD_ENABLED: bool = True

    # 性能配置
    AI_REQUEST_TIMEOUT: int = 30
    MAX_CONVERSATION_HISTORY: int = 10
    CACHE_TTL: int = 300  # 5分钟

    # 安全配置
    MAX_MESSAGE_LENGTH: int = 1000
    MAX_DAILY_REQUESTS: int = 100
    RATE_LIMIT_PER_MINUTE: int = 20

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "standard"
    LOG_FILE: str = "logs/edupath.log"

    # 监控配置
    ENABLE_METRICS: bool = False
    METRICS_PORT: int = 9090

    # 缓存配置
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_DEFAULT_TIMEOUT: int = 300

    # 邮件配置（可选）
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    # 上传文件配置
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_PATH: str = "uploads/"
    ALLOWED_FILE_TYPES: list = [".jpg", ".jpeg", ".png", ".pdf", ".docx"]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # 🔑 关键：允许额外的配置项，避免validation error


# 创建全局配置实例
settings = Settings()