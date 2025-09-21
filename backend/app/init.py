from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine, Base
from app.api import auth, users, capsules, tokens, market, wallet, admin


def create_app():
    app = FastAPI(title="EduPath API", version="1.0.0")

    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 创建数据库表
    Base.metadata.create_all(bind=engine)

    # 注册路由
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(users.router, prefix="/api/users", tags=["users"])
    app.include_router(capsules.router, prefix="/api/capsules", tags=["capsules"])
    app.include_router(tokens.router, prefix="/api/tokens", tags=["tokens"])
    app.include_router(market.router, prefix="/api/market", tags=["market"])
    app.include_router(wallet.router, prefix="/api/wallet", tags=["wallet"])
    app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

    @app.get("/")
    def read_root():
        return {"message": "EduPath API is running"}

    return app