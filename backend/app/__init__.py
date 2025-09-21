from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    app = FastAPI(title="EduPath API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def read_root():
        return {"message": "EduPath API is running"}

    @app.post("/api/auth/login")
    def login():
        return {"message": "Login endpoint - not implemented yet"}

    @app.post("/api/auth/register") 
    def register():
        return {"message": "Register endpoint - not implemented yet"}

    return app
