from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class CapsuleCreate(BaseModel):
    title: str
    description: str
    content: str
    code: Optional[str] = None
    category: str


class CapsuleUpdate(BaseModel):
    status: str


class CapsuleResponse(BaseModel):
    id: str
    title: str
    description: str
    content: str
    code: Optional[str]
    image_url: Optional[str]
    category: str
    author_id: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True