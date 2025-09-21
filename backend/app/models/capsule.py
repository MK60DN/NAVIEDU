from sqlalchemy import Column, String, Text, ForeignKey, DateTime, Enum, Integer  # 添加Integer
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
import uuid
import enum

# ... 其余代码保持不变


class CapsuleStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Capsule(Base):
    __tablename__ = "capsules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text)
    content = Column(Text, nullable=False)
    code = Column(Text)
    image_url = Column(String)
    category = Column(String)
    author_id = Column(String, ForeignKey("users.id"))
    status = Column(Enum(CapsuleStatus), default=CapsuleStatus.PENDING)
    parent_id = Column(String, ForeignKey("capsules.id"), nullable=True)
    order_index = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    author = relationship("User", backref="capsules")