from sqlalchemy import Column, String, Float, ForeignKey, DateTime, Integer, Boolean  # 添加Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
import uuid

# ... 其余代码保持不变


class Token(Base):
    __tablename__ = "tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    symbol = Column(String, unique=True, nullable=False)
    price = Column(Float, default=1.0)
    total_supply = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserToken(Base):
    __tablename__ = "user_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    capsule_id = Column(String, ForeignKey("capsules.id"))
    unlocked = Column(Boolean, default=False)
    unlocked_at = Column(DateTime(timezone=True))

    user = relationship("User", backref="unlocked_capsules")
    capsule = relationship("Capsule", backref="unlocks")