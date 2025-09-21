# backend/app/models/knowledge.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base

class KnowledgeCapsule(Base):
    """知识胶囊模型"""
    __tablename__ = "knowledge_capsules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    content = Column(Text, nullable=False)
    code_example = Column(Text)
    difficulty = Column(String(20), default="中级")
    category = Column(String(50), default="编程", index=True)
    estimated_time = Column(String(20), default="30分钟")
    prerequisites = Column(Text)
    parent_id = Column(Integer, ForeignKey("knowledge_capsules.id"))
    status = Column(String(20), default="published")  # published, draft, pending
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 自关联关系
    children = relationship("KnowledgeCapsule", backref="parent", remote_side=[id])

    def __repr__(self):
        return f"<KnowledgeCapsule(id={self.id}, name='{self.name}', category='{self.category}')>"

class UserToken(Base):
    """用户代币模型"""
    __tablename__ = "user_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_type = Column(String(20), nullable=False)  # E_COIN, PYTHON_TOKEN, etc.
    balance = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关系
    user = relationship("User", backref="tokens")

    def __repr__(self):
        return f"<UserToken(user_id={self.user_id}, type='{self.token_type}', balance={self.balance})>"

class LearningProgress(Base):
    """学习进度模型"""
    __tablename__ = "learning_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    capsule_id = Column(Integer, ForeignKey("knowledge_capsules.id"), nullable=False)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    user = relationship("User", backref="learning_progress")
    capsule = relationship("KnowledgeCapsule", backref="learners")

    def __repr__(self):
        return f"<LearningProgress(user_id={self.user_id}, capsule_id={self.capsule_id}, completed={self.completed})>"