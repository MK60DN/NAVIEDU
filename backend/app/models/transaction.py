from sqlalchemy import Column, String, Float, ForeignKey, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
import uuid
import enum


class TransactionType(str, enum.Enum):
    RECHARGE = "recharge"
    BUY_TOKEN = "buy_token"
    SELL_TOKEN = "sell_token"
    UNLOCK_CAPSULE = "unlock_capsule"
    REWARD = "reward"


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    type = Column(Enum(TransactionType))
    amount = Column(Float)
    currency = Column(String)  # "E_COIN" or "PYTHON"
    description = Column(String)
    status = Column(String, default="completed")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="transactions")