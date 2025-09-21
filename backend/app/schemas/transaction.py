from pydantic import BaseModel
from datetime import datetime


class TransactionResponse(BaseModel):
    id: str
    type: str
    amount: float
    currency: str
    description: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True