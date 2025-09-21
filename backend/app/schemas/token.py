from pydantic import BaseModel
from datetime import datetime


class TokenResponse(BaseModel):
    id: str
    name: str
    symbol: str
    price: float
    total_supply: float

    class Config:
        from_attributes = True


class TokenExchange(BaseModel):
    amount: float