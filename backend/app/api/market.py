from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Transaction
from app.schemas import TokenExchange
from app.utils.security import get_current_user
from app.services import token_service

router = APIRouter()

@router.post("/buy-python")
def buy_python_token(
    exchange: TokenExchange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = token_service.buy_python_token(db, current_user.id, exchange.amount)
    return result

@router.post("/sell-python")
def sell_python_token(
    exchange: TokenExchange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = token_service.sell_python_token(db, current_user.id, exchange.amount)
    return result