from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Transaction
from app.schemas import TransactionResponse
from app.utils.security import get_current_user
from app.services.payment_service import payment_service  # 添加导入
from pydantic import BaseModel

# ... 其余代码保持不变

router = APIRouter()


class RechargeRequest(BaseModel):
    amount: float
    payment_method: str


@router.get("/balance")
def get_balance(current_user: User = Depends(get_current_user)):
    return {
        "eCoin": current_user.e_coin_balance,
        "pythonToken": current_user.python_token_balance
    }


@router.get("/transactions", response_model=List[TransactionResponse])
def get_transactions(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id
    ).order_by(Transaction.created_at.desc()).limit(50).all()
    return transactions


@router.post("/recharge")
def recharge(
        request: RechargeRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    # 模拟充值流程
    current_user.e_coin_balance += request.amount

    # 记录交易
    transaction = Transaction(
        user_id=current_user.id,
        type="recharge",
        amount=request.amount,
        currency="E_COIN",
        description=f"充值 {request.amount} E币"
    )
    db.add(transaction)
    db.commit()

    return {"message": "充值成功", "new_balance": current_user.e_coin_balance}