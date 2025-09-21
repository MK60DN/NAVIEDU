from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import User, Transaction


def buy_python_token(db: Session, user_id: str, amount: float):
    user = db.query(User).filter(User.id == user_id).first()

    if user.e_coin_balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient E-coin balance")

    # 扣除E币，增加$PYTHON
    user.e_coin_balance -= amount
    user.python_token_balance += amount

    # 记录交易
    transaction = Transaction(
        user_id=user_id,
        type="buy_token",
        amount=amount,
        currency="PYTHON",
        description=f"购买 {amount} $PYTHON"
    )
    db.add(transaction)

    db.commit()
    return {
        "message": "Purchase successful",
        "new_e_coin_balance": user.e_coin_balance,
        "new_python_balance": user.python_token_balance
    }


def sell_python_token(db: Session, user_id: str, amount: float):
    user = db.query(User).filter(User.id == user_id).first()

    if user.python_token_balance < amount:
        raise HTTPException(status_code=400, detail="Insufficient $PYTHON balance")

    # 扣除$PYTHON，增加E币
    user.python_token_balance -= amount
    user.e_coin_balance += amount

    # 记录交易
    transaction = Transaction(
        user_id=user_id,
        type="sell_token",
        amount=amount,
        currency="E_COIN",
        description=f"出售 {amount} $PYTHON"
    )
    db.add(transaction)

    db.commit()
    return {
        "message": "Sale successful",
        "new_e_coin_balance": user.e_coin_balance,
        "new_python_balance": user.python_token_balance
    }