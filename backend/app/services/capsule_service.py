from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.models import Capsule, UserToken, User, Transaction
from app.schemas import CapsuleCreate
from datetime import datetime


def create_capsule(db: Session, capsule_data: CapsuleCreate, author_id: str):
    capsule = Capsule(
        title=capsule_data.title,
        description=capsule_data.description,
        content=capsule_data.content,
        code=capsule_data.code,
        category=capsule_data.category,
        author_id=author_id,
        status="pending"
    )
    db.add(capsule)
    db.commit()
    db.refresh(capsule)
    return capsule


def unlock_capsule(db: Session, capsule_id: str, user_id: str):
    # 检查胶囊是否存在
    capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
    if not capsule:
        raise HTTPException(status_code=404, detail="Capsule not found")

    # 检查是否已解锁
    existing = db.query(UserToken).filter(
        UserToken.user_id == user_id,
        UserToken.capsule_id == capsule_id
    ).first()

    if existing and existing.unlocked:
        raise HTTPException(status_code=400, detail="Already unlocked")

    # 检查用户余额
    user = db.query(User).filter(User.id == user_id).first()
    if user.python_token_balance < 1:
        raise HTTPException(status_code=400, detail="Insufficient $PYTHON balance")

    # 扣除代币
    user.python_token_balance -= 1

    # 创建或更新解锁记录
    if existing:
        existing.unlocked = True
        existing.unlocked_at = datetime.utcnow()
    else:
        user_token = UserToken(
            user_id=user_id,
            capsule_id=capsule_id,
            unlocked=True,
            unlocked_at=datetime.utcnow()
        )
        db.add(user_token)

    # 记录交易
    transaction = Transaction(
        user_id=user_id,
        type="unlock_capsule",
        amount=-1,
        currency="PYTHON",
        description=f"解锁胶囊: {capsule.title}"
    )
    db.add(transaction)

    db.commit()
    return {"message": "Capsule unlocked successfully"}