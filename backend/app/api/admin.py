from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Capsule, Transaction  # 添加Transaction
from app.schemas import UserResponse, CapsuleResponse, CapsuleUpdate
from app.utils.security import get_current_user, require_admin
from app.models.transaction import TransactionType  # 添加导入

# ... 其余代码保持不变

router = APIRouter()


@router.get("/users", response_model=List[UserResponse])
def get_all_users(
        current_user: User = Depends(require_admin),
        db: Session = Depends(get_db)
):
    users = db.query(User).all()
    return users


@router.get("/capsules/pending", response_model=List[CapsuleResponse])
def get_pending_capsules(
        current_user: User = Depends(require_admin),
        db: Session = Depends(get_db)
):
    capsules = db.query(Capsule).filter(Capsule.status == "pending").all()
    return capsules


@router.post("/capsules/{capsule_id}/approve")
def approve_capsule(
        capsule_id: str,
        current_user: User = Depends(require_admin),
        db: Session = Depends(get_db)
):
    capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
    if not capsule:
        raise HTTPException(status_code=404, detail="Capsule not found")

    capsule.status = "approved"

    # 给贡献者奖励
    author = db.query(User).filter(User.id == capsule.author_id).first()
    if author:
        author.python_token_balance += 10  # 奖励10个$PYTHON

    db.commit()
    return {"message": "Capsule approved"}


@router.post("/capsules/{capsule_id}/reject")
def reject_capsule(
        capsule_id: str,
        current_user: User = Depends(require_admin),
        db: Session = Depends(get_db)
):
    capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
    if not capsule:
        raise HTTPException(status_code=404, detail="Capsule not found")

    capsule.status = "rejected"
    db.commit()
    return {"message": "Capsule rejected"}