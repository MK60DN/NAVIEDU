from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import User, Capsule, UserToken
from app.schemas import CapsuleCreate, CapsuleResponse
from app.utils.security import get_current_user
from app.services import capsule_service

router = APIRouter()


@router.get("/", response_model=List[CapsuleResponse])
def get_capsules(db: Session = Depends(get_db)):
    capsules = db.query(Capsule).filter(Capsule.status == "approved").all()
    return capsules


@router.get("/my-contributions", response_model=List[CapsuleResponse])
def get_my_contributions(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    capsules = db.query(Capsule).filter(Capsule.author_id == current_user.id).all()
    return capsules


@router.get("/{capsule_id}")
def get_capsule(
        capsule_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
    if not capsule:
        raise HTTPException(status_code=404, detail="Capsule not found")

    # 检查是否已解锁
    unlocked = db.query(UserToken).filter(
        UserToken.user_id == current_user.id,
        UserToken.capsule_id == capsule_id,
        UserToken.unlocked == True
    ).first() is not None

    return {
        "capsule": CapsuleResponse.from_orm(capsule),
        "unlocked": unlocked
    }


@router.post("/", response_model=CapsuleResponse)
def create_capsule(
        capsule_data: CapsuleCreate,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    capsule = capsule_service.create_capsule(db, capsule_data, current_user.id)
    return capsule


@router.post("/{capsule_id}/unlock")
def unlock_capsule(
        capsule_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    result = capsule_service.unlock_capsule(db, capsule_id, current_user.id)
    return result