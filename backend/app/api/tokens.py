from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models import Token
from app.schemas import TokenResponse

router = APIRouter()

@router.get("/", response_model=List[TokenResponse])
def get_tokens(db: Session = Depends(get_db)):
    tokens = db.query(Token).all()
    return tokens

@router.get("/{symbol}", response_model=TokenResponse)
def get_token(symbol: str, db: Session = Depends(get_db)):
    token = db.query(Token).filter(Token.symbol == symbol).first()
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    return token