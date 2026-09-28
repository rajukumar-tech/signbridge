from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Sign
from app.schemas import SignOut

router = APIRouter(prefix="/signs", tags=["signs"])


@router.get("", response_model=list[SignOut])
def list_signs(category: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Sign).order_by(Sign.category, Sign.label)
    if category:
        stmt = stmt.where(Sign.category == category)
    return db.scalars(stmt).all()


@router.get("/categories", response_model=list[str])
def list_categories(db: Session = Depends(get_db)):
    return db.scalars(select(Sign.category).distinct().order_by(Sign.category)).all()


@router.get("/{label}", response_model=SignOut)
def get_sign(label: str, db: Session = Depends(get_db)):
    sign = db.scalar(select(Sign).where(Sign.label == label.lower()))
    if not sign:
        raise HTTPException(404, "Sign not found")
    return sign
