from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_roles
from app.db import get_db
from app.models import Contribution, ContributionStatus, Role, User
from app.schemas import (
    FEATURES_PER_FRAME,
    ContributionCreate,
    ContributionDetail,
    ContributionExport,
    ContributionOut,
    ContributionReview,
)

router = APIRouter(prefix="/contributions", tags=["contributions"])

contributor_or_reviewer = require_roles(Role.contributor, Role.reviewer)
reviewer_only = require_roles(Role.reviewer)


@router.post("", response_model=ContributionOut, status_code=status.HTTP_201_CREATED)
def create_contribution(
    body: ContributionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(contributor_or_reviewer),
):
    c = Contribution(
        sign_label=body.sign_label,
        landmarks=body.landmarks,
        num_frames=len(body.landmarks),
        handedness=body.handedness,
        consent=body.consent,
        contributor_id=user.id,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@router.get("/mine", response_model=list[ContributionOut])
def my_contributions(db: Session = Depends(get_db), user: User = Depends(contributor_or_reviewer)):
    return db.scalars(
        select(Contribution).where(Contribution.contributor_id == user.id).order_by(Contribution.created_at.desc())
    ).all()


@router.get("/export", response_model=ContributionExport)
def export_contributions(
    sign_label: str | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(reviewer_only),
):
    """Approved samples as an npz-compatible JSON payload for the training pipeline."""
    stmt = select(Contribution).where(Contribution.status == ContributionStatus.approved).order_by(Contribution.id)
    if sign_label:
        stmt = stmt.where(Contribution.sign_label == sign_label)
    rows = db.scalars(stmt).all()

    classes = sorted({r.sign_label for r in rows})
    idx = {c: i for i, c in enumerate(classes)}
    max_frames = max((r.num_frames for r in rows), default=0)
    pad = [0.0] * FEATURES_PER_FRAME
    X = [r.landmarks + [pad] * (max_frames - r.num_frames) for r in rows]

    return ContributionExport(
        num_samples=len(rows),
        max_frames=max_frames,
        classes=classes,
        arrays={
            "X": X,
            "lengths": [r.num_frames for r in rows],
            "y": [idx[r.sign_label] for r in rows],
            "sign_labels": [r.sign_label for r in rows],
            "handedness": [r.handedness.value for r in rows],
            "ids": [r.id for r in rows],
            "classes": classes,
        },
    )


@router.get("", response_model=list[ContributionOut])
def list_contributions(
    status_: ContributionStatus | None = Query(default=ContributionStatus.pending, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(reviewer_only),
):
    stmt = select(Contribution)
    if status_:
        stmt = stmt.where(Contribution.status == status_)
    stmt = stmt.order_by(Contribution.created_at, Contribution.id).limit(limit).offset(offset)
    return db.scalars(stmt).all()


@router.get("/{contribution_id}", response_model=ContributionDetail)
def get_contribution(contribution_id: int, db: Session = Depends(get_db), _: User = Depends(reviewer_only)):
    c = db.get(Contribution, contribution_id)
    if not c:
        raise HTTPException(404, "Contribution not found")
    return c


def _review(db: Session, contribution_id: int, reviewer: User, new_status: str, note: str | None) -> Contribution:
    c = db.get(Contribution, contribution_id)
    if not c:
        raise HTTPException(404, "Contribution not found")
    c.status = ContributionStatus(new_status)
    c.review_note = note
    c.reviewer_id = reviewer.id
    c.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(c)
    return c


@router.patch("/{contribution_id}", response_model=ContributionOut)
def review_contribution(
    contribution_id: int,
    body: ContributionReview,
    db: Session = Depends(get_db),
    reviewer: User = Depends(reviewer_only),
):
    """Set status to `approved` or `rejected` (with optional note)."""
    return _review(db, contribution_id, reviewer, body.status, body.note)


@router.patch("/{contribution_id}/approve", response_model=ContributionOut)
def approve(contribution_id: int, db: Session = Depends(get_db), reviewer: User = Depends(reviewer_only)):
    return _review(db, contribution_id, reviewer, "approved", None)


@router.patch("/{contribution_id}/reject", response_model=ContributionOut)
def reject(
    contribution_id: int,
    note: str | None = Body(default=None, embed=True, max_length=500),
    db: Session = Depends(get_db),
    reviewer: User = Depends(reviewer_only),
):
    return _review(db, contribution_id, reviewer, "rejected", note)
