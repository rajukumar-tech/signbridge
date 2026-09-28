from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.db import get_db
from app.models import PracticeAttempt, User
from app.schemas import PracticeAttemptCreate, PracticeAttemptOut, PracticeStats, SignStats

router = APIRouter(prefix="/practice", tags=["practice"])


def streaks(results: list[bool]) -> tuple[int, int]:
    """(current, best) run of consecutive correct answers; `results` is oldest-first."""
    best = run = 0
    for ok in results:
        run = run + 1 if ok else 0
        best = max(best, run)
    return run, best


def day_streak(days: set[date], today: date) -> int:
    """Consecutive practice days ending today (or yesterday, so a streak isn't lost before today's session)."""
    d = today if today in days else today - timedelta(days=1)
    n = 0
    while d in days:
        n += 1
        d -= timedelta(days=1)
    return n


@router.post("/attempts", response_model=PracticeAttemptOut, status_code=status.HTTP_201_CREATED)
def create_attempt(body: PracticeAttemptCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    a = PracticeAttempt(user_id=user.id, **body.model_dump())
    db.add(a)
    db.commit()
    db.refresh(a)
    return a


@router.get("/attempts", response_model=list[PracticeAttemptOut])
def list_attempts(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return db.scalars(
        select(PracticeAttempt)
        .where(PracticeAttempt.user_id == user.id)
        .order_by(PracticeAttempt.created_at.desc(), PracticeAttempt.id.desc())
        .limit(limit)
    ).all()


@router.get("/stats", response_model=PracticeStats)
def stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    attempts = db.scalars(
        select(PracticeAttempt)
        .where(PracticeAttempt.user_id == user.id)
        .order_by(PracticeAttempt.created_at, PracticeAttempt.id)
    ).all()

    by_sign: dict[str, list[PracticeAttempt]] = {}
    for a in attempts:
        by_sign.setdefault(a.sign_label, []).append(a)

    per_sign = []
    for label, items in sorted(by_sign.items()):
        n_ok = sum(a.correct for a in items)
        cur, best = streaks([a.correct for a in items])
        per_sign.append(
            SignStats(
                sign_label=label,
                attempts=len(items),
                correct=n_ok,
                accuracy=round(n_ok / len(items), 4),
                avg_confidence=round(sum(a.confidence for a in items) / len(items), 4),
                current_streak=cur,
                best_streak=best,
                last_attempt_at=items[-1].created_at,
            )
        )

    total = len(attempts)
    total_ok = sum(a.correct for a in attempts)
    cur, best = streaks([a.correct for a in attempts])
    days = {a.created_at.date() for a in attempts}
    return PracticeStats(
        total_attempts=total,
        total_correct=total_ok,
        overall_accuracy=round(total_ok / total, 4) if total else 0.0,
        current_streak=cur,
        best_streak=best,
        day_streak=day_streak(days, datetime.now(timezone.utc).date()),
        per_sign=per_sign,
    )
