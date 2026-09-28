import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    learner = "learner"
    contributor = "contributor"
    reviewer = "reviewer"


class ContributionStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Handedness(str, enum.Enum):
    left = "left"
    right = "right"
    both = "both"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    role: Mapped[Role] = mapped_column(Enum(Role, native_enum=False, length=20), default=Role.learner)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    contributions: Mapped[list["Contribution"]] = relationship(
        back_populates="contributor", foreign_keys="Contribution.contributor_id"
    )


class Sign(Base):
    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_text: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(64), index=True)
    reference_keyframes: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)


class Contribution(Base):
    __tablename__ = "contributions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sign_label: Mapped[str] = mapped_column(String(64), index=True)
    landmarks: Mapped[list] = mapped_column(JSON)
    num_frames: Mapped[int] = mapped_column(Integer)
    handedness: Mapped[Handedness] = mapped_column(Enum(Handedness, native_enum=False, length=10))
    consent: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[ContributionStatus] = mapped_column(
        Enum(ContributionStatus, native_enum=False, length=10), default=ContributionStatus.pending, index=True
    )
    review_note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    contributor_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    contributor: Mapped[User] = relationship(back_populates="contributions", foreign_keys=[contributor_id])


class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    sign_label: Mapped[str] = mapped_column(String(64), index=True)
    predicted_label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    correct: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
