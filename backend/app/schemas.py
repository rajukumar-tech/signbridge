from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models import ContributionStatus, Handedness, Role

# 147 = 2 hands x 21 landmarks x 3 coords (126) + 7 upper-body pose points x 3 coords (21)
FEATURES_PER_FRAME = 147
MIN_FRAMES = 10
MAX_FRAMES = 120

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Frame = Annotated[list[FiniteFloat], Field(min_length=FEATURES_PER_FRAME, max_length=FEATURES_PER_FRAME)]
LandmarkSequence = Annotated[list[Frame], Field(min_length=MIN_FRAMES, max_length=MAX_FRAMES)]
Label = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")]


def _normalise_label(v):
    return v.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(v, str) else v


# ---------- auth ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)
    # Reviewer cannot be self-assigned; see REVIEWER_EMAILS in settings.
    role: Literal["learner", "contributor"] = "learner"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    display_name: str | None
    role: Role
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------- signs ----------
class SignOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    label: str
    display_text: str
    category: str
    reference_keyframes: list | dict | None = None


# ---------- contributions ----------
class ContributionCreate(BaseModel):
    sign_label: Label
    landmarks: LandmarkSequence
    consent: bool
    handedness: Handedness

    @field_validator("sign_label", mode="before")
    @classmethod
    def normalise_label(cls, v):
        return _normalise_label(v)

    @field_validator("consent")
    @classmethod
    def consent_required(cls, v: bool) -> bool:
        if v is not True:
            raise ValueError("consent must be true to submit a contribution")
        return v


class ContributionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    sign_label: str
    num_frames: int
    handedness: Handedness
    consent: bool
    status: ContributionStatus
    review_note: str | None
    contributor_id: int
    reviewer_id: int | None
    created_at: datetime
    reviewed_at: datetime | None


class ContributionDetail(ContributionOut):
    landmarks: list[list[float]]


class ContributionReview(BaseModel):
    status: Literal["approved", "rejected"]
    note: str | None = Field(default=None, max_length=500)


class ContributionExport(BaseModel):
    """npz-compatible payload: every entry of `arrays` is a rectangular array-like.

    X: float (N, max_frames, 147), zero-padded; lengths: int (N,); y: int (N,) index into classes;
    sign_labels / handedness / ids: (N,); classes: (C,).
    Convert: np.savez(path, **{k: np.asarray(v) for k, v in payload["arrays"].items()})
    """

    format: str = "signbridge-landmarks-v1"
    features_per_frame: int = FEATURES_PER_FRAME
    num_samples: int
    max_frames: int
    classes: list[str]
    arrays: dict[str, list]


# ---------- practice ----------
class PracticeAttemptCreate(BaseModel):
    sign_label: Label
    predicted_label: Annotated[str, Field(min_length=1, max_length=64)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    correct: bool

    @field_validator("sign_label", "predicted_label", mode="before")
    @classmethod
    def normalise(cls, v):
        return _normalise_label(v)


class PracticeAttemptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    sign_label: str
    predicted_label: str
    confidence: float
    correct: bool
    created_at: datetime


class SignStats(BaseModel):
    sign_label: str
    attempts: int
    correct: int
    accuracy: float
    avg_confidence: float
    current_streak: int
    best_streak: int
    last_attempt_at: datetime | None


class PracticeStats(BaseModel):
    total_attempts: int
    total_correct: int
    overall_accuracy: float
    current_streak: int
    best_streak: int
    day_streak: int
    per_sign: list[SignStats]
