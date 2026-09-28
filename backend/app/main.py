from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, get_db, init_db
from app.routers import auth, contributions, practice, signs
from app.seed import seed_signs

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # For schema migrations in production, switch to Alembic.
    if settings.seed_on_startup:
        with SessionLocal() as db:
            seed_signs(db)
    yield


app = FastAPI(
    title="SignBridge API",
    version="0.1.0",
    description="Accounts, landmark contributions and practice scores for the SignBridge ISL translator.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(signs.router)
app.include_router(contributions.router)
app.include_router(practice.router)


@app.get("/health", tags=["health"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "database": "ok" if db_ok else "unreachable"}
