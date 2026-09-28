import os
import tempfile
from pathlib import Path

# Configure a throwaway SQLite DB *before* the app is imported.
_tmpdir = tempfile.mkdtemp(prefix="signbridge-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmpdir, 'test.db').as_posix()}"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"
os.environ["REVIEWER_EMAILS"] = "reviewer@example.com"
os.environ["SEED_ON_STARTUP"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.seed import seed_signs  # noqa: E402

FEATURES = 147


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as c:  # runs lifespan: create_all + seed
        yield c
    engine.dispose()


def register(client, email, password="s3cret-pass", role="learner"):
    r = client.post("/auth/register", json={"email": email, "password": password, "role": role})
    assert r.status_code == 201, r.text
    return r.json()


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def learner(client):
    return auth_headers(register(client, "learner@example.com")["access_token"])


@pytest.fixture()
def contributor(client):
    return auth_headers(register(client, "contrib@example.com", role="contributor")["access_token"])


@pytest.fixture()
def reviewer(client):
    data = register(client, "reviewer@example.com")
    assert data["user"]["role"] == "reviewer"
    return auth_headers(data["access_token"])


def make_landmarks(frames=30, features=FEATURES, value=0.5):
    return [[value + i * 0.001] * features for i in range(frames)]


__all__ = ["seed_signs", "SessionLocal"]
