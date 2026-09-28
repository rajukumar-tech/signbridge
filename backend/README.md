# SignBridge API

FastAPI backend for SignBridge. The browser runs sign recognition itself. This API stores accounts, "teach a new sign" contributions (hand landmarks only, never video) and practice scores.

## Run locally (SQLite, no Postgres needed)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

## Run with PostgreSQL

```powershell
docker compose up --build
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Endpoints

| Method | Path | Who | Purpose |
|---|---|---|---|
| POST | `/auth/register` | anyone | Create a learner or contributor account |
| POST | `/auth/login`, `/auth/token` | anyone | Log in (JSON or OAuth2 form), returns a JWT |
| GET | `/auth/me` | logged in | Current user |
| GET | `/signs`, `/signs/categories`, `/signs/{label}` | anyone | ISL vocabulary (about 55 seeded words) |
| POST | `/contributions` | logged in | Upload a landmark sequence (10–120 frames × 147 floats, consent required) |
| GET | `/contributions/mine` | logged in | Your own contributions |
| GET | `/contributions?status=pending` | reviewer | Review queue |
| PATCH | `/contributions/{id}` (or `/approve`, `/reject`) | reviewer | Approve or reject |
| GET | `/contributions/export` | reviewer | Approved data for `training/` |
| POST / GET | `/practice/attempts` | logged in | Record or list practice attempts |
| GET | `/practice/stats` | logged in | Per-sign accuracy and streak |
| GET | `/health` | anyone | Liveness + database check |

Reviewers are not self-selected: list their emails in `REVIEWER_EMAILS`. Set a long random `JWT_SECRET` before deploying. The schema is created on startup; use Alembic for migrations in production.
