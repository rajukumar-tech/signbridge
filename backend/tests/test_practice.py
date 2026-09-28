from datetime import date

from app.routers.practice import day_streak, streaks
from tests.conftest import auth_headers, register


def attempt(client, headers, sign, correct, confidence=0.9, predicted=None):
    r = client.post(
        "/practice/attempts",
        json={
            "sign_label": sign,
            "predicted_label": predicted or (sign if correct else "other"),
            "confidence": confidence,
            "correct": correct,
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_stats_empty(client, learner):
    s = client.get("/practice/stats", headers=learner).json()
    assert s["total_attempts"] == 0
    assert s["overall_accuracy"] == 0.0
    assert s["per_sign"] == []
    assert s["day_streak"] == 0


def test_per_sign_accuracy_and_streaks(client, learner):
    for ok in [True, False, True, True, True]:
        attempt(client, learner, "hello", ok, confidence=0.8)
    for ok in [True, True, False]:
        attempt(client, learner, "water", ok, confidence=0.6)

    s = client.get("/practice/stats", headers=learner).json()
    assert s["total_attempts"] == 8
    assert s["total_correct"] == 6
    assert s["overall_accuracy"] == 0.75
    assert s["current_streak"] == 0  # last attempt overall was wrong
    assert s["best_streak"] == 5  # hello x3 + water x2
    assert s["day_streak"] == 1

    per = {p["sign_label"]: p for p in s["per_sign"]}
    assert per["hello"]["attempts"] == 5
    assert per["hello"]["accuracy"] == 0.8
    assert per["hello"]["current_streak"] == 3
    assert per["hello"]["best_streak"] == 3
    assert per["hello"]["avg_confidence"] == 0.8
    assert per["water"]["accuracy"] == round(2 / 3, 4)
    assert per["water"]["current_streak"] == 0
    assert per["water"]["best_streak"] == 2


def test_stats_are_per_user(client, learner):
    attempt(client, learner, "hello", True)
    other = auth_headers(register(client, "other@example.com")["access_token"])
    assert client.get("/practice/stats", headers=other).json()["total_attempts"] == 0
    assert len(client.get("/practice/attempts", headers=learner).json()) == 1


def test_attempt_validation(client, learner):
    bad = {"sign_label": "hello", "predicted_label": "hello", "confidence": 1.5, "correct": True}
    assert client.post("/practice/attempts", json=bad, headers=learner).status_code == 422
    missing = {"sign_label": "hello", "confidence": 0.5, "correct": True}
    assert client.post("/practice/attempts", json=missing, headers=learner).status_code == 422
    assert client.post("/practice/attempts", json={**bad, "confidence": 0.5}).status_code == 401


def test_streak_helpers():
    assert streaks([]) == (0, 0)
    assert streaks([True, True, False, True]) == (1, 2)
    days = {date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2025, 12, 30)}
    assert day_streak(days, date(2026, 1, 3)) == 3
    assert day_streak(days, date(2026, 1, 4)) == 3  # yesterday counts
    assert day_streak(days, date(2026, 1, 5)) == 0
