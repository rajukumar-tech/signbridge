"""Seed the signs catalog with a starter ISL vocabulary (idempotent).

Run manually:  python -m app.seed
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Sign

# (label, display_text, category)
ISL_VOCABULARY: list[tuple[str, str, str]] = [
    # greetings & courtesy
    ("hello", "Hello / Namaste", "greetings"),
    ("goodbye", "Goodbye", "greetings"),
    ("thank_you", "Thank you", "greetings"),
    ("please", "Please", "greetings"),
    ("sorry", "Sorry", "greetings"),
    ("welcome", "Welcome", "greetings"),
    ("good_morning", "Good morning", "greetings"),
    ("good_night", "Good night", "greetings"),
    ("how_are_you", "How are you?", "greetings"),
    ("fine", "Fine", "greetings"),
    # responses
    ("yes", "Yes", "responses"),
    ("no", "No", "responses"),
    ("ok", "OK", "responses"),
    ("help", "Help", "responses"),
    ("understand", "Understand", "responses"),
    # people & family
    ("i", "I / Me", "people"),
    ("you", "You", "people"),
    ("mother", "Mother", "people"),
    ("father", "Father", "people"),
    ("brother", "Brother", "people"),
    ("sister", "Sister", "people"),
    ("friend", "Friend", "people"),
    ("teacher", "Teacher", "people"),
    ("doctor", "Doctor", "people"),
    ("family", "Family", "people"),
    # questions
    ("what", "What?", "questions"),
    ("where", "Where?", "questions"),
    ("when", "When?", "questions"),
    ("why", "Why?", "questions"),
    ("who", "Who?", "questions"),
    ("how", "How?", "questions"),
    # daily life
    ("water", "Water", "daily_life"),
    ("food", "Food", "daily_life"),
    ("eat", "Eat", "daily_life"),
    ("drink", "Drink", "daily_life"),
    ("home", "Home", "daily_life"),
    ("school", "School", "daily_life"),
    ("work", "Work", "daily_life"),
    ("hospital", "Hospital", "daily_life"),
    ("toilet", "Toilet", "daily_life"),
    ("sleep", "Sleep", "daily_life"),
    ("money", "Money", "daily_life"),
    # feelings
    ("happy", "Happy", "feelings"),
    ("sad", "Sad", "feelings"),
    ("angry", "Angry", "feelings"),
    ("love", "Love", "feelings"),
    ("pain", "Pain", "feelings"),
    # time
    ("today", "Today", "time"),
    ("tomorrow", "Tomorrow", "time"),
    ("yesterday", "Yesterday", "time"),
    ("time", "Time", "time"),
    # numbers
    ("one", "One (1)", "numbers"),
    ("two", "Two (2)", "numbers"),
    ("three", "Three (3)", "numbers"),
    ("four", "Four (4)", "numbers"),
    ("five", "Five (5)", "numbers"),
]


def seed_signs(db: Session) -> int:
    """Insert any missing signs. Returns number inserted."""
    existing = set(db.scalars(select(Sign.label)).all())
    new = [
        Sign(label=label, display_text=text, category=cat, reference_keyframes=None)
        for label, text, cat in ISL_VOCABULARY
        if label not in existing
    ]
    db.add_all(new)
    db.commit()
    return len(new)


if __name__ == "__main__":
    from app.db import SessionLocal, init_db

    init_db()
    with SessionLocal() as session:
        print(f"Inserted {seed_signs(session)} signs")
