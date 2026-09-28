from tests.conftest import auth_headers, register


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_cors_allows_frontend(client):
    r = client.options(
        "/signs",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_register_and_login(client):
    data = register(client, "Alice@Example.com")
    assert data["user"]["email"] == "alice@example.com"
    assert data["user"]["role"] == "learner"
    assert data["access_token"]

    r = client.post("/auth/login", json={"email": "alice@example.com", "password": "s3cret-pass"})
    assert r.status_code == 200
    token = r.json()["access_token"]

    me = client.get("/auth/me", headers=auth_headers(token))
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


def test_oauth2_token_form(client):
    register(client, "bob@example.com")
    r = client.post("/auth/token", data={"username": "bob@example.com", "password": "s3cret-pass"})
    assert r.status_code == 200
    assert r.json()["token_type"] == "bearer"


def test_duplicate_email_rejected(client):
    register(client, "dup@example.com")
    r = client.post("/auth/register", json={"email": "dup@example.com", "password": "another-pass"})
    assert r.status_code == 409


def test_wrong_password(client):
    register(client, "carol@example.com")
    r = client.post("/auth/login", json={"email": "carol@example.com", "password": "wrong-pass"})
    assert r.status_code == 401


def test_cannot_self_assign_reviewer(client):
    r = client.post(
        "/auth/register", json={"email": "sneaky@example.com", "password": "s3cret-pass", "role": "reviewer"}
    )
    assert r.status_code == 422


def test_short_password_rejected(client):
    r = client.post("/auth/register", json={"email": "x@example.com", "password": "short"})
    assert r.status_code == 422


def test_invalid_token(client):
    r = client.get("/auth/me", headers=auth_headers("not-a-jwt"))
    assert r.status_code == 401
    assert client.get("/auth/me").status_code == 401


def test_signs_catalog_seeded(client):
    r = client.get("/signs")
    assert r.status_code == 200
    signs = r.json()
    assert len(signs) >= 50
    labels = {s["label"] for s in signs}
    assert {"hello", "thank_you", "water"} <= labels
    assert client.get("/signs/hello").json()["display_text"].startswith("Hello")
    assert client.get("/signs/nonexistent").status_code == 404
    greetings = client.get("/signs", params={"category": "greetings"}).json()
    assert greetings and all(s["category"] == "greetings" for s in greetings)
