import pytest

from tests.conftest import make_landmarks


def payload(**overrides):
    body = {
        "sign_label": "hello",
        "landmarks": make_landmarks(30),
        "consent": True,
        "handedness": "right",
    }
    body.update(overrides)
    return body


def test_contributor_can_submit(client, contributor):
    r = client.post("/contributions", json=payload(sign_label="Thank You"), headers=contributor)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["status"] == "pending"
    assert data["num_frames"] == 30
    assert data["sign_label"] == "thank_you"
    assert "landmarks" not in data


@pytest.mark.parametrize("frames", [10, 120])
def test_frame_count_boundaries_accepted(client, contributor, frames):
    r = client.post("/contributions", json=payload(landmarks=make_landmarks(frames)), headers=contributor)
    assert r.status_code == 201, r.text


@pytest.mark.parametrize(
    "landmarks",
    [
        make_landmarks(9),  # too few frames
        make_landmarks(121),  # too many frames
        make_landmarks(20, features=146),  # frame too short
        make_landmarks(20, features=148),  # frame too long
        make_landmarks(19) + [[0.1] * 100],  # one bad frame
        [],
    ],
    ids=["9-frames", "121-frames", "146-floats", "148-floats", "one-bad-frame", "empty"],
)
def test_invalid_landmarks_rejected(client, contributor, landmarks):
    r = client.post("/contributions", json=payload(landmarks=landmarks), headers=contributor)
    assert r.status_code == 422


def test_non_numeric_landmarks_rejected(client, contributor):
    lm = make_landmarks(20)
    lm[5][3] = "abc"
    r = client.post("/contributions", json=payload(landmarks=lm), headers=contributor)
    assert r.status_code == 422


def test_consent_required(client, contributor):
    r = client.post("/contributions", json=payload(consent=False), headers=contributor)
    assert r.status_code == 422
    body = payload()
    del body["consent"]
    assert client.post("/contributions", json=body, headers=contributor).status_code == 422


def test_invalid_handedness(client, contributor):
    r = client.post("/contributions", json=payload(handedness="ambidextrous"), headers=contributor)
    assert r.status_code == 422


def test_learner_cannot_contribute(client, learner):
    assert client.post("/contributions", json=payload(), headers=learner).status_code == 403


def test_unauthenticated_cannot_contribute(client):
    assert client.post("/contributions", json=payload()).status_code == 401


def test_review_flow_and_export(client, contributor, reviewer):
    ids = []
    for label, frames in [("hello", 12), ("water", 20), ("hello", 15)]:
        r = client.post(
            "/contributions", json=payload(sign_label=label, landmarks=make_landmarks(frames)), headers=contributor
        )
        ids.append(r.json()["id"])

    # contributors can't list the review queue
    assert client.get("/contributions", params={"status": "pending"}, headers=contributor).status_code == 403

    pending = client.get("/contributions", params={"status": "pending"}, headers=reviewer).json()
    assert [c["id"] for c in pending] == ids

    detail = client.get(f"/contributions/{ids[0]}", headers=reviewer).json()
    assert len(detail["landmarks"]) == 12 and len(detail["landmarks"][0]) == 147

    assert client.patch(f"/contributions/{ids[0]}/approve", headers=reviewer).json()["status"] == "approved"
    r = client.patch(f"/contributions/{ids[1]}", json={"status": "approved"}, headers=reviewer)
    assert r.json()["status"] == "approved"
    r = client.patch(f"/contributions/{ids[2]}/reject", json={"note": "blurry hands"}, headers=reviewer)
    assert r.json()["status"] == "rejected" and r.json()["review_note"] == "blurry hands"
    assert client.patch(f"/contributions/{ids[0]}", json={"status": "maybe"}, headers=reviewer).status_code == 422
    assert client.patch("/contributions/9999/approve", headers=reviewer).status_code == 404

    assert client.get("/contributions", params={"status": "pending"}, headers=reviewer).json() == []

    # export: reviewer only
    assert client.get("/contributions/export", headers=contributor).status_code == 403
    exp = client.get("/contributions/export", headers=reviewer).json()
    assert exp["num_samples"] == 2
    assert exp["classes"] == ["hello", "water"]
    assert exp["max_frames"] == 20
    arrays = exp["arrays"]
    assert arrays["lengths"] == [12, 20]
    assert arrays["y"] == [0, 1]
    assert arrays["handedness"] == ["right", "right"]
    # rectangular (N, max_frames, 147) → np.asarray-able
    assert len(arrays["X"]) == 2
    assert all(len(sample) == 20 and all(len(f) == 147 for f in sample) for sample in arrays["X"])
    assert arrays["X"][0][12] == [0.0] * 147  # padding

    mine = client.get("/contributions/mine", headers=contributor).json()
    assert len(mine) == 3
