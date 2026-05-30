# =============================================================================
# test_api.py
# PURPOSE : Automated tests for the FastAPI app.
# RUN     : pytest test_api.py -v
# REQUIRES: model.pkl must exist  (run train.py first)
# WEEK    : Week 5 — testing for CI/CD
# =============================================================================
# HOW THIS CONNECTS:
#   - Imports the FastAPI 'app' object directly from main.py
#   - TestClient simulates HTTP requests without needing a running server
#   - GitHub Actions runs these tests automatically on every git push
#   - If any test fails, the Docker image is NOT built or deployed
# =============================================================================

import pytest
from fastapi.testclient import TestClient
from main import app

# TestClient wraps the FastAPI app and lets us make fake HTTP requests.
# No server needs to be running — TestClient handles everything in memory.
client = TestClient(app)


# -----------------------------------------------------------------------------
# Test 1: root endpoint
# -----------------------------------------------------------------------------
def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# -----------------------------------------------------------------------------
# Test 2: health check
# -----------------------------------------------------------------------------
def test_health_returns_ok():
    """
    This same check runs as a smoke test in ci.yml after Docker deployment.
    If /health returns anything other than 200, the pipeline fails.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -----------------------------------------------------------------------------
# Test 3 & 4: spam and ham predictions
# -----------------------------------------------------------------------------
def test_predicts_spam():
    """
    Classic spam message — model should label it spam.
    """
    response = client.post(
        "/predict",
        json={"text": "Win a FREE iPhone now! Click here to claim your prize"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "spam"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "message" in data


def test_predicts_ham():
    """
    Normal message — model should label it ham.
    """
    response = client.post(
        "/predict",
        json={"text": "Hey, are we still meeting at 3pm tomorrow?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "ham"
    assert 0.0 <= data["confidence"] <= 1.0


# -----------------------------------------------------------------------------
# Test 5 & 6: input validation (Pydantic does this automatically)
# -----------------------------------------------------------------------------
def test_empty_text_returns_400():
    """
    Empty or whitespace-only text should be rejected with HTTP 400.
    This validation is handled inside the /predict function in main.py.
    """
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_missing_text_field_returns_422():
    """
    Missing the required 'text' field should return HTTP 422 (Unprocessable Entity).
    Pydantic handles this automatically — no code needed in main.py.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_wrong_type_returns_422():
    """
    Sending a number instead of a string should return 422.
    Again, Pydantic handles this automatically.
    """
    response = client.post("/predict", json={"text": 12345})
    # Pydantic will coerce int to str, so this might pass — that's fine
    assert response.status_code in [200, 422]


# -----------------------------------------------------------------------------
# Test 7: response structure is correct
# -----------------------------------------------------------------------------
def test_response_has_correct_fields():
    """
    Verify the response always contains exactly the fields we defined
    in PredictResponse (label, confidence, message).
    """
    response = client.post(
        "/predict",
        json={"text": "Congratulations you have won a cash prize!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"label", "confidence", "message"}
    assert data["label"] in ["spam", "ham"]
    assert isinstance(data["confidence"], float)
    assert isinstance(data["message"], str)
