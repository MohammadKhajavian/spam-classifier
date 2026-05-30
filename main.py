# =============================================================================
# main.py
# PURPOSE : Serve the trained model as a REST API using FastAPI.
# RUN     : uvicorn main:app --reload
# REQUIRES: model.pkl must exist  (run train.py first)
# TEST    : open http://localhost:8000/docs in your browser
# WEEK    : Week 2 — FastAPI
# =============================================================================

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# 1. Create the FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Spam Classifier API",
    description="Predicts whether a text message is spam or ham.",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# 2. Load the model ONCE at startup
# -----------------------------------------------------------------------------
# We load model.pkl when the server starts — NOT inside the predict function.
# If we loaded it on every request, the API would be very slow.
# model.pkl was created by train.py and contains the full sklearn Pipeline.
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(
        "model.pkl not found. Please run `python train.py` first."
    )

# -----------------------------------------------------------------------------
# 3. Define request and response schemas using Pydantic
# -----------------------------------------------------------------------------
# Pydantic automatically:
#   - Validates incoming JSON (missing fields → 422 error with clear message)
#   - Converts types (string → int if needed)
#   - Documents the API in the /docs Swagger UI

class PredictRequest(BaseModel):
    text: str

    class Config:
        # This shows an example in the /docs Swagger UI
        json_schema_extra = {
            "example": {"text": "Win a FREE iPhone now! Click here to claim"}
        }

class PredictResponse(BaseModel):
    label: str          # "spam" or "ham"
    confidence: float   # probability score between 0.0 and 1.0
    message: str        # human-readable explanation

# -----------------------------------------------------------------------------
# 4. Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Root endpoint — just confirms the API is running.
    """
    return {"message": "Spam Classifier API is running. Go to /docs to test."}


@app.get("/health")
def health():
    """
    Health check endpoint.
    Used by:
      - CI/CD smoke test (ci.yml) to verify the container started correctly
      - Monitoring tools in production
    Always returns 200 OK if the server is alive.
    """
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    Input  (JSON body): {"text": "some message here"}
    Output (JSON)     : {"label": "spam", "confidence": 0.94, "message": "..."}

    Connected to train.py  : uses the sklearn pipeline saved as model.pkl
    Connected to Dockerfile : this entire file runs inside the Docker container
    Connected to test_api.py: the tests call this endpoint
    """
    # Validate that text is not empty
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="The 'text' field cannot be empty."
        )

    # Run prediction
    # model.predict([text]) returns an array like [1] or [0]
    prediction = model.predict([request.text])[0]

    # model.predict_proba([text]) returns [[prob_ham, prob_spam]]
    proba = model.predict_proba([request.text])[0]
    confidence = float(np.max(proba))

    label = "spam" if prediction == 1 else "ham"
    message = (
        f"This message looks like spam ({confidence:.0%} confidence)."
        if label == "spam"
        else f"This message looks like a normal message ({confidence:.0%} confidence)."
    )

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        message=message,
    )
