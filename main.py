import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Spam Classifier API",
    description="Predicts whether a text message is spam or ham.",
    version="1.0.0",
)

# Load model once at startup
try:
    model = joblib.load("model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError("model.pkl not found. Please run `python train.py` first.")


class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "Win a FREE iPhone now! Click here to claim"}
        }

class PredictResponse(BaseModel):
    label: str
    confidence: float
    message: str


@app.get("/")
def root():
    return {"message": "Spam Classifier API is running. Go to /docs to test."}


@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="The 'text' field cannot be empty."
        )

    # Pipeline handles vectorizing + predicting in one step
    prediction = model.predict([request.text])[0]
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
