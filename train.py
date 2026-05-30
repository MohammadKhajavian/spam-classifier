# =============================================================================
# train.py
# PURPOSE : Load data, train a spam classifier, save the model to disk.
# RUN     : python train.py
# PRODUCES: model.pkl  (used by main.py and the Dockerfile)
# WEEK    : Week 1 — ML modeling
# =============================================================================

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
# Read the CSV file from the data/ folder.
# The CSV has two columns: 'label' (spam/ham) and 'text' (message).
DATA_PATH = os.path.join("data", "spam_data.csv")

print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

print(f"Total samples: {len(df)}")
print(f"Spam messages : {(df['label'] == 'spam').sum()}")
print(f"Ham messages  : {(df['label'] == 'ham').sum()}")
print()

# -----------------------------------------------------------------------------
# 2. Prepare features and target
# -----------------------------------------------------------------------------
X = df["text"].tolist()           # list of message strings
y = (df["label"] == "spam").astype(int).tolist()  # 1 = spam, 0 = ham

# -----------------------------------------------------------------------------
# 3. Split into train / test
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print()

# -----------------------------------------------------------------------------
# 4. Build the ML pipeline
# -----------------------------------------------------------------------------
# A Pipeline bundles TWO steps into ONE object:
#   Step 1 - TfidfVectorizer : converts raw text into numbers (TF-IDF matrix)
#   Step 2 - LogisticRegression : binary classifier (spam vs ham)
#
# WHY a Pipeline?
#   - You save ONE file (model.pkl) instead of two separate objects.
#   - At prediction time, new text is automatically vectorized before
#     classification. You just call model.predict(["some text"]).
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,       # convert to lowercase before tokenizing
        stop_words="english", # remove common words like "the", "is", "and"
        ngram_range=(1, 2),   # use single words AND two-word pairs
        max_features=5000,    # keep only the top 5000 most frequent terms
    )),
    ("clf", LogisticRegression(
        max_iter=1000,        # enough iterations to converge
        C=1.0,                # regularization strength (default is fine)
        random_state=42,
    )),
])

# -----------------------------------------------------------------------------
# 5. Train
# -----------------------------------------------------------------------------
print("Training model ...")
pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. Evaluate
# -----------------------------------------------------------------------------
preds = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Accuracy : {accuracy:.2%}")
print()
print("Detailed report:")
print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# -----------------------------------------------------------------------------
# 7. Quick manual test — run a few examples yourself
# -----------------------------------------------------------------------------
test_messages = [
    "Win a FREE iPhone now! Click here to claim",
    "Hey, are we still meeting at 3pm tomorrow?",
    "URGENT: Your account has been compromised",
    "Can you review my pull request when you get a chance?",
]
print("Manual predictions:")
for msg in test_messages:
    pred = pipeline.predict([msg])[0]
    proba = pipeline.predict_proba([msg])[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = max(proba)
    print(f"  [{label} {confidence:.0%}] {msg[:60]}")
print()

# -----------------------------------------------------------------------------
# 8. Save model to disk
# -----------------------------------------------------------------------------
# joblib.dump serializes the entire pipeline (vectorizer + classifier)
# into a single binary file. This is what FastAPI will load.
MODEL_PATH = "model.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
print("Done! You can now run: uvicorn main:app --reload")
