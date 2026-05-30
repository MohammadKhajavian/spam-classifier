import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

print("Loading data from data/spam_data.csv ...")
df = pd.read_csv('data/spam_data.csv')

print(f"Total samples: {len(df)}")

# 🔧 FIXED: Swap these to match your CSV column order
# Your CSV has: "label","text"
# So column 0 is label, column 1 is text
X = df['text']      # This is correct - 'text' is second column
y = df['label']     # This is correct - 'label' is first column

print("Training model ...")
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# Save both the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("Model saved to model.pkl")

# Calculate accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy : {accuracy:.2%}")
