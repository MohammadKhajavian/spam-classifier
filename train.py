import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

print("Loading data from data/spam_data.csv ...")
df = pd.read_csv('data/spam_data.csv')
print(f"Total samples: {len(df)}")

X = df.iloc[:, 1]  # Text column
y = df.iloc[:, 0]  # Label column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model ...")
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

joblib.dump(pipeline, 'model.pkl')
print("Model saved to model.pkl")
