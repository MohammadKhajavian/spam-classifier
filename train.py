import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

print("Loading data from data/spam_data.csv ...")
df = pd.read_csv('data/spam_data.csv')
print(f"Total samples: {len(df)}")

X = df.iloc[:, 1]  # Text column
y = df.iloc[:, 0]  # Label column

# Split FIRST before anything else
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model ...")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # fit only on train
X_test_vec = vectorizer.transform(X_test)         # only transform test

model = MultinomialNB()
model.fit(X_train_vec, y_train)

accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)
print("Model saved to model.pkl")
