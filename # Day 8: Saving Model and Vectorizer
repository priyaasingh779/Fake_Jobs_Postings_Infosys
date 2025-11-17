# Day 8: Saving Model and Vectorizer

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load preprocessed data
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['description'])

# Train model (TF-IDF + Logistic Regression)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['description'])
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Model and Vectorizer Saved Successfully!")

# Quick Test
test_jobs = [
    "Work from home with high pay, no experience required! Apply immediately!",
    "We are hiring a software engineer with 2+ years of Python experience."
]

X_test_jobs = vectorizer.transform(test_jobs)
predictions = model.predict(X_test_jobs)

for job, pred in zip(test_jobs, predictions):
    label = "Fake Job" if pred == 1 else "Real Job"
    print(f"\nJob: {job}\nPrediction: {label}")
