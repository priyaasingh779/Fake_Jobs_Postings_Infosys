# Inference: Testing the Saved Model
import joblib

# Load saved files
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example job descriptions
test_jobs = [
    "Work from home with high pay, no experience required! Apply immediately!",
    "We are hiring a software engineer with 2+ years of Python experience."
]

# Transform and predict
X_test_jobs = vectorizer.transform(test_jobs)
predictions = model.predict(X_test_jobs)

# Display predictions
for job, pred in zip(test_jobs, predictions):
    label = "Fake Job" if pred == 1 else "Real Job"
    print(f"\nJob: {job}\nPrediction: {label}")
