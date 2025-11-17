from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "✅ Fake Job Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    job_text = data.get('description', '')

    if not job_text:
        return jsonify({"error": "No job description provided"}), 400

    # Transform and predict
    X_input = vectorizer.transform([job_text])
    prediction = model.predict(X_input)[0]
    label = "Fake Job" if prediction == 1 else "Real Job"

    return jsonify({"prediction": label})

if __name__ == '__main__':
    app.run(debug=True)
