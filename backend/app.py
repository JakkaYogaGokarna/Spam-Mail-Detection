

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'spam_pipeline.joblib')

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route('/')
def home():
    return "✅ SpamShield AI is running. Use POST /predict with JSON { 'email': 'your email text' }"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'email' not in data:
            return jsonify({'error': 'Missing key: email'}), 400

        email = data.get('email', '').strip()
        if email == '':
            return jsonify({'error': 'Email content is empty.'}), 400

        if model is None:
            return jsonify({'error': 'Model not loaded properly on server.'}), 500

        prediction = model.predict([email])[0]
        probability = model.predict_proba([email]).max()

        return jsonify({
            'prediction': prediction,
            'confidence': round(probability * 100, 2)
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Run Server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
  