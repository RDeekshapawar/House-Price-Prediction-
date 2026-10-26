
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "house_price_model.joblib"
)

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "House Price Prediction API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Send a JSON object"
            }), 400

        required = [
            "location", "total_sqft", "bedrooms", "bath"
        ]

        missing = [
            key for key in required
            if key not in payload
        ]

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "fields": missing
            }), 400

        location = str(payload["location"]).strip().lower()

        if not location:
            return jsonify({
                "error": "Location cannot be empty"
            }), 400

        try:
            total_sqft = float(payload["total_sqft"])
            bedrooms = float(payload["bedrooms"])
            bath = float(payload["bath"])
        except (TypeError, ValueError):
            return jsonify({
                "error": "Area, bedrooms and bath must be numbers"
            }), 400

        if not all(map(
            lambda value: pd.notna(value) and abs(value) != float("inf"),
            [total_sqft, bedrooms, bath]
        )):
            return jsonify({
                "error": "Numeric values must be finite"
            }), 400

        if total_sqft <= 0 or bedrooms <= 0 or bath <= 0:
            return jsonify({
                "error": "Area, bedrooms and bath must be positive"
            }), 400

        features = pd.DataFrame([{
            "location": location,
            "total_sqft": total_sqft,
            "bedrooms": bedrooms,
            "bath": bath
        }])

        prediction = float(model.predict(features)[0])

        return jsonify({
            "predicted_price_lakhs": round(prediction, 2),
            "currency": "INR",
            "unit": "lakhs"
        })

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({
            "error": "Unable to generate prediction"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
