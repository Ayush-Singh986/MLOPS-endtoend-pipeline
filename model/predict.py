from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ----------------------------
# SAFE MODEL PATH (Docker-safe)
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.route("/")
def home():
    return jsonify({"status": "model API running"})

# ----------------------------
# PREDICTION
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "features missing"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)