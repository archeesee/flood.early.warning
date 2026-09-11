"""
Flood Risk Early-Warning Alert System
--------------------------------------
Instead of a lookup app users have to check, this backend:
  1. Loads the trained regression model (from the notebook).
  2. Lets "regions" register subscribers (name + phone + area).
  3. Scores incoming regional risk data with the model.
  4. When predicted FloodProbability crosses a danger threshold,
     automatically pushes an SMS-style alert to every subscriber in that area
     (real SMS via Twilio if credentials are set, otherwise a realistic
     simulated log — perfect for a live demo without needing paid credentials).

Run: python app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_DIR = "model_artifacts"
DATA_FILE = "subscribers.json"
ALERT_LOG_FILE = "alert_log.json"
RISK_THRESHOLD = float(os.environ.get("FLOOD_RISK_THRESHOLD", 0.65))  # tune after seeing model output range

# ---------------- Twilio (optional real SMS) ----------------
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER")
TWILIO_ENABLED = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM])

if TWILIO_ENABLED:
    from twilio.rest import Client
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)


def send_sms(to_number: str, message: str):
    """Send a real SMS via Twilio if configured, else simulate + log it."""
    if TWILIO_ENABLED:
        try:
            twilio_client.messages.create(body=message, from_=TWILIO_FROM, to=to_number)
            return {"status": "sent", "mode": "twilio"}
        except Exception as e:
            return {"status": "failed", "mode": "twilio", "error": str(e)}
    else:
        # Simulated send — still logged and shown on the dashboard for the demo
        return {"status": "simulated", "mode": "simulation"}


# ---------------- Model loading ----------------
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "flood_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    uses_scaling = joblib.load(os.path.join(MODEL_DIR, "uses_scaling.pkl"))
    return model, scaler, feature_columns, uses_scaling


MODEL_LOADED = False
try:
    model, scaler, feature_columns, uses_scaling = load_model()
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARN] Model artifacts not found yet ({e}). Run the notebook first, "
          f"then restart this app. The UI will still load in demo mode.")


# ---------------- Storage helpers ----------------
def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED, threshold=RISK_THRESHOLD,
                            twilio_enabled=TWILIO_ENABLED)


@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name, phone, area = data.get("name"), data.get("phone"), data.get("area")
    if not all([name, phone, area]):
        return jsonify({"error": "name, phone, and area are required"}), 400

    subscribers = _load_json(DATA_FILE, [])
    subscribers.append({"name": name, "phone": phone, "area": area,
                         "registered_at": datetime.utcnow().isoformat()})
    _save_json(DATA_FILE, subscribers)
    return jsonify({"status": "registered", "total_subscribers": len(subscribers)})


@app.route("/api/subscribers")
def list_subscribers():
    return jsonify(_load_json(DATA_FILE, []))


@app.route("/api/predict", methods=["POST"])
def predict():
    """Score a region's factor readings and broadcast alerts if risk is high."""
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded — run the notebook to produce model_artifacts/*.pkl"}), 503

    payload = request.json
    area = payload.get("area", "Unknown Area")
    factors = payload.get("factors", {})

    row = pd.DataFrame([{col: factors.get(col, 0) for col in feature_columns}])
    X_input = scaler.transform(row) if uses_scaling else row.values

    risk_score = float(model.predict(X_input)[0])
    risk_score = max(0.0, min(1.0, risk_score))
    alert_triggered = risk_score >= RISK_THRESHOLD

    result = {
        "area": area,
        "risk_score": round(risk_score, 4),
        "threshold": RISK_THRESHOLD,
        "alert_triggered": alert_triggered,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if alert_triggered:
        subscribers = [s for s in _load_json(DATA_FILE, []) if s["area"].lower() == area.lower()]
        message = (f"⚠ FLOOD ALERT — {area}: Predicted flood risk is HIGH "
                   f"({risk_score*100:.0f}%). Please move valuables to higher ground and "
                   f"stay updated with local authorities.")
        send_results = []
        for s in subscribers:
            send_results.append({"to": s["phone"], "name": s["name"], **send_sms(s["phone"], message)})

        result["alerts_sent"] = send_results
        result["message"] = message

        log = _load_json(ALERT_LOG_FILE, [])
        log.append(result)
        _save_json(ALERT_LOG_FILE, log)

    return jsonify(result)


@app.route("/api/alert-log")
def alert_log():
    return jsonify(_load_json(ALERT_LOG_FILE, []))


@app.route("/api/feature-columns")
def feature_cols():
    if not MODEL_LOADED:
        return jsonify([])
    return jsonify(feature_columns)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
