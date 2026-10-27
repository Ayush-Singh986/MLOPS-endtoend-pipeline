import hmac
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, g, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_flask_exporter import PrometheusMetrics
from werkzeug.middleware.proxy_fix import ProxyFix


API_KEY = os.environ.get("API_KEY")
if not API_KEY or len(API_KEY) < 32:
    raise RuntimeError("API_KEY must be configured with at least 32 characters")

MODEL_PATH = Path(__file__).with_name("model.pkl")
model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = int(model.n_features_in_)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

if os.environ.get("TRUST_PROXY", "").lower() in {"1", "true", "yes"}:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
)
PrometheusMetrics(app, group_by="endpoint")


@app.before_request
def start_request_timer():
    g.request_started_at = time.perf_counter()


@app.after_request
def log_request(response):
    elapsed_ms = (time.perf_counter() - g.request_started_at) * 1000
    app.logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="request body too large"), 413


@app.errorhandler(429)
def rate_limit_exceeded(_error):
    return jsonify(error="rate limit exceeded"), 429


@app.get("/")
@app.get("/health/live")
def liveness():
    return jsonify(status="alive")


@app.get("/health/ready")
def readiness():
    if model is None:
        return jsonify(status="not ready"), 503
    return jsonify(status="ready")


def require_api_key():
    supplied_key = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(supplied_key.encode("utf-8"), API_KEY.encode("utf-8")):
        return jsonify(error="unauthorized"), 401
    return None


@app.post("/predict")
@limiter.limit(os.environ.get("PREDICT_RATE_LIMIT", "30 per minute"))
def predict():
    auth_error = require_api_key()
    if auth_error is not None:
        return auth_error

    if not request.is_json:
        return jsonify(error="Content-Type must be application/json"), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="request body must be a JSON object"), 400

    features = payload.get("features")
    if not isinstance(features, list):
        return jsonify(error="features must be a JSON array"), 400
    if len(features) != EXPECTED_FEATURES:
        return jsonify(error=f"features must contain exactly {EXPECTED_FEATURES} values"), 400
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in features):
        return jsonify(error="features must contain only numeric values"), 400

    try:
        feature_array = np.asarray(features, dtype=np.float64)
    except (OverflowError, ValueError):
        return jsonify(error="features must contain finite numeric values"), 400
    if not np.isfinite(feature_array).all():
        return jsonify(error="features must contain only finite values"), 400

    try:
        prediction = model.predict(feature_array.reshape(1, -1))[0]
    except Exception:
        app.logger.exception("model prediction failed")
        return jsonify(error="prediction failed"), 500

    return jsonify(prediction=int(prediction))
