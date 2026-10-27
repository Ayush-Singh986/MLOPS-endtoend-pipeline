import os
import unittest


os.environ.setdefault("API_KEY", "test-api-key-that-is-at-least-32-characters")

from predict import app, model  # noqa: E402


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.features = [0.0] * model.n_features_in_
        self.headers = {"X-API-Key": os.environ["API_KEY"]}

    def test_health_endpoints_do_not_require_authentication(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/health/ready").status_code, 200)

    def test_metrics_endpoint_is_available(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"flask_http_request_total", response.data)

    def test_prediction_requires_api_key(self):
        response = self.client.post("/predict", json={"features": self.features})

        self.assertEqual(response.status_code, 401)

    def test_prediction_rejects_incorrect_api_key(self):
        response = self.client.post(
            "/predict",
            json={"features": self.features},
            headers={"X-API-Key": "x" * 40},
        )

        self.assertEqual(response.status_code, 401)

    def test_prediction_rejects_wrong_feature_count(self):
        response = self.client.post(
            "/predict",
            json={"features": self.features[:-1]},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 400)

    def test_prediction_rejects_non_numeric_features(self):
        features = list(self.features)
        features[0] = "not-a-number"
        response = self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 400)

    def test_prediction_rejects_non_finite_features(self):
        features = list(self.features)
        features[0] = float("inf")
        response = self.client.post(
            "/predict",
            json={"features": features},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 400)

    def test_prediction_returns_class_for_valid_input(self):
        response = self.client.post(
            "/predict",
            json={"features": self.features},
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json["prediction"], (0, 1))

if __name__ == "__main__":
    unittest.main()
