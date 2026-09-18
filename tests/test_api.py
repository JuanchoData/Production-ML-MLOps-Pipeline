from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient


class DummyModel:
    """
    Fake model used only for API testing.
    """

    def predict_proba(self, X):
        n = len(X)

        return np.array(
            [
                [0.95, 0.05]
                for _ in range(n)
            ]
        )


# api.py calls joblib.load() twice:
# 1. model
# 2. threshold
#
# We replace both during testing.
with patch(
    "joblib.load",
    side_effect=[
        DummyModel(),
        0.13,
    ],
):
    from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["threshold"] == 0.13


def test_predict():
    payload = {
        "sensors": {
            "sensor_000": 3030.93,
            "sensor_001": 2564.0,
            "sensor_002": 2187.7333,
            "sensor_003": 1411.1265,
            "sensor_004": 1.3602,
        }
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "PASS"
    assert data["prediction_numeric"] == 0

    assert (
        data["failure_probability"]
        == 0.05
    )

    assert data["threshold"] == 0.13
    assert data["sensors_provided"] == 5


def test_unknown_sensor():
    payload = {
        "sensors": {
            "sensor_999": 123.4
        }
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400