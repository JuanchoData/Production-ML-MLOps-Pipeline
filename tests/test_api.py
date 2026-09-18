from fastapi.testclient import TestClient

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

    assert data["prediction"] in [
        "PASS",
        "FAIL",
    ]

    assert data["prediction_numeric"] in [
        0,
        1,
    ]

    assert 0 <= data["failure_probability"] <= 1

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