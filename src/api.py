from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = Path(
    "models/random_forest_pipeline.joblib"
)

THRESHOLD_PATH = Path(
    "models/random_forest_threshold.joblib"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(
    MODEL_PATH
)

threshold = joblib.load(
    THRESHOLD_PATH
)


# --------------------------------------------------
# EXPECTED SENSOR NAMES
# --------------------------------------------------

FEATURE_NAMES = [
    f"sensor_{i:03d}"
    for i in range(590)
]


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Semiconductor Failure Prediction API",
    description=(
        "Predict semiconductor manufacturing "
        "PASS/FAIL outcomes from process sensor data."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# REQUEST FORMAT
# --------------------------------------------------

class SensorInput(BaseModel):

    sensors: dict[
        str,
        float | None
    ]


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message":
            "Semiconductor Failure Prediction API",
        "status":
            "running",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold": float(threshold),
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(
    request: SensorInput
):

    sensors = request.sensors

    # ----------------------------------------------
    # Check for unknown sensor names
    # ----------------------------------------------

    unknown_sensors = [
        sensor
        for sensor in sensors
        if sensor not in FEATURE_NAMES
    ]

    if unknown_sensors:

        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Unknown sensor names detected.",
                "unknown_sensors":
                    unknown_sensors,
            },
        )


    # ----------------------------------------------
    # Create one complete 590-feature observation
    #
    # Missing sensors become NaN.
    # The preprocessing pipeline handles them.
    # ----------------------------------------------

    row = {
        feature: np.nan
        for feature in FEATURE_NAMES
    }

    row.update(
        sensors
    )


    X_new = pd.DataFrame(
        [row],
        columns=FEATURE_NAMES,
    )


    # ----------------------------------------------
    # PREDICT FAILURE PROBABILITY
    # ----------------------------------------------

    failure_probability = (
        model.predict_proba(
            X_new
        )[0, 1]
    )


    # ----------------------------------------------
    # APPLY OUR VALIDATED THRESHOLD
    # ----------------------------------------------

    prediction_numeric = int(
        failure_probability
        >= threshold
    )

    prediction_label = (
        "FAIL"
        if prediction_numeric == 1
        else "PASS"
    )


    # ----------------------------------------------
    # RETURN JSON RESPONSE
    # ----------------------------------------------

    return {
        "prediction":
            prediction_label,

        "prediction_numeric":
            prediction_numeric,

        "failure_probability":
            round(
                float(
                    failure_probability
                ),
                4,
            ),

        "threshold":
            float(
                threshold
            ),

        "sensors_provided":
            len(sensors),
    }