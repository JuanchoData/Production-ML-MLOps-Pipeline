Production ML / MLOps Pipeline for Semiconductor Failure Prediction



End-to-end machine learning and MLOps project for predicting PASS/FAIL outcomes in semiconductor manufacturing from high-dimensional process sensor data. The project demonstrates a reproducible workflow from data ingestion and exploratory analysis through model training, experiment tracking, API serving, automated testing, continuous integration, and Docker containerization.

Portfolio focus: production-oriented data science and MLOps practices rather than only model training.

Project Overview

Semiconductor manufacturing produces large numbers of sensor measurements for each production instance. The goal of this project is to identify manufacturing failures from these process measurements while handling three practical challenges:

High dimensionality: 590 sensor variables.

Missing data: most sensors contain at least some missing observations.

Severe class imbalance: failures represent only about 6.6% of the data.

The project uses the UCI SECOM semiconductor manufacturing dataset and compares a Logistic Regression baseline with a Random Forest classifier.

Dataset

Property

Value

Observations

1,567

Raw sensor features

590

PASS observations

1,463 (93.36%)

FAIL observations

104 (6.64%)

Missing cells

41,951

Sensors with at least one missing value

538

Sensors with >50% missing values in the full dataset

28

Constant features in the full dataset

116

Target encoding:

-1 = PASS
 1 = FAIL

Machine Learning Workflow

flowchart LR
    A[UCI SECOM Data] --> B[EDA]
    B --> C[Train / Validation / Test Split]
    C --> D[Missingness Filter]
    D --> E[Median Imputation]
    E --> F[Constant Feature Removal]
    F --> G1[Logistic Regression]
    F --> G2[Random Forest]
    G1 --> H[Evaluation]
    G2 --> I[Validation Threshold Tuning]
    I --> H
    H --> J[MLflow Tracking]
    J --> K[Saved Model Pipeline]
    K --> L[FastAPI]
    L --> M[pytest]
    M --> N[GitHub Actions CI]
    N --> O[Docker Container]

Preprocessing

The preprocessing pipeline is implemented with scikit-learn and is fitted only on training data to reduce data leakage.

Remove features with more than 50% missing values.

Fill remaining missing values using the training-set median.

Remove zero-variance features.

Standardize features when required by the model.

For the training split used in the pipeline:

590 raw sensors
    ↓
24 sensors removed by training-only missingness filtering
    ↓
median imputation
    ↓
constant-feature removal
    ↓
450 final model features

The difference between the 28 sensors identified during full-data EDA and the 24 removed by the model pipeline occurs because preprocessing thresholds are learned from the training subset only.

Train / Validation / Test Strategy

The Random Forest workflow uses separate training, validation, and test partitions:

Split

Total

PASS

FAIL

Train

1,002

936

66

Validation

251

234

17

Test

314

293

21

The validation set is used to choose the classification threshold. The test set is then evaluated using the locked threshold.

Model Results

Because FAIL cases are rare, accuracy alone is not sufficient. The main metrics are FAIL-class precision, recall, F1, ROC-AUC, PR-AUC, and the confusion matrix.

Test-set comparison

Metric

Logistic Regression

Random Forest

Accuracy

0.8408

0.8981

FAIL Precision

0.1081

0.2609

FAIL Recall

0.1905

0.2857

FAIL F1

0.1379

0.2727

ROC-AUC

0.6203

0.7851

PR-AUC

0.1271

0.2020

Logistic Regression confusion matrix

                 Predicted PASS   Predicted FAIL
Actual PASS            260              33
Actual FAIL             17               4

The Logistic Regression baseline detected 4 of 21 failures and generated 33 false alarms.

Random Forest threshold selection

The Random Forest produced useful ranking performance but few positive predictions at the default threshold of 0.50. A decision threshold was therefore selected on the validation set by maximizing FAIL-class F1.

Selected threshold = 0.13

Validation metrics at the selected threshold:

Metric

Value

FAIL Precision

0.1724

FAIL Recall

0.2941

FAIL F1

0.2174

ROC-AUC

0.7074

PR-AUC

0.1696

Validation confusion matrix:

                 Predicted PASS   Predicted FAIL
Actual PASS            210              24
Actual FAIL             12               5

Random Forest final test results

Using the locked threshold of 0.13:

                 Predicted PASS   Predicted FAIL
Actual PASS            276              17
Actual FAIL             15               6

The model detected 6 of 21 test failures while producing 17 false alarms.

The returned Random Forest score is used as an operational failure score for thresholding. It should not automatically be interpreted as a calibrated probability.

MLflow Experiment Tracking

MLflow is used to record training runs, parameters, model artifacts, and evaluation metrics.

Tracked information includes:

model family

number of trees

class weighting

minimum samples per leaf

feature sampling strategy

selected classification threshold

training / validation / test sizes

FAIL precision, recall, and F1

ROC-AUC

PR-AUC

trained model artifact

Experiments are grouped under:

Semiconductor-Failure-Prediction

Example local MLflow server:

mlflow server --port 5000

Then open:

http://127.0.0.1:5000

FastAPI Model Serving

The selected Random Forest pipeline is exposed through a REST API using FastAPI.

Available endpoints:

Endpoint

Method

Purpose

/

GET

API status

/health

GET

Verify model loading and threshold

/predict

POST

Generate PASS/FAIL prediction

Run locally:

uvicorn src.api:app --host 0.0.0.0 --port 8000

Interactive API documentation:

http://127.0.0.1:8000/docs

Example request:

{
  "sensors": {
    "sensor_000": 3030.93,
    "sensor_001": 2564.0,
    "sensor_002": 2187.7333,
    "sensor_003": 1411.1265,
    "sensor_004": 1.3602
  }
}

Example response from the Dockerized API:

{
  "prediction": "FAIL",
  "prediction_numeric": 1,
  "failure_probability": 0.1659,
  "threshold": 0.13,
  "sensors_provided": 5
}

This five-sensor request is only an API functionality test. A meaningful model inference should provide the available measurements for the manufacturing observation rather than intentionally omitting most sensors.

Automated Testing

API behavior is tested with pytest and FastAPI's test client.

Tests cover:

root endpoint

health endpoint

prediction endpoint

invalid / unknown sensor names

The CI tests mock model loading so GitHub Actions does not require locally generated .joblib artifacts.

Run tests locally:

pytest -v

Continuous Integration

GitHub Actions runs the test suite automatically on pushes and pull requests to main.

The CI workflow:

Git push / Pull request
        ↓
GitHub Actions Ubuntu runner
        ↓
Python 3.11
        ↓
Install dependencies
        ↓
pytest
        ↓
Pass / Fail

Workflow file:

.github/workflows/ci.yml

Docker Deployment

The FastAPI service is packaged into a Linux Docker image with Python 3.11 and the model dependencies.

Build the image:

docker build -t semiconductor-ml-api .

Run the container:

docker run --rm -p 8000:8000 semiconductor-ml-api

Then open:

http://127.0.0.1:8000/docs

Health check:

curl http://127.0.0.1:8000/health

Example response:

{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.13
}

Project Structure

Production-ML-MLOps-Pipeline/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── raw/
│
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── random_forest_threshold.joblib
│
├── notebooks/
│   └── 01_eda.py
│
├── reports/
│   ├── missing_values_summary.csv
│   ├── feature_summary_statistics.csv
│   └── target_distribution.csv
│
├── scripts/
│   ├── prepare_data.py
│   └── show_test_observation.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── threshold.py
│   ├── train.py
│   └── train_random_forest.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements-api.txt
├── requirements.txt
└── README.md

Reproducing the Project

1. Clone the repository

git clone https://github.com/JuanchoData/Production-ML-MLOps-Pipeline.git
cd Production-ML-MLOps-Pipeline

2. Create a Python environment

python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Linux/macOS:

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Download the data

python -m src.data

5. Train the models

Start MLflow in a separate terminal if experiment logging is enabled:

mlflow server --port 5000

Then train:

python -m src.train
python -m src.train_random_forest

6. Run tests

pytest -v

7. Start the API

uvicorn src.api:app --host 0.0.0.0 --port 8000

8. Or run with Docker

docker build -t semiconductor-ml-api .
docker run --rm -p 8000:8000 semiconductor-ml-api

Important Artifact Note

Model files are intentionally excluded from Git using .gitignore:

models/*.joblib
models/*.pkl

Therefore, a clean clone should train the models first before building the current Docker image, because the Dockerfile copies the locally generated Random Forest model and threshold into the image.

A future production implementation could store versioned model artifacts in an MLflow Model Registry, object storage, or another artifact repository rather than relying on local files.

Limitations

This project is designed as a production-oriented MLOps portfolio implementation, not as a claim of production-ready semiconductor yield prediction.

Key limitations include:

The dataset contains only 104 FAIL observations, so failure metrics have substantial uncertainty.

The validation set contains only 17 failures, making the selected threshold sensitive to sampling variation.

Random Forest scores are not explicitly probability-calibrated.

The Logistic Regression and Random Forest training protocols are not perfectly identical because the Random Forest reserves a separate validation subset for threshold selection.

The project currently demonstrates local container deployment rather than managed cloud deployment.

Model drift monitoring is not yet included.

Technology Stack

Data Science / ML

Python

pandas

NumPy

scikit-learn

joblib

MLOps / Software Engineering

MLflow

FastAPI

pytest

GitHub Actions

Docker

Git / GitHub

Key Takeaway

This repository demonstrates how a machine-learning model can move beyond a notebook into a reproducible software system:

Data → Preprocessing → Modeling → Validation → Experiment Tracking
     → Model Artifact → REST API → Automated Tests → CI → Docker

The focus is not only on predictive performance, but also on reproducibility, deployment, testing, version-controlled workflows, and operational model serving.

