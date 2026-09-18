from pathlib import Path

import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Semiconductor-Failure-Prediction"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    X = pd.read_csv(
        DATA_DIR / "secom_features.csv"
    )

    y = pd.read_csv(
        DATA_DIR / "secom_target.csv"
    )["target"]

    # Original:
    # -1 = PASS
    #  1 = FAIL
    #
    # Convert to:
    #  0 = PASS
    #  1 = FAIL

    y = y.map(
        {
            -1: 0,
             1: 1,
        }
    )

    return X, y


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

def train_model():

    X, y = load_data()


    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )


    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training observations: "
        f"{len(X_train)}"
    )

    print(
        f"Testing observations:  "
        f"{len(X_test)}"
    )


    # --------------------------------------------------
    # LOGISTIC REGRESSION PIPELINE
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(
                    scale=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )


    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "PASS",
                "FAIL",
            ],
            digits=4,
            zero_division=0,
        )
    )


    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    test_precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

    test_pr_auc = average_precision_score(
        y_test,
        y_prob,
    )


    print("\n" + "=" * 70)
    print("PROBABILITY METRICS")
    print("=" * 70)

    print(
        f"ROC-AUC: {test_roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {test_pr_auc:.4f}"
    )


    # --------------------------------------------------
    # SAVE MODEL LOCALLY
    # --------------------------------------------------

    model_path = (
        MODEL_DIR /
        "logistic_regression_pipeline.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"\nModel saved to: "
        f"{model_path}"
    )


    # --------------------------------------------------
    # LOG TO MLFLOW
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="Logistic-Regression-v1"
    ):

        # Parameters

        mlflow.log_params(
            {
                "model_type":
                    "LogisticRegression",

                "class_weight":
                    "balanced",

                "max_iter":
                    2000,

                "random_state":
                    42,

                "missing_threshold":
                    0.50,

                "decision_threshold":
                    0.50,

                "training_observations":
                    len(X_train),

                "testing_observations":
                    len(X_test),

                "training_failures":
                    int(y_train.sum()),

                "testing_failures":
                    int(y_test.sum()),
            }
        )


        # Metrics

        mlflow.log_metrics(
            {
                "test_precision":
                    float(test_precision),

                "test_recall":
                    float(test_recall),

                "test_f1":
                    float(test_f1),

                "test_roc_auc":
                    float(test_roc_auc),

                "test_pr_auc":
                    float(test_pr_auc),
            }
        )


        # Tags

        mlflow.set_tags(
            {
                "dataset":
                    "UCI SECOM",

                "task":
                    "Semiconductor failure detection",

                "model_family":
                    "Logistic Regression",
            }
        )


        # Save model inside MLflow

        mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_regression_pipeline",
            serialization_format="cloudpickle",
            code_paths=["src"],
        )


        print(
            "\nLogistic Regression "
            "logged successfully to MLflow."
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    train_model()