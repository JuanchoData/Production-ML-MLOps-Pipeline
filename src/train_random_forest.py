from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.threshold import find_best_threshold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
# --------------------------------------------------
# MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "Semiconductor-Failure-Prediction"
)


X = pd.read_csv(
    DATA_DIR / "secom_features.csv"
)

y = pd.read_csv(
    DATA_DIR / "secom_target.csv"
)["target"]


# Convert target:
# -1 = PASS -> 0
#  1 = FAIL -> 1

y = y.map({
    -1: 0,
     1: 1,
})


# --------------------------------------------------
# FIRST SPLIT:
# DEVELOPMENT DATA vs TEST DATA
#
# Test data stays separate.
# --------------------------------------------------

X_train_full, X_test, y_train_full, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)


# --------------------------------------------------
# SECOND SPLIT:
# TRAIN vs VALIDATION
#
# Threshold selection happens on validation data.
# --------------------------------------------------

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )
)


print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(f"Training observations:   {len(X_train)}")
print(f"Validation observations: {len(X_val)}")
print(f"Testing observations:    {len(X_test)}")


print("\nTraining failures:")
print(y_train.value_counts().sort_index())

print("\nValidation failures:")
print(y_val.value_counts().sort_index())

print("\nTesting failures:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# RANDOM FOREST PIPELINE
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessing",
            build_preprocessor(
                scale=False
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train,
)


# --------------------------------------------------
# VALIDATION PROBABILITIES
# --------------------------------------------------

y_val_prob = model.predict_proba(
    X_val
)[:, 1]


print("\n" + "=" * 70)
print("VALIDATION PROBABILITY DISTRIBUTION")
print("=" * 70)

print(
    pd.Series(
        y_val_prob,
        name="failure_probability",
    ).describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


# --------------------------------------------------
# FIND BEST THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

threshold_results, best_result = (
    find_best_threshold(
        y_val,
        y_val_prob,
    )
)


best_threshold = best_result["threshold"]


print("\n" + "=" * 70)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 70)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision: "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1: "
    f"{best_result['f1']:.4f}"
)


# --------------------------------------------------
# VALIDATION CLASSIFICATIONS
# --------------------------------------------------

y_val_pred = (
    y_val_prob >= best_threshold
).astype(int)


print("\n" + "=" * 70)
print("VALIDATION CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_matrix(
        y_val,
        y_val_pred,
    )
)


print("\n" + "=" * 70)
print("VALIDATION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_val_pred,
        target_names=[
            "PASS",
            "FAIL",
        ],
        digits=4,
        zero_division=0,
    )
)


# --------------------------------------------------
# VALIDATION PROBABILITY METRICS
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_val,
    y_val_prob,
)

pr_auc = average_precision_score(
    y_val,
    y_val_prob,
)


print("\n" + "=" * 70)
print("VALIDATION PROBABILITY METRICS")
print("=" * 70)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC:  {pr_auc:.4f}"
)

# --------------------------------------------------
# FINAL TEST EVALUATION
#
# IMPORTANT:
# The threshold was selected using validation data.
# We do NOT change it based on test performance.
# --------------------------------------------------

y_test_prob = model.predict_proba(
    X_test
)[:, 1]


y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(
    f"Locked threshold: "
    f"{best_threshold:.2f}"
)


print("\n" + "=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_matrix(
        y_test,
        y_test_pred,
    )
)


print("\n" + "=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "PASS",
            "FAIL",
        ],
        digits=4,
        zero_division=0,
    )
)


test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob,
)

test_pr_auc = average_precision_score(
    y_test,
    y_test_prob,
)


print("\n" + "=" * 70)
print("TEST PROBABILITY METRICS")
print("=" * 70)

print(
    f"ROC-AUC: {test_roc_auc:.4f}"
)

print(
    f"PR-AUC:  {test_pr_auc:.4f}"
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = (
    MODEL_DIR /
    "random_forest_pipeline.joblib"
)

joblib.dump(
    model,
    model_path,
)


# Save selected threshold separately
threshold_path = (
    MODEL_DIR /
    "random_forest_threshold.joblib"
)

joblib.dump(
    best_threshold,
    threshold_path,
)


print("\nModel saved to:")
print(model_path)

print("\nThreshold saved to:")
print(threshold_path)

# --------------------------------------------------
# LOG EXPERIMENT TO MLFLOW
# --------------------------------------------------

with mlflow.start_run(
    run_name="Random-Forest-v1"
):

    # -----------------------------
    # Parameters
    # -----------------------------

    mlflow.log_params(
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "missing_threshold": 0.50,
            "selected_threshold": float(
                best_threshold
            ),
        }
    )

    # -----------------------------
    # Dataset information
    # -----------------------------

    mlflow.log_params(
        {
            "training_observations": len(X_train),
            "validation_observations": len(X_val),
            "testing_observations": len(X_test),
            "training_failures": int(
                y_train.sum()
            ),
            "validation_failures": int(
                y_val.sum()
            ),
            "testing_failures": int(
                y_test.sum()
            ),
        }
    )

    # -----------------------------
    # Validation metrics
    # -----------------------------

    mlflow.log_metrics(
        {
            "validation_precision": float(
                best_result["precision"]
            ),
            "validation_recall": float(
                best_result["recall"]
            ),
            "validation_f1": float(
                best_result["f1"]
            ),
            "validation_roc_auc": float(
                roc_auc
            ),
            "validation_pr_auc": float(
                pr_auc
            ),
        }
    )

    # -----------------------------
    # Test metrics
    # -----------------------------

    mlflow.log_metrics(
        {
            "test_precision": float(
                test_precision
            ),
            "test_recall": float(
                test_recall
            ),
            "test_f1": float(
                test_f1
            ),
            "test_roc_auc": float(
                test_roc_auc
            ),
            "test_pr_auc": float(
                test_pr_auc
            ),
        }
    )

    # -----------------------------
    # Tags
    # -----------------------------

    mlflow.set_tags(
        {
            "dataset": "UCI SECOM",
            "task": "Semiconductor failure detection",
            "model_family": "Random Forest",
        }
    )

    # -----------------------------
    # Log trained sklearn pipeline
    # -----------------------------

    mlflow.sklearn.log_model(
    sk_model=model,
    name="random_forest_pipeline",
    serialization_format="cloudpickle",
    code_paths=["src"],
)

    # -----------------------------
    # Log threshold artifact
    # -----------------------------

    mlflow.log_artifact(
        str(threshold_path),
        artifact_path="threshold",
    )

    print(
        "\nExperiment logged successfully to MLflow."
    )