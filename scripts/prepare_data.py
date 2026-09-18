from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Allow imports from project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.preprocessing import build_preprocessor


DATA_DIR = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# Load data
# --------------------------------------------------

X = pd.read_csv(
    DATA_DIR / "secom_features.csv"
)

y = pd.read_csv(
    DATA_DIR / "secom_target.csv"
)["target"]


# --------------------------------------------------
# Convert labels
#
# Original SECOM:
# -1 = PASS
#  1 = FAIL
#
# New:
#  0 = PASS
#  1 = FAIL
# --------------------------------------------------

y = y.map(
    {
        -1: 0,
         1: 1,
    }
)


print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print(
    f"X shape: {X.shape}"
)

print(
    f"y shape: {y.shape}"
)


print("\nTarget counts:")

print(
    y.value_counts()
    .sort_index()
)


print("\nTarget percentages:")

print(
    y.value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# Train-test split
#
# stratify=y is extremely important because
# failures are rare.
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


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training observations: {len(X_train)}"
)

print(
    f"Testing observations:  {len(X_test)}"
)


print("\nTraining target distribution:")

print(
    y_train.value_counts()
    .sort_index()
)


print("\nTesting target distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# --------------------------------------------------
# Build preprocessing pipeline
# --------------------------------------------------

preprocessor = build_preprocessor(
    scale=True
)


# IMPORTANT:
# Fit ONLY using the training data.
X_train_processed = (
    preprocessor.fit_transform(
        X_train,
        y_train,
    )
)


# Test data only gets transformed.
X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# --------------------------------------------------
# Inspect missingness filter
# --------------------------------------------------

missingness_filter = (
    preprocessor.named_steps[
        "missingness_filter"
    ]
)


print("\n" + "=" * 70)
print("MISSINGNESS FILTER")
print("=" * 70)

print(
    "Features removed because of >50% missing:",
    len(
        missingness_filter.columns_to_drop_
    ),
)


print(
    "Features retained:",
    len(
        missingness_filter.columns_to_keep_
    ),
)


print("\nRemoved features:")

print(
    missingness_filter.columns_to_drop_
)


# --------------------------------------------------
# Final processed matrices
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL PREPROCESSED DATA")
print("=" * 70)

print(
    f"Training matrix: {X_train_processed.shape}"
)

print(
    f"Testing matrix:  {X_test_processed.shape}"
)


print(
    "\nMissing values after preprocessing:"
)

print(
    f"Train: "
    f"{pd.isna(X_train_processed).sum()}"
)

print(
    f"Test:  "
    f"{pd.isna(X_test_processed).sum()}"
)