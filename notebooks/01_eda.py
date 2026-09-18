from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

X = pd.read_csv(
    DATA_DIR / "secom_features.csv"
)

y = pd.read_csv(
    DATA_DIR / "secom_target.csv"
)


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Observations: {X.shape[0]}")
print(f"Predictors:   {X.shape[1]}")
print(f"Target rows:  {y.shape[0]}")


# --------------------------------------------------
# Data types
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(X.dtypes.value_counts())


# --------------------------------------------------
# Missing values
# --------------------------------------------------

missing_summary = pd.DataFrame(
    {
        "missing_count": X.isna().sum(),
        "missing_percent": X.isna().mean() * 100,
    }
).sort_values(
    "missing_percent",
    ascending=False,
)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print("\nTop 20 features with most missing data:")

print(
    missing_summary
    .head(20)
    .round(2)
)


features_with_missing = (
    missing_summary["missing_count"] > 0
).sum()

print(
    f"\nFeatures containing missing values: "
    f"{features_with_missing}"
)


# --------------------------------------------------
# Features with very high missingness
# --------------------------------------------------

high_missing = missing_summary[
    missing_summary["missing_percent"] > 50
]

print("\nFeatures with >50% missing data:")
print(high_missing)


# --------------------------------------------------
# Constant features
# --------------------------------------------------

nunique = X.nunique(
    dropna=True
)

constant_features = nunique[
    nunique <= 1
]

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

print(
    f"Number of constant features: "
    f"{len(constant_features)}"
)

print(
    constant_features.index.tolist()
)


# --------------------------------------------------
# Near-zero variance / low uniqueness
# --------------------------------------------------

low_unique = nunique[
    nunique <= 5
].sort_values()

print("\n" + "=" * 70)
print("LOW-UNIQUE FEATURES")
print("=" * 70)

print(
    f"Features with <=5 unique values: "
    f"{len(low_unique)}"
)

print(low_unique)


# --------------------------------------------------
# Target distribution
# --------------------------------------------------

target_counts = (
    y["target"]
    .value_counts()
    .sort_index()
)

target_percent = (
    y["target"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

target_summary = pd.DataFrame(
    {
        "count": target_counts,
        "percent": target_percent,
    }
)


print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    target_summary.round(2)
)


# --------------------------------------------------
# Descriptive statistics
# --------------------------------------------------

summary_stats = (
    X.describe()
    .T
)

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    summary_stats.head(20)
)


# --------------------------------------------------
# Save reports
# --------------------------------------------------

missing_summary.to_csv(
    REPORT_DIR / "missing_values_summary.csv"
)

summary_stats.to_csv(
    REPORT_DIR / "feature_summary_statistics.csv"
)

target_summary.to_csv(
    REPORT_DIR / "target_distribution.csv"
)


print("\n" + "=" * 70)
print("REPORTS SAVED")
print("=" * 70)

print(
    REPORT_DIR / "missing_values_summary.csv"
)

print(
    REPORT_DIR / "feature_summary_statistics.csv"
)

print(
    REPORT_DIR / "target_distribution.csv"
)