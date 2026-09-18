from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

FEATURES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "secom/secom.data"
)

LABELS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "secom/secom_labels.data"
)


def download_secom():
    """
    Download and prepare the UCI SECOM semiconductor dataset.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SECOM feature data...")

    X = pd.read_csv(
        FEATURES_URL,
        sep=r"\s+",
        header=None,
        na_values="NaN",
    )

    print("Downloading SECOM labels...")

    labels = pd.read_csv(
        LABELS_URL,
        sep=r"\s+",
        header=None,
        names=["target", "timestamp"],
    )

    # Give sensor columns readable generic names
    X.columns = [
        f"sensor_{i:03d}"
        for i in range(X.shape[1])
    ]

    y = labels[["target"]].copy()

    print("\nDataset downloaded successfully.")

    print(f"Observations: {X.shape[0]}")
    print(f"Features:     {X.shape[1]}")
    print(f"Target shape: {y.shape}")

    print("\nTarget distribution:")
    print(y["target"].value_counts().sort_index())

    print("\nTarget percentages:")
    print(
        y["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nMissing values:")
    print(f"Total missing cells: {X.isna().sum().sum()}")

    # Save local reproducible copies
    features_path = RAW_DATA_DIR / "secom_features.csv"
    target_path = RAW_DATA_DIR / "secom_target.csv"
    labels_path = RAW_DATA_DIR / "secom_labels.csv"

    X.to_csv(
        features_path,
        index=False,
    )

    y.to_csv(
        target_path,
        index=False,
    )

    labels.to_csv(
        labels_path,
        index=False,
    )

    print("\nFiles saved:")
    print(features_path)
    print(target_path)
    print(labels_path)


if __name__ == "__main__":
    download_secom()