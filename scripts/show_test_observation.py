import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/raw")


X = pd.read_csv(
    DATA_DIR / "secom_features.csv"
)

y = pd.read_csv(
    DATA_DIR / "secom_target.csv"
)["target"]


# Choose one observation
row_index = 0

row = X.iloc[row_index]
target = y.iloc[row_index]


# Remove missing values from JSON
sensors = {
    key: float(value)
    for key, value in row.items()
    if pd.notna(value)
}


payload = {
    "sensors": sensors
}


print(
    f"Row: {row_index}"
)

print(
    f"True label: "
    f"{'FAIL' if target == 1 else 'PASS'}"
)

print(
    f"Sensors available: "
    f"{len(sensors)}"
)

print("\nJSON payload:\n")

print(
    json.dumps(
        payload,
        indent=2,
    )
)