import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

input_file = Path("data/interim/emit_geology.csv")

engineered_file = Path(
    "data/interim/emit_engineered.csv"
)

balanced_file = Path(
    "data/processed/gnn_dataset.csv"
)

balanced_file.parent.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 70)
print("SPECTRAL FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading spatially labeled EMIT dataset...")

df = pd.read_csv(input_file)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# ORIGINAL SPECTRAL FEATURES
# ============================================================

spectral_columns = [
    "reflectance_448nm",
    "reflectance_552nm",
    "reflectance_649nm",
    "reflectance_701nm",
    "reflectance_798nm",
    "reflectance_902nm",
    "reflectance_999nm",
    "reflectance_1201nm",
    "reflectance_1603nm",
    "reflectance_1796nm",
    "reflectance_1997nm",
    "reflectance_2197nm",
    "reflectance_2301nm",
    "reflectance_2397nm"
]

print("\nChecking spectral columns...")

missing = [
    col for col in spectral_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing spectral columns: {missing}"
    )


# ============================================================
# CLEAN ONLY SPECTRAL VALUES
# ============================================================

print("\nCleaning spectral values...")

df[spectral_columns] = df[spectral_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(df)

df = df.dropna(
    subset=spectral_columns,
    how="any"
).copy()

after = len(df)

print(f"Rows before spectral cleaning: {before:,}")
print(f"Rows after spectral cleaning:  {after:,}")


# ============================================================
# SPECTRAL RATIOS
# ============================================================

print("\nCreating spectral ratios...")

eps = 1e-8

df["ratio_900_550"] = (
    df["reflectance_902nm"] /
    (df["reflectance_552nm"] + eps)
)

df["ratio_1000_900"] = (
    df["reflectance_999nm"] /
    (df["reflectance_902nm"] + eps)
)

df["ratio_1200_1000"] = (
    df["reflectance_1201nm"] /
    (df["reflectance_999nm"] + eps)
)

df["ratio_1600_1200"] = (
    df["reflectance_1603nm"] /
    (df["reflectance_1201nm"] + eps)
)

df["ratio_2200_2000"] = (
    df["reflectance_2197nm"] /
    (df["reflectance_1997nm"] + eps)
)

df["ratio_2300_2200"] = (
    df["reflectance_2301nm"] /
    (df["reflectance_2197nm"] + eps)
)

df["ratio_2400_2300"] = (
    df["reflectance_2397nm"] /
    (df["reflectance_2301nm"] + eps)
)


# ============================================================
# NORMALIZED DIFFERENCE FEATURES
# ============================================================

print("Creating normalized spectral differences...")

df["nd_900_550"] = (
    (df["reflectance_902nm"] - df["reflectance_552nm"]) /
    (
        df["reflectance_902nm"] +
        df["reflectance_552nm"] +
        eps
    )
)

df["nd_1600_1200"] = (
    (df["reflectance_1603nm"] - df["reflectance_1201nm"]) /
    (
        df["reflectance_1603nm"] +
        df["reflectance_1201nm"] +
        eps
    )
)

df["nd_2200_2000"] = (
    (df["reflectance_2197nm"] - df["reflectance_1997nm"]) /
    (
        df["reflectance_2197nm"] +
        df["reflectance_1997nm"] +
        eps
    )
)


# ============================================================
# SPECTRAL DIFFERENCES
# ============================================================

print("Creating spectral differences...")

df["diff_700_650"] = (
    df["reflectance_701nm"] -
    df["reflectance_649nm"]
)

df["diff_1000_900"] = (
    df["reflectance_999nm"] -
    df["reflectance_902nm"]
)

df["diff_2200_2000"] = (
    df["reflectance_2197nm"] -
    df["reflectance_1997nm"]
)


# ============================================================
# ENGINEERED FEATURE LIST
# ============================================================

engineered_features = [
    "ratio_900_550",
    "ratio_1000_900",
    "ratio_1200_1000",
    "ratio_1600_1200",
    "ratio_2200_2000",
    "ratio_2300_2200",
    "ratio_2400_2300",
    "nd_900_550",
    "nd_1600_1200",
    "nd_2200_2000",
    "diff_700_650",
    "diff_1000_900",
    "diff_2200_2000"
]


# ============================================================
# CLEAN ENGINEERED FEATURES
# ============================================================

print("\nCleaning engineered features...")

df[engineered_features] = df[
    engineered_features
].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=engineered_features,
    how="any"
).copy()

print(
    f"Rows after feature cleaning: {len(df):,}"
)


# ============================================================
# REMOVE EXTREME NUMERICAL VALUES
# ============================================================

print("\nChecking numerical feature ranges...")

feature_columns = (
    spectral_columns +
    engineered_features +
    ["elevation_m"]
)

feature_columns = [
    col for col in feature_columns
    if col in df.columns
]

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=feature_columns,
    how="any"
).copy()


# ============================================================
# SAVE FULL ENGINEERED DATASET
# ============================================================

print("\nSaving engineered dataset...")

df.to_csv(
    engineered_file,
    index=False
)

print(
    f"Saved: {engineered_file}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nOriginal class distribution:")

class_counts = df["lithologic"].value_counts()

print(class_counts)


# ============================================================
# BALANCED DATASET
# ============================================================

print("\nCreating balanced dataset...")

MAX_PER_CLASS = 20000

balanced_parts = []

for lithology, group in df.groupby(
    "lithologic",
    sort=False
):

    if len(group) > MAX_PER_CLASS:

        group = group.sample(
            n=MAX_PER_CLASS,
            random_state=42
        )

    balanced_parts.append(group)


if not balanced_parts:
    raise ValueError(
        "No classes available for balancing."
    )


balanced = pd.concat(
    balanced_parts,
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

balanced = balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# RESET NODE IDs
# ============================================================

balanced["node_id"] = np.arange(
    len(balanced)
)


# ============================================================
# SAVE BALANCED DATASET
# ============================================================

balanced.to_csv(
    balanced_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"\nFull engineered dataset:"
)

print(engineered_file)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"\nBalanced GNN dataset:"
)

print(balanced_file)

print(
    f"Rows: {len(balanced):,}"
)

print(
    f"Columns: {len(balanced.columns)}"
)

print("\nBalanced class distribution:")

print(
    balanced["lithologic"]
    .value_counts()
)

print("\nNew spectral features:")

for feature in engineered_features:
    print(f"  - {feature}")

print("\nDone.")