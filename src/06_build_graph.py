import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


# ============================================================
# PATHS
# ============================================================

input_file = Path(
    "data/processed/gnn_dataset.csv"
)

output_file = Path(
    "data/processed/mineral_graph.pt"
)


print("=" * 70)
print("GNN GRAPH CONSTRUCTION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading balanced dataset...")

df = pd.read_csv(input_file)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# SAMPLE DATA
# ============================================================

MAX_PER_CLASS = 5000

print(
    f"\nSampling maximum {MAX_PER_CLASS:,} nodes per lithology..."
)

parts = []

for lithology, group in df.groupby(
    "lithologic",
    sort=False
):

    n = min(
        len(group),
        MAX_PER_CLASS
    )

    sampled = group.sample(
        n=n,
        random_state=42
    )

    parts.append(sampled)


df = pd.concat(
    parts,
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print(f"Graph nodes: {len(df):,}")


# ============================================================
# LABEL ENCODING
# ============================================================

print("\nEncoding lithology labels...")

classes = sorted(
    df["lithologic"].unique()
)

class_to_id = {
    name: i
    for i, name in enumerate(classes)
}

df["label"] = df["lithologic"].map(
    class_to_id
)

print("\nClasses:")

for name, idx in class_to_id.items():
    print(f"  {idx}: {name}")


# ============================================================
# NODE FEATURES
# ============================================================

print("\nPreparing node features...")

feature_columns = [
    "elevation_m",

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
    "reflectance_2397nm",

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


missing = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing feature columns: {missing}"
    )


X = df[feature_columns].values.astype(
    np.float32
)


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

print(
    f"\nNumber of node features: {X.shape[1]}"
)

print("Standardizing features...")

scaler = StandardScaler()

X = scaler.fit_transform(X).astype(
    np.float32
)


# ============================================================
# COORDINATES
# ============================================================

print("\nPreparing spatial coordinates...")

coordinates = df[
    ["longitude", "latitude"]
].values.astype(
    np.float32
)


# ============================================================
# KNN GRAPH
# ============================================================

print("\nBuilding spatial KNN graph...")

K = 8

knn = NearestNeighbors(
    n_neighbors=K + 1,
    metric="euclidean",
    algorithm="ball_tree"
)

knn.fit(coordinates)

distances, neighbors = knn.kneighbors(
    coordinates
)


# Remove self-neighbor
neighbors = neighbors[:, 1:]
distances = distances[:, 1:]


# ============================================================
# CREATE EDGES
# ============================================================

print("Creating graph edges...")

source = np.repeat(
    np.arange(len(df)),
    K
)

target = neighbors.reshape(-1)

edge_index = np.vstack(
    [source, target]
)

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long
)


# ============================================================
# EDGE WEIGHTS
# ============================================================

edge_distance = distances.reshape(-1)

edge_weight = torch.tensor(
    1.0 / (edge_distance + 1e-8),
    dtype=torch.float32
)


# ============================================================
# LABELS
# ============================================================

y = torch.tensor(
    df["label"].values,
    dtype=torch.long
)


# ============================================================
# TRAIN / VALIDATION / TEST MASKS
# ============================================================

print("\nCreating train/validation/test masks...")

indices = np.arange(
    len(df)
)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=df.iloc[temp_idx]["label"]
)


train_mask = torch.zeros(
    len(df),
    dtype=torch.bool
)

val_mask = torch.zeros(
    len(df),
    dtype=torch.bool
)

test_mask = torch.zeros(
    len(df),
    dtype=torch.bool
)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True


# ============================================================
# CREATE PYTORCH GEOMETRIC DATA
# ============================================================

print("\nCreating PyTorch Geometric graph...")

data = Data(
    x=torch.tensor(
        X,
        dtype=torch.float32
    ),
    edge_index=edge_index,
    edge_weight=edge_weight,
    y=y
)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# ============================================================
# METADATA
# ============================================================

data.num_classes = len(classes)
data.num_features = X.shape[1]

data.class_names = classes

data.latitude = torch.tensor(
    df["latitude"].values,
    dtype=torch.float32
)

data.longitude = torch.tensor(
    df["longitude"].values,
    dtype=torch.float32
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving graph...")

torch.save(
    data,
    output_file
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAPH CONSTRUCTION COMPLETE")
print("=" * 70)

print(f"\nGraph file:")
print(output_file)

print(f"\nNodes: {data.num_nodes:,}")

print(
    f"Edges: {data.num_edges:,}"
)

print(
    f"Node features: {data.num_features}"
)

print(
    f"Classes: {data.num_classes}"
)

print(
    f"Training nodes: {train_mask.sum().item():,}"
)

print(
    f"Validation nodes: {val_mask.sum().item():,}"
)

print(
    f"Test nodes: {test_mask.sum().item():,}"
)

print("\nClass mapping:")

for idx, name in enumerate(classes):
    print(f"  {idx}: {name}")

print("\nGraph object:")

print(data)

print("\nDone.")