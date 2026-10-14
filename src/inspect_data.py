import os
import pandas as pd
import torch


# ============================================================
# PROJECT DATA INSPECTION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTERIM = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")


def show_file(path):
    exists = os.path.exists(path)

    print(f"\n{path}")
    print(f"Exists: {exists}")

    if exists:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")

    return exists


# ============================================================
# 1. CHECK EXPECTED FILES
# ============================================================

print("=" * 70)
print("PROJECT DATA INSPECTION")
print("=" * 70)

files = [
    os.path.join(INTERIM, "emit_features.csv"),
    os.path.join(INTERIM, "emit_geology.csv"),
    os.path.join(INTERIM, "emit_engineered.csv"),
    os.path.join(PROCESSED, "gnn_dataset.csv"),
    os.path.join(PROCESSED, "mineral_graph.pt"),
    os.path.join(OUTPUTS, "gnn_model.pt"),
]

print("\nFILE CHECK")
print("-" * 70)

for file in files:
    show_file(file)


# ============================================================
# 2. INSPECT GNN DATASET
# ============================================================

dataset_file = os.path.join(PROCESSED, "gnn_dataset.csv")

if os.path.exists(dataset_file):

    print("\n" + "=" * 70)
    print("GNN DATASET")
    print("=" * 70)

    df = pd.read_csv(dataset_file)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nCoordinate range:")

    print(
        f"Latitude : "
        f"{df['latitude'].min():.6f} "
        f"to "
        f"{df['latitude'].max():.6f}"
    )

    print(
        f"Longitude: "
        f"{df['longitude'].min():.6f} "
        f"to "
        f"{df['longitude'].max():.6f}"
    )

    print(
        f"Elevation: "
        f"{df['elevation_m'].min():.2f} "
        f"to "
        f"{df['elevation_m'].max():.2f} m"
    )

    print("\nLithology classes:")
    print(df["lithologic"].value_counts())


# ============================================================
# 3. INSPECT GRAPH
# ============================================================

graph_file = os.path.join(PROCESSED, "mineral_graph.pt")

if os.path.exists(graph_file):

    print("\n" + "=" * 70)
    print("GRAPH")
    print("=" * 70)

    graph = torch.load(
        graph_file,
        map_location="cpu",
        weights_only=False
    )

    print(f"\nNodes       : {graph.x.shape[0]}")
    print(f"Features    : {graph.x.shape[1]}")
    print(f"Edges       : {graph.edge_index.shape[1]}")
    print(f"Classes     : {len(graph.class_names)}")

    print("\nClass names:")
    for i, name in enumerate(graph.class_names):
        print(f"  {i}: {name}")

    print("\nMasks:")
    print(f"Train nodes : {graph.train_mask.sum().item()}")
    print(f"Val nodes   : {graph.val_mask.sum().item()}")
    print(f"Test nodes  : {graph.test_mask.sum().item()}")


# ============================================================
# 4. INSPECT TRAINED MODEL
# ============================================================

model_file = os.path.join(OUTPUTS, "gnn_model.pt")

if os.path.exists(model_file):

    print("\n" + "=" * 70)
    print("TRAINED MODEL")
    print("=" * 70)

    model = torch.load(
        model_file,
        map_location="cpu",
        weights_only=False
    )

    print("\nModel file loaded successfully.")

    if isinstance(model, dict):
        print("Saved object type: dictionary")
        print("Keys:")
        for key in model.keys():
            print(f"  {key}")
    else:
        print(f"Saved object type: {type(model)}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)