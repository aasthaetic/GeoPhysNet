import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

graph_file = Path("data/processed/mineral_graph.pt")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "gnn_graph_visualization.png"


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("GNN GRAPH VISUALIZATION")
print("=" * 70)

print("\nLoading graph...")

data = torch.load(
    graph_file,
    weights_only=False
)

print(data)


# ============================================================
# SAMPLE NODES
# ============================================================

MAX_NODES = 600

num_nodes = data.num_nodes

np.random.seed(42)

if num_nodes > MAX_NODES:
    selected = np.random.choice(
        num_nodes,
        MAX_NODES,
        replace=False
    )
else:
    selected = np.arange(num_nodes)

selected = np.sort(selected)

selected_set = set(selected)


# ============================================================
# NODE COORDINATES
# ============================================================

longitude = data.longitude.numpy()
latitude = data.latitude.numpy()

x = longitude[selected]
y = latitude[selected]

labels = data.y.numpy()[selected]


# ============================================================
# BUILD LOCAL EDGES
# ============================================================

edge_index = data.edge_index.numpy()

local_edges = []

for source, target in zip(
    edge_index[0],
    edge_index[1]
):

    if source in selected_set and target in selected_set:

        local_edges.append(
            (
                source,
                target
            )
        )


# Map original node IDs to plot positions
position = {
    node_id: i
    for i, node_id in enumerate(selected)
}


# ============================================================
# PLOT
# ============================================================

print(
    f"\nVisualizing {len(selected):,} nodes..."
)

print(
    f"Local edges: {len(local_edges):,}"
)

plt.figure(
    figsize=(14, 10)
)


# ------------------------------------------------------------
# Draw edges
# ------------------------------------------------------------

for source, target in local_edges:

    i = position[source]
    j = position[target]

    plt.plot(
        [x[i], x[j]],
        [y[i], y[j]],
        linewidth=0.25,
        alpha=0.25
    )


# ------------------------------------------------------------
# Draw nodes
# ------------------------------------------------------------

scatter = plt.scatter(
    x,
    y,
    c=labels,
    s=12,
    alpha=0.85
)


# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "Longitude",
    fontsize=12
)

plt.ylabel(
    "Latitude",
    fontsize=12
)

plt.title(
    "EMIT Spatial Graph for Geological Mineral Discovery",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    alpha=0.2
)


# ============================================================
# COLORBAR
# ============================================================

colorbar = plt.colorbar(
    scatter
)

colorbar.set_label(
    "Lithology Class",
    fontsize=11
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nSaved to:")

print(output_file)

print("\nDone.")