import torch
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
GRAPH_PATH = "data/processed/mineral_graph.pt"
MODEL_PATH = "outputs/gnn_model.pt"
OUTPUT_PATH = "outputs/gcn_prediction_map.png"


# ---------------------------------------------------------
# LOAD GRAPH
# ---------------------------------------------------------
data = torch.load(
    GRAPH_PATH,
    map_location="cpu",
    weights_only=False
)

print("Graph loaded")
print("Nodes:", data.num_nodes)
print("Features:", data.num_features)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
class GCN(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)

        return x


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model = GCN(
    checkpoint["num_features"],
    checkpoint["num_classes"]
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
with torch.no_grad():

    logits = model(
        data.x,
        data.edge_index
    )

    predictions = logits.argmax(dim=1)


# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------
class_names = checkpoint["class_names"]

print("\nPrediction distribution:")

for i, name in enumerate(class_names):

    count = int((predictions == i).sum())

    print(f"{name}: {count}")


# ---------------------------------------------------------
# COORDINATES
# ---------------------------------------------------------
latitude = data.latitude.numpy()
longitude = data.longitude.numpy()
predictions = predictions.numpy()


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------
plt.figure(figsize=(12, 9))

scatter = plt.scatter(
    longitude,
    latitude,
    c=predictions,
    cmap="tab10",
    s=5,
    alpha=0.8
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "GCN-Based Lithology Prediction Map\n"
    "EMIT Hyperspectral Data"
)

# Colorbar
cbar = plt.colorbar(scatter)

cbar.set_ticks(range(len(class_names)))
cbar.set_ticklabels(class_names)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n====================================")
print("PREDICTION MAP GENERATED")
print("====================================")
print(f"Saved to: {OUTPUT_PATH}")