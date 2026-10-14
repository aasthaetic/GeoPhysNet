import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPH_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "mineral_graph.pt"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "gnn_model.pt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

CONFUSION_FILE = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)


# ============================================================
# LOAD GRAPH
# ============================================================

print("=" * 70)
print("GNN MODEL EVALUATION")
print("=" * 70)

graph = torch.load(
    GRAPH_FILE,
    map_location="cpu",
    weights_only=False
)

x = graph.x
edge_index = graph.edge_index
y = graph.y

test_mask = graph.test_mask

num_features = x.shape[1]
num_classes = len(graph.class_names)


print("\nGraph:")
print(f"Nodes       : {x.shape[0]}")
print(f"Features    : {num_features}")
print(f"Edges       : {edge_index.shape[1]}")
print(f"Classes     : {num_classes}")
print(f"Test nodes  : {test_mask.sum().item()}")


# ============================================================
# MODEL DEFINITION
# ============================================================

class GCN(torch.nn.Module):

    def __init__(self, input_features, num_classes):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            64
        )

        self.conv2 = GCNConv(
            64,
            num_classes
        )

        self.dropout = 0.30

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

model = GCN(
    num_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

with torch.no_grad():

    logits = model(
        x,
        edge_index
    )

    predictions = logits.argmax(
        dim=1
    )


true_labels = y[test_mask].cpu().numpy()

predicted_labels = predictions[test_mask].cpu().numpy()


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall   : {recall:.4f}")
print(f"Weighted F1       : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=graph.class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(10, 8))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("GCN Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(
    num_classes
)

plt.xticks(
    tick_marks,
    graph.class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    graph.class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    CONFUSION_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved to:\n{CONFUSION_FILE}"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)