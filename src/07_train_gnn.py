import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.nn import GCNConv


# ============================================================
# PATHS
# ============================================================

GRAPH_FILE = Path("data/processed/mineral_graph.pt")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = OUTPUT_DIR / "gnn_model.pt"
CURVE_FILE = OUTPUT_DIR / "training_curve.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 70)
print("GNN TRAINING")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading graph...")

data = torch.load(
    GRAPH_FILE,
    weights_only=False
)

data = data.to(device)

print(data)

print("\nNodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_features)
print("Classes:", data.num_classes)


# ============================================================
# MODEL
# ============================================================

class MineralGCN(torch.nn.Module):

    def __init__(
        self,
        input_features,
        hidden_features,
        num_classes
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features
        )

        self.conv2 = GCNConv(
            hidden_features,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.30,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# CREATE MODEL
# ============================================================

model = MineralGCN(
    input_features=data.num_features,
    hidden_features=64,
    num_classes=data.num_classes
).to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 30

train_losses = []
train_accuracies = []
val_accuracies = []


print("\nStarting training...")
print("-" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # Training accuracy
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        predictions = output.argmax(
            dim=1
        )

        train_correct = (
            predictions[data.train_mask]
            ==
            data.y[data.train_mask]
        ).sum().item()

        train_total = (
            data.train_mask.sum().item()
        )

        train_accuracy = (
            train_correct /
            train_total
        )


        # ----------------------------------------------------
        # Validation accuracy
        # ----------------------------------------------------

        val_correct = (
            predictions[data.val_mask]
            ==
            data.y[data.val_mask]
        ).sum().item()

        val_total = (
            data.val_mask.sum().item()
        )

        val_accuracy = (
            val_correct /
            val_total
        )


    train_losses.append(
        loss.item()
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {loss.item():.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    predictions = output.argmax(
        dim=1
    )

    test_correct = (
        predictions[data.test_mask]
        ==
        data.y[data.test_mask]
    ).sum().item()

    test_total = (
        data.test_mask.sum().item()
    )

    test_accuracy = (
        test_correct /
        test_total
    )


print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_features": data.num_features,
        "num_classes": data.num_classes,
        "class_names": data.class_names
    },
    MODEL_FILE
)

print(
    "Saved:",
    MODEL_FILE
)


# ============================================================
# TRAINING CURVE
# ============================================================

print("\nCreating training curve...")

epochs = range(
    1,
    EPOCHS + 1
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "GNN Training Loss"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CURVE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("GNN TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nModel:",
    MODEL_FILE
)

print(
    "Training curve:",
    CURVE_FILE
)

print("\nDone.")