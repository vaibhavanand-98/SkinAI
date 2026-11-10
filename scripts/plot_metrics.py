#!/usr/bin/env python3
"""
plot_metrics.py
===============
Generate all training visualisations:
1. Accuracy + Loss history curves
2. Per-class accuracy bar chart
3. Confusion matrix heatmap
4. Separate loss curve
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import sys
from pathlib import Path

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MultimodalGAT
from src.data_prep import build_graph, LABEL_MAP, LABEL_NAMES, VISUAL_DIM

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
MODEL_PATH = CHECKPOINT_DIR / "best_gat.pt"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

IDX_TO_LABEL = {v: k.upper() for k, v in LABEL_MAP.items()}
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def plot_history():
    """Plot training + validation accuracy and loss curves."""
    print("[INFO] Plotting training history...")
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs, history["train_acc"], label="Train Accuracy", marker='o', markersize=3)
    ax1.plot(epochs, history["val_acc"], label="Validation Accuracy", marker='o', markersize=3)
    if "val_f1" in history:
        ax1.plot(epochs, history["val_f1"], label="Validation Macro-F1",
                 marker='s', markersize=3, linestyle='--')
    ax1.set_title("Model Accuracy & Macro-F1 over Epochs", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Loss plot
    ax2.plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=3, color='red')
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", marker='o', markersize=3, color='orange')
    ax2.set_title("Model Loss over Epochs", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "accuracy_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_loss_curve():
    """Plot a separate, detailed loss curve."""
    print("[INFO] Plotting loss curve...")
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            linewidth=2, color='#e74c3c')
    ax.plot(epochs, history["val_loss"], label="Validation Loss",
            linewidth=2, color='#f39c12')
    ax.fill_between(epochs, history["train_loss"], alpha=0.1, color='#e74c3c')
    ax.fill_between(epochs, history["val_loss"], alpha=0.1, color='#f39c12')
    ax.set_title("Training Loss Curve (ViT+GAT)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Focal Loss", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "training_loss_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def _load_model_and_data():
    """Load the trained model and graph data."""
    data = build_graph().to(DEVICE)
    num_features = data.num_node_features
    num_classes = len(LABEL_MAP)

    model = MultimodalGAT(
        visual_channels=VISUAL_DIM,
        meta_channels=num_features - VISUAL_DIM,
        hidden_channels=256,
        out_channels=num_classes,
        heads=4,
        dropout=0.3,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, data, num_classes


def plot_per_class_accuracy():
    """Plot per-class accuracy on the test set."""
    print("[INFO] Evaluating model on test set for per-class accuracy...")
    model, data, num_classes = _load_model_and_data()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    class_totals = cm.sum(axis=1)
    class_totals_safe = np.where(class_totals == 0, 1, class_totals)
    per_class_acc = (cm.diagonal() / class_totals_safe) * 100

    labels = [IDX_TO_LABEL[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if l in ('MEL', 'BCC', 'AKIEC') else '#2ecc71'
              for l in labels]
    bars = ax.bar(labels, per_class_acc, color=colors, edgecolor='black')

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1,
                f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_title("Test Accuracy per Skin Cancer Class (ViT+GAT)", fontsize=15, pad=20)
    ax.set_xlabel("Lesion Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)

    new_labels = [f"{label}\n(n={count})" for label, count in zip(labels, class_totals)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(new_labels)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#e74c3c', label='Malignant / Pre-malignant'),
        Patch(facecolor='#2ecc71', label='Benign'),
    ], loc='lower right')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "per_class_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_confusion_matrix():
    """Plot and save a confusion matrix heatmap."""
    print("[INFO] Generating confusion matrix...")
    model, data, num_classes = _load_model_and_data()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    labels = [IDX_TO_LABEL[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — ViT+GAT on Test Set", fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

    # Also print classification report
    report = classification_report(targets, preds, target_names=labels, zero_division=0)
    print(f"\n{report}")


if __name__ == "__main__":
    plot_history()
    plot_loss_curve()
    plot_per_class_accuracy()
    plot_confusion_matrix()
    print("[DONE] All visualizations generated.")
