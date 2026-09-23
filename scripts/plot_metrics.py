#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MultimodalGAT
from src.data_prep import build_graph, LABEL_MAP

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
MODEL_PATH = CHECKPOINT_DIR / "best_gat.pt"
OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

IDX_TO_LABEL = {v: k.upper() for k, v in LABEL_MAP.items()}

def plot_history():
    print("[INFO] Plotting training history...")
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    
    epochs = range(1, len(history["train_acc"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history["train_acc"], label="Train Accuracy", marker='o', markersize=4)
    ax1.plot(epochs, history["val_acc"], label="Validation Accuracy", marker='o', markersize=4)
    ax1.set_title("Model Accuracy over Epochs", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Loss plot
    ax2.plot(epochs, history["train_loss"], label="Train Loss", marker='o', markersize=4, color='red')
    ax2.plot(epochs, history["val_loss"], label="Validation Loss", marker='o', markersize=4, color='orange')
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

def plot_per_class_accuracy():
    print("[INFO] Evaluating model on test set for per-class accuracy...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data and model
    data = build_graph().to(device)
    num_features = data.num_node_features
    num_classes = len(LABEL_MAP)
    
    model = MultimodalGAT(
        in_channels=num_features,
        hidden_channels=256,
        out_channels=num_classes,
        heads=4,
        dropout=0.3,
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        targets = data.y[data.test_mask].cpu().numpy()
        
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    # Calculate accuracy per class: True Positives / Total true instances for that class
    class_totals = cm.sum(axis=1)
    # Avoid division by zero
    class_totals_safe = np.where(class_totals == 0, 1, class_totals)
    per_class_acc = (cm.diagonal() / class_totals_safe) * 100
    
    labels = [IDX_TO_LABEL[i] for i in range(num_classes)]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, per_class_acc, color='#4CAF50', edgecolor='black')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    ax.set_title("Test Accuracy per Skin Cancer Class", fontsize=15, pad=20)
    ax.set_xlabel("Lesion Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105) # Leave room for labels
    
    # Add counts below x-axis labels to give context on dataset imbalance
    new_labels = [f"{label}\n(n={count})" for label, count in zip(labels, class_totals)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(new_labels)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "per_class_accuracy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")

if __name__ == "__main__":
    plot_history()
    plot_per_class_accuracy()
    print("[DONE] Visualizations generated.")
