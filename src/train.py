#!/usr/bin/env python3
"""
train.py
========
Training loop for the MultimodalGAT on the HAM10000 patient graph.

Handles
-------
- Automatic device selection (MPS / CUDA / CPU).
- Weighted Cross-Entropy loss to counter severe class imbalance
  (the 'nv' class dominates HAM10000).
- Full-graph training when the graph fits in memory; falls back to
  PyG NeighborLoader mini-batching otherwise.
- Accuracy, macro-F1, and per-class metrics on the validation set.
- Saves best model checkpoint based on validation macro-F1.

Regularisation techniques (v2)
------------------------------
- **L2 Regularization** via optimizer weight_decay.
- **DropEdge**: Randomly drops edges during training to prevent
  over-smoothing in GNNs.
- **Feature Masking**: Randomly zeroes out input feature dimensions
  during training for data augmentation on the graph side.
- **Early Stopping**: Monitors validation macro-F1 with configurable
  patience.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import dropout_edge

from src.data_prep import build_graph, LABEL_MAP
from src.model import MultimodalGAT


# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Inverse label map for reporting
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ── Class weights ──────────────────────────────────────────

def compute_class_weights(y: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    """
    Compute inverse-frequency weights so that rare classes
    (melanoma, dermatofibroma, vascular) are penalised more
    during training.
    """
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalise
    return weights


# ── GNN Data Augmentation ─────────────────────────────────

def apply_feature_masking(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
    """
    Randomly zero out `mask_ratio` fraction of feature dimensions
    for each node during training. Acts as feature-level dropout /
    data augmentation for the GNN.
    """
    mask = torch.bernoulli(
        torch.full(x.shape, 1.0 - mask_ratio, device=x.device)
    )
    return x * mask


# ── Training & Evaluation ─────────────────────────────────

def train_epoch(model, data, optimizer, criterion,
                drop_edge_rate: float = 0.1,
                feature_mask_rate: float = 0.1):
    """
    One full-graph training step with DropEdge and feature masking.
    """
    model.train()
    optimizer.zero_grad()

    # ── GNN-specific augmentation ─────────────────────────
    # DropEdge: randomly remove edges to combat over-smoothing
    edge_index_aug, _ = dropout_edge(
        data.edge_index, p=drop_edge_rate, training=True
    )

    # Feature masking: randomly zero-out feature dimensions
    x_aug = apply_feature_masking(data.x, mask_ratio=feature_mask_rate)

    logits = model(x_aug, edge_index_aug)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    preds = logits[data.train_mask].argmax(dim=1).cpu()
    acc = accuracy_score(data.y[data.train_mask].cpu(), preds)
    f1 = f1_score(data.y[data.train_mask].cpu(), preds, average="macro", zero_division=0)
    return loss.item(), acc, f1


def train_epoch_batched(model, loader, optimizer, criterion, device,
                        drop_edge_rate: float = 0.1,
                        feature_mask_rate: float = 0.1):
    """Mini-batch training via NeighborLoader with augmentation."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # DropEdge on batch
        edge_index_aug, _ = dropout_edge(
            batch.edge_index, p=drop_edge_rate, training=True
        )

        # Feature masking
        x_aug = apply_feature_masking(batch.x, mask_ratio=feature_mask_rate)

        logits = model(x_aug, edge_index_aug)

        # Only compute loss on seed nodes (train_mask within the batch)
        mask = batch.train_mask[:batch.batch_size]
        loss = criterion(logits[:batch.batch_size][mask],
                         batch.y[:batch.batch_size][mask])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits[:batch.batch_size][mask].argmax(dim=1)
        total_correct += (preds == batch.y[:batch.batch_size][mask]).sum().item()
        total_nodes += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / max(total_nodes, 1)
    avg_acc = total_correct / max(total_nodes, 1)
    return avg_loss, avg_acc, 0.0  # f1 not computed per-batch for speed


@torch.no_grad()
def evaluate(model, data, mask, criterion):
    """Evaluate on the given mask (val or test)."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[mask], data.y[mask]).item()
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = data.y[mask].cpu().numpy()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return loss, acc, f1, preds, targets


# ── Main Training Loop ────────────────────────────────────

def train(
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    hidden: int = 256,
    heads: int = 4,
    dropout: float = 0.4,
    patience: int = 20,
    use_batching: bool = False,
    batch_size: int = 512,
    drop_edge_rate: float = 0.1,
    feature_mask_rate: float = 0.1,
):
    """
    Full training pipeline with L2 regularization (weight_decay),
    DropEdge, feature masking, and macro-F1-based early stopping.
    """
    print("=" * 60)
    print("  GAT Training Pipeline (v2 — with Regularisation)")
    print("=" * 60)

    # ── Load graph ─────────────────────────────────────────
    data = build_graph()
    data = data.to(DEVICE)

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    print(f"\n[CONFIG]")
    print(f"  Device            : {DEVICE}")
    print(f"  Nodes             : {data.num_nodes}")
    print(f"  Edges             : {data.num_edges}")
    print(f"  Node features     : {num_features}")
    print(f"  Classes           : {num_classes}")
    print(f"  Epochs            : {epochs}")
    print(f"  Learning rate     : {lr}")
    print(f"  Weight decay (L2) : {weight_decay}")
    print(f"  Dropout           : {dropout}")
    print(f"  DropEdge rate     : {drop_edge_rate}")
    print(f"  Feature mask rate : {feature_mask_rate}")
    print(f"  Patience          : {patience}")
    print(f"  Batched training  : {use_batching}")
    print()

    # ── Model ──────────────────────────────────────────────
    model = MultimodalGAT(
        visual_channels=2048,
        meta_channels=num_features - 2048,
        hidden_channels=hidden,
        out_channels=num_classes,
        heads=heads,
        dropout=dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] MultimodalGAT (v2) — {total_params:,} parameters\n")

    # ── Loss (class-weighted) ──────────────────────────────
    class_weights = compute_class_weights(data.y.cpu(), num_classes).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"[LOSS] Class weights: {class_weights.cpu().numpy().round(3)}\n")

    # ── Optimizer + Scheduler ──────────────────────────────
    # L2 Regularization is applied via weight_decay in AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Optional mini-batch loader ─────────────────────────
    loader = None
    if use_batching:
        loader = NeighborLoader(
            data,
            num_neighbors=[10, 10, 5],
            batch_size=batch_size,
            input_nodes=data.train_mask,
            shuffle=True,
        )

    # ── Training loop ──────────────────────────────────────
    best_val_f1 = 0.0
    best_val_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if use_batching and loader is not None:
            train_loss, train_acc, train_f1 = train_epoch_batched(
                model, loader, optimizer, criterion, DEVICE,
                drop_edge_rate=drop_edge_rate,
                feature_mask_rate=feature_mask_rate,
            )
        else:
            train_loss, train_acc, train_f1 = train_epoch(
                model, data, optimizer, criterion,
                drop_edge_rate=drop_edge_rate,
                feature_mask_rate=feature_mask_rate,
            )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, data, data.val_mask, criterion
        )
        scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = ""
        # Monitor macro-F1 for early stopping (better metric for imbalanced data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_gat.pt")
            improved = " ★"
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.4f}  │  "
            f"ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}  "
            f"ValF1={val_f1:.4f}  LR={current_lr:.2e}  "
            f"({elapsed:.1f}s){improved}"
        )

        if epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping at epoch {epoch} "
                  f"(no improvement in macro-F1 for {patience} epochs).")
            break

    # ── Final Test Evaluation ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "best_gat.pt", map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc, test_f1, test_preds, test_targets = evaluate(
        model, data, data.test_mask, criterion
    )

    target_names = [IDX_TO_LABEL[i] for i in range(num_classes)]
    report = classification_report(
        test_targets, test_preds, target_names=target_names, zero_division=0
    )

    print(f"\n  Test Accuracy : {test_acc:.4f}")
    print(f"  Test Macro-F1 : {test_f1:.4f}")
    print(f"  Best Val F1   : {best_val_f1:.4f}")
    print(f"\n{report}")

    # ── Save history ───────────────────────────────────────
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training history saved → {history_path}")

    return model, data, history


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultimodalGAT (v2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="L2 regularization strength")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batched", action="store_true",
                        help="Use NeighborLoader mini-batching")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drop-edge-rate", type=float, default=0.1,
                        help="DropEdge rate during training")
    parser.add_argument("--feature-mask-rate", type=float, default=0.1,
                        help="Feature masking rate during training")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        patience=args.patience,
        use_batching=args.batched,
        batch_size=args.batch_size,
        drop_edge_rate=args.drop_edge_rate,
        feature_mask_rate=args.feature_mask_rate,
    )
