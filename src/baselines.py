#!/usr/bin/env python3
"""
baselines.py
============
Trains and evaluates 3 baseline classifiers for comparison against the
main ViT+GAT model on the HAM10000 skin-cancer classification task.

Baselines (matching Table 3 in the report)
------------------------------------------
1. **CNN Baseline (ResNet-50 features → MLP)**
   Uses only the 2048-d ResNet-50 visual embeddings (the first
   ``VISUAL_DIM`` columns of ``data.x``) with a 2-layer MLP.

2. **CNN + Metadata (no graph)**
   Uses the full multimodal feature vector ``data.x``
   (2048-d visual + D_meta clinical metadata) with a 2-layer MLP.

3. **CNN + Metadata + kNN Graph (GAT)**
   This is the main model trained in ``train.py``; its results are
   loaded from the saved checkpoint for side-by-side comparison.

Each MLP baseline uses:
- FocalLoss with inverse-frequency class weights (same as GAT training).
- AdamW optimiser with ``weight_decay=1e-3``.
- Cosine-annealing LR schedule.
- Early stopping on validation macro-F1 (patience=15).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_prep import LABEL_MAP, VISUAL_DIM, build_graph

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ─────────────────────────────────────────────────
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Inverse label map for human-readable reporting
IDX_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES: int = len(LABEL_MAP)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Focal Loss (same formulation as train.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting for imbalanced data."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Class-weight computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_class_weights(
    y: torch.Tensor, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
    """Inverse-frequency weights normalised to sum to ``num_classes``."""
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MLP Classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MLPClassifier(nn.Module):
    """Simple 2-layer MLP for flat (non-graph) classification.

    Architecture
    ------------
    ``in_features → hidden → ReLU → Dropout → num_classes``

    Parameters
    ----------
    in_features:
        Dimensionality of the input feature vector.
    hidden:
        Width of the hidden layer.
    num_classes:
        Number of output classes.
    dropout:
        Dropout probability after the hidden layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 256,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[N, num_classes]``."""
        return self.net(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_baseline(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 15,
    checkpoint_name: str = "baseline",
    device: str = DEVICE,
) -> dict[str, list[float]]:
    """Generic training loop for MLP baselines.

    Parameters
    ----------
    model:
        An ``MLPClassifier`` (or compatible ``nn.Module``).
    features:
        Node feature matrix ``[N, D]`` (already on *device*).
    labels:
        Ground-truth class indices ``[N]`` (already on *device*).
    train_mask / val_mask:
        Boolean masks selecting training / validation nodes.
    epochs:
        Maximum number of training epochs.
    lr:
        Initial learning rate for AdamW.
    weight_decay:
        L2 regularisation coefficient.
    patience:
        Early-stopping patience (epochs without macro-F1 improvement).
    checkpoint_name:
        Filename stem for the best-model checkpoint.
    device:
        Torch device string.

    Returns
    -------
    history:
        Dict with keys ``train_loss``, ``train_acc``, ``val_loss``,
        ``val_acc``, ``val_f1`` — each a list of per-epoch values.
    """
    model = model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    # ── Loss ───────────────────────────────────────────────
    class_weights = compute_class_weights(labels.cpu()).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # ── Optimiser & scheduler ──────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Tracking ───────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }
    best_val_f1 = 0.0
    epochs_no_improve = 0
    ckpt_path = CHECKPOINT_DIR / f"best_{checkpoint_name}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_preds = logits[train_mask].argmax(dim=1).cpu().numpy()
        train_targets = labels[train_mask].cpu().numpy()
        train_acc = accuracy_score(train_targets, train_preds)

        # ── Validate ───────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_logits = model(features)
            val_loss = criterion(val_logits[val_mask], labels[val_mask]).item()
            val_preds = val_logits[val_mask].argmax(dim=1).cpu().numpy()
            val_targets = labels[val_mask].cpu().numpy()

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            improved = " ★"
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{epochs}  "
            f"TrainLoss={loss.item():.4f}  TrainAcc={train_acc:.4f}  │  "
            f"ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}  "
            f"ValF1={val_f1:.4f}  LR={current_lr:.2e}  "
            f"({elapsed:.1f}s){improved}"
        )

        if epochs_no_improve >= patience:
            print(
                f"  [Early stop] No macro-F1 improvement for "
                f"{patience} epochs — stopping at epoch {epoch}."
            )
            break

    # Restore best weights
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str = DEVICE,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained MLP baseline and return a metrics dict.

    Returned keys: ``accuracy``, ``precision``, ``recall``,
    ``macro_f1``, ``per_class_report`` (text), ``confusion_matrix``.
    """
    model = model.to(device).eval()
    features = features.to(device)
    labels = labels.to(device)
    mask = mask.to(device)

    logits = model(features)
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    targets = labels[mask].cpu().numpy()

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    rec = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        targets, preds, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(targets, preds)

    # Per-class recall (especially important for melanoma = class 4)
    per_class_recall = recall_score(
        targets, preds, average=None, zero_division=0
    )
    mel_idx = LABEL_MAP["mel"]
    mel_recall = float(per_class_recall[mel_idx])

    if verbose:
        print(f"\n  Test Accuracy  : {acc:.4f}")
        print(f"  Test Precision : {prec:.4f}  (macro)")
        print(f"  Test Recall    : {rec:.4f}  (macro)")
        print(f"  Test Macro-F1  : {f1:.4f}")
        print(f"  Melanoma Recall: {mel_recall:.4f}")
        print(f"\n{report}")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "melanoma_recall": round(mel_recall, 4),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Run All Baselines
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_all_baselines(epochs: int = 50) -> dict[str, dict[str, Any]]:
    """Train all 3 MLP baselines sequentially and print a comparison table.

    Saves results to ``checkpoints/baseline_results.json``.

    Returns
    -------
    results : dict
        ``{ "cnn_baseline": {…}, "cnn_meta": {…} }``
        Each inner dict contains ``accuracy``, ``precision``, ``recall``,
        ``macro_f1``, ``melanoma_recall``, and the full classification
        report / confusion matrix.
    """
    print("=" * 64)
    print("  Baseline Evaluation Pipeline")
    print("=" * 64)
    print(f"  Device : {DEVICE}")

    # ── Load graph ─────────────────────────────────────────
    data = build_graph()
    num_features = data.num_node_features
    meta_dim = num_features - VISUAL_DIM

    print(f"  Nodes  : {data.num_nodes}")
    print(f"  Visual : {VISUAL_DIM}-d   Metadata : {meta_dim}-d")
    print(f"  Classes: {NUM_CLASSES}")
    print()

    results: dict[str, dict[str, Any]] = {}

    # ──────────────────────────────────────────────────────
    #  Baseline 1: CNN (ResNet-50 visual features → MLP)
    # ──────────────────────────────────────────────────────
    print("─" * 64)
    print("  Baseline 1 / 4 : CNN (ResNet-50 visual features → MLP)")
    print("─" * 64)

    resnet_path = PROJECT_ROOT / "data" / "processed" / "visual_embeddings_resnet50.pt"
    if resnet_path.exists():
        print(f"  [INFO] Loading ResNet-50 embeddings from {resnet_path}")
        cnn_features = torch.load(resnet_path, map_location="cpu", weights_only=True)
        cnn_in_dim = cnn_features.shape[1]
        model_cnn = MLPClassifier(
            in_features=cnn_in_dim,
            hidden=256,
            num_classes=NUM_CLASSES,
            dropout=0.3,
        )
        print(f"  CNN Feature Dim: {cnn_in_dim}, Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}\n")

        train_baseline(
            model_cnn,
            cnn_features,
            data.y,
            data.train_mask,
            data.val_mask,
            epochs=epochs,
            checkpoint_name="cnn_baseline",
        )
        results["cnn_baseline"] = evaluate_baseline(
            model_cnn, cnn_features, data.y, data.test_mask
        )
    else:
        print("  [WARN] ResNet-50 embeddings not found. Skipping CNN baseline.")

    # ──────────────────────────────────────────────────────
    #  Baseline 2: ViT (ViT-B/16 visual features only → MLP)
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Baseline 2 / 4 : ViT (ViT-B/16 visual features only → MLP)")
    print("─" * 64)

    vit_features = data.x[:, :VISUAL_DIM]          # [N, 768]

    model_vit = MLPClassifier(
        in_features=VISUAL_DIM,
        hidden=256,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )
    print(f"  ViT Feature Dim: {VISUAL_DIM}, Parameters: {sum(p.numel() for p in model_vit.parameters()):,}\n")

    train_baseline(
        model_vit,
        vit_features,
        data.y,
        data.train_mask,
        data.val_mask,
        epochs=epochs,
        checkpoint_name="vit_only",
    )
    results["vit_only"] = evaluate_baseline(
        model_vit, vit_features, data.y, data.test_mask
    )

    # ──────────────────────────────────────────────────────
    #  Baseline 3: ViT + Metadata (full features → MLP)
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Baseline 3 / 4 : ViT + Metadata (full features → MLP)")
    print("─" * 64)

    full_features = data.x                             # [N, 768 + D_meta]

    model_vit_meta = MLPClassifier(
        in_features=num_features,
        hidden=256,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )
    print(f"  Parameters: {sum(p.numel() for p in model_vit_meta.parameters()):,}\n")

    train_baseline(
        model_vit_meta,
        full_features,
        data.y,
        data.train_mask,
        data.val_mask,
        epochs=epochs,
        checkpoint_name="vit_meta",
    )
    results["vit_meta"] = evaluate_baseline(
        model_vit_meta, full_features, data.y, data.test_mask
    )

    # ──────────────────────────────────────────────────────
    #  Baseline 4: Proposed ViT + GNN (GAT) — load from ckpt
    # ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Baseline 4 / 4 : Proposed ViT + GNN (GAT from checkpoint)")
    print("─" * 64)

    gat_ckpt = CHECKPOINT_DIR / "best_gat.pt"
    if gat_ckpt.exists():
        from src.model import MultimodalGAT

        gat_model = MultimodalGAT(
            visual_channels=VISUAL_DIM,
            meta_channels=meta_dim,
            hidden_channels=256,
            out_channels=NUM_CLASSES,
            heads=4,
            dropout=0.4,
        ).to(DEVICE)

        gat_model.load_state_dict(
            torch.load(gat_ckpt, map_location=DEVICE, weights_only=True)
        )
        gat_model.eval()

        gat_data = data.to(DEVICE)
        logits = gat_model(gat_data.x, gat_data.edge_index)
        test_preds = logits[gat_data.test_mask].argmax(dim=1).cpu().numpy()
        test_targets = gat_data.y[gat_data.test_mask].cpu().numpy()

        target_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
        acc = accuracy_score(test_targets, test_preds)
        prec = precision_score(
            test_targets, test_preds, average="macro", zero_division=0
        )
        rec = recall_score(
            test_targets, test_preds, average="macro", zero_division=0
        )
        f1 = f1_score(
            test_targets, test_preds, average="macro", zero_division=0
        )
        per_class_rec = recall_score(
            test_targets, test_preds, average=None, zero_division=0
        )
        mel_recall = float(per_class_rec[LABEL_MAP["mel"]])
        report = classification_report(
            test_targets, test_preds, target_names=target_names, zero_division=0
        )
        cm = confusion_matrix(test_targets, test_preds)

        print(f"\n  Test Accuracy  : {acc:.4f}")
        print(f"  Test Precision : {prec:.4f}  (macro)")
        print(f"  Test Recall    : {rec:.4f}  (macro)")
        print(f"  Test Macro-F1  : {f1:.4f}")
        print(f"  Melanoma Recall: {mel_recall:.4f}")
        print(f"\n{report}")

        results["proposed_vit_gat"] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "melanoma_recall": round(mel_recall, 4),
            "per_class_report": report,
            "confusion_matrix": cm.tolist(),
        }
    else:
        print(
            f"  [SKIP] GAT checkpoint not found at {gat_ckpt}.\n"
            "  Train the GAT first via:  python main.py --stage train"
        )
        results["proposed_vit_gat"] = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "macro_f1": None,
            "melanoma_recall": None,
            "per_class_report": "N/A — checkpoint missing",
            "confusion_matrix": None,
        }

    # ── Comparison table ───────────────────────────────────
    print_comparison_table(results)

    # ── Persist results ────────────────────────────────────
    # Strip non-serialisable fields for JSON output
    serialisable: dict[str, dict[str, Any]] = {}
    for key, metrics in results.items():
        serialisable[key] = {
            k: v
            for k, v in metrics.items()
            if k not in ("per_class_report",)
        }

    out_path = CHECKPOINT_DIR / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[INFO] Results saved → {out_path}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Comparison Table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_DISPLAY_NAMES: dict[str, str] = {
    "cnn_baseline": "CNN (ResNet-50)",
    "vit_only": "ViT (image only)",
    "vit_meta": "ViT + Metadata",
    "proposed_vit_gat": "Proposed ViT + GNN",
}


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a nicely formatted comparison table matching Table 3."""
    print("\n" + "=" * 64)
    print("  Model Comparison  (Report Table 3)")
    print("=" * 64)

    header = (
        f"  {'Model':<22s} {'Acc':>7s} {'Prec':>7s} "
        f"{'Rec':>7s} {'F1':>7s} {'Mel-Rec':>8s}"
    )
    print(header)
    print("  " + "─" * 60)

    for key in ("cnn_baseline", "vit_only", "vit_meta", "proposed_vit_gat"):
        if key not in results:
            continue
        m = results[key]
        name = _DISPLAY_NAMES.get(key, key)

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "  N/A "

        print(
            f"  {name:<22s} {_fmt(m.get('accuracy')):>7s} "
            f"{_fmt(m.get('precision')):>7s} "
            f"{_fmt(m.get('recall')):>7s} "
            f"{_fmt(m.get('macro_f1')):>7s} "
            f"{_fmt(m.get('melanoma_recall')):>8s}"
        )

    print("=" * 64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Entry-point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train & evaluate baseline models for HAM10000"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Max training epochs per baseline (default: 50)",
    )
    args = parser.parse_args()

    run_all_baselines(epochs=args.epochs)
