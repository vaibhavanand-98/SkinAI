#!/usr/bin/env python3
"""
main.py
=======
Orchestrator for the GNN-Based Explainable Framework for
Skin Cancer Progression Prediction.

Usage
-----
    # Full pipeline: download → preprocess → train → explain
    python main.py

    # Individual stages
    python main.py --stage download
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage explain

    # Training with custom hyperparameters
    python main.py --stage train --epochs 150 --lr 0.0005

    # Explain specific patients
    python main.py --stage explain --explain-nodes 42 100 256
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def stage_download():
    """Download HAM10000 from Kaggle."""
    print("\n" + "█" * 60)
    print("  STAGE 1 / 4 — Downloading Dataset")
    print("█" * 60 + "\n")
    from scripts.download_data import download_dataset, verify_dataset, RAW_DIR

    if RAW_DIR.exists() and any(RAW_DIR.rglob("HAM10000_metadata.csv")):
        print("[INFO] Dataset already present — skipping download.\n")
    else:
        download_dataset()
    verify_dataset()


def stage_preprocess(args):
    """Build the patient graph."""
    print("\n" + "█" * 60)
    print("  STAGE 2 / 5 — Preprocessing & Graph Construction")
    print("█" * 60 + "\n")
    from src.data_prep import build_graph

    data = build_graph(k=args.k, alpha=args.alpha)
    print(f"\n[SUMMARY] Graph ready:")
    print(f"  Nodes            : {data.num_nodes}")
    print(f"  Edges            : {data.num_edges}")
    print(f"  Node features    : {data.num_node_features}")
    print(f"  Train / Val / Test : "
          f"{data.train_mask.sum()} / {data.val_mask.sum()} / {data.test_mask.sum()}")


def stage_train(args):
    """Train the GAT model."""
    print("\n" + "█" * 60)
    print("  STAGE 3 / 5 — Training MultimodalGAT")
    print("█" * 60 + "\n")
    from src.train import train

    train(
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        heads=args.heads,
        dropout=args.dropout,
        patience=args.patience,
    )


def stage_explain(args):
    """Run GNNExplainer and ViT Image-level Explanations."""
    print("\n" + "█" * 60)
    print("  STAGE 4 / 5 — Generating Explanations (GNNExplainer + ViT Rollout/IG)")
    print("█" * 60 + "\n")
    from src.explain import run_explanations

    run_explanations(
        node_indices=args.explain_nodes,
        num_samples=args.explain_samples,
    )


def stage_baselines(args):
    """Run baseline models for report Table 3."""
    print("\n" + "█" * 60)
    print("  STAGE 5 / 5 — Running Baseline Models (CNN, ViT, ViT+Metadata)")
    print("█" * 60 + "\n")
    from src.baselines import run_all_baselines

    run_all_baselines(epochs=args.baseline_epochs)


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN-Based Explainable Framework for Skin Cancer Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["download", "preprocess", "train", "explain", "baselines", "all"],
        default="all",
        help=(
            "Which stage to run:\n"
            "  download   — Download HAM10000 from Kaggle\n"
            "  preprocess — Extract ViT features & build graph\n"
            "  train      — Train the ViT+GAT model\n"
            "  explain    — Run GNNExplainer & ViT explainability\n"
            "  baselines  — Train baseline classifiers for Table 3 comparison\n"
            "  all        — Run the full pipeline (default)"
        ),
    )

    # Graph construction parameters
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors in kNN graph")
    parser.add_argument("--alpha", type=float, default=0.7, help="Visual (alpha) vs clinical (1-alpha) similarity balance")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--baseline-epochs", type=int, default=40, help="Epochs for baseline MLP models")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)

    # Explainer options
    parser.add_argument("--explain-nodes", type=int, nargs="*", default=None)
    parser.add_argument("--explain-samples", type=int, default=5)

    args = parser.parse_args()

    if args.stage == "all":
        stage_download()
        stage_preprocess(args)
        stage_train(args)
        stage_explain(args)
        stage_baselines(args)
    elif args.stage == "download":
        stage_download()
    elif args.stage == "preprocess":
        stage_preprocess(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "explain":
        stage_explain(args)
    elif args.stage == "baselines":
        stage_baselines(args)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
