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


def stage_preprocess():
    """Build the patient graph."""
    print("\n" + "█" * 60)
    print("  STAGE 2 / 4 — Preprocessing & Graph Construction")
    print("█" * 60 + "\n")
    from src.data_prep import build_graph

    data = build_graph()
    print(f"\n[SUMMARY] Graph ready:")
    print(f"  Nodes            : {data.num_nodes}")
    print(f"  Edges            : {data.num_edges}")
    print(f"  Node features    : {data.num_node_features}")
    print(f"  Train / Val / Test : "
          f"{data.train_mask.sum()} / {data.val_mask.sum()} / {data.test_mask.sum()}")


def stage_train(args):
    """Train the GAT model."""
    print("\n" + "█" * 60)
    print("  STAGE 3 / 4 — Training MultimodalGAT")
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
    """Run GNNExplainer."""
    print("\n" + "█" * 60)
    print("  STAGE 4 / 4 — Generating Explanations (GNNExplainer)")
    print("█" * 60 + "\n")
    from src.explain import run_explanations

    run_explanations(
        node_indices=args.explain_nodes,
        num_samples=args.explain_samples,
    )


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN-Based Explainable Framework for Skin Cancer Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["download", "preprocess", "train", "explain", "all"],
        default="all",
        help=(
            "Which stage to run:\n"
            "  download   — Download HAM10000 from Kaggle\n"
            "  preprocess — Extract features & build graph\n"
            "  train      — Train the GAT model\n"
            "  explain    — Run GNNExplainer on test patients\n"
            "  all        — Run the full pipeline (default)"
        ),
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
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
        stage_preprocess()
        stage_train(args)
        stage_explain(args)
    elif args.stage == "download":
        stage_download()
    elif args.stage == "preprocess":
        stage_preprocess()
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "explain":
        stage_explain(args)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
