#!/usr/bin/env python3
"""
explain.py
==========
Intuitive GNNExplainer integration for the MultimodalGAT.

For any given patient (node), this module produces:
1. **Influential Neighbors** – the top-K most similar patients that
   drove the model's prediction, along with their images, diagnoses,
   and clinical metadata.  This is the key "intuitive" output: instead
   of a confusing graph plot, the user sees "We predicted Melanoma
   because this lesion is similar to these 3 past patients who had
   Melanoma".
2. **Feature Group Importance** – whether visual (ViT) features or
   clinical metadata contributed more to the prediction.
3. **Prediction Summary** – predicted class, true class, confidence
   scores, and whether the model was correct.
4. **Visualisations** – saved as PNG files:
   - An intuitive "Similar Patients" card showing the input image
     alongside the most influential neighbor images.
   - A feature importance breakdown chart.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, to_networkx

from src.data_prep import build_graph, LABEL_MAP, LABEL_NAMES, VISUAL_DIM, RAW_DIR
from src.model import MultimodalGAT


# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPLAIN_DIR = PROJECT_ROOT / "explanations"
EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Metadata feature group names (must match data_prep.encode_metadata)
# These are approximate; exact count depends on the dataset values.
META_GROUPS = {
    "age": 1,
    "sex": 2,       # typically male / female
    "localization": 15,  # ~15 unique sites in HAM10000
}


# ── Helper: find image path ──────────────────────────────

def _find_image_path(image_id: str) -> Path | None:
    """Locate the JPEG file for a given image_id."""
    candidates = list(RAW_DIR.rglob(f"{image_id}.jpg"))
    return candidates[0] if candidates else None


# ── Explainer Setup ───────────────────────────────────────

def create_explainer(model: MultimodalGAT) -> Explainer:
    """
    Wrap the trained model in PyG's Explainer with GNNExplainer
    algorithm.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=30, lr=0.05),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    return explainer


# ── Explanation Generation ─────────────────────────────────

def explain_node(
    explainer: Explainer,
    data,
    node_idx: int,
    top_k_neighbors: int = 5,
    top_k_features: int = 20,
) -> dict:
    """
    Generate and return an intuitive explanation for a single node
    (patient).

    Returns
    -------
    dict with keys:
        node_idx, predicted_class, predicted_label, true_class,
        true_label, is_correct, confidence, confidence_scores,
        influential_neighbors (list of dicts with image_id, diagnosis,
        age, sex, localization, importance),
        feature_summary, top_features, image_id
    """
    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_idx,
    )

    # Predicted & true class
    model = explainer.model
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits[node_idx], dim=0)
    pred_class = logits[node_idx].argmax().item()
    true_class = data.y[node_idx].item()

    pred_label = IDX_TO_LABEL.get(pred_class, str(pred_class))
    true_label = IDX_TO_LABEL.get(true_class, str(true_class))

    # Confidence scores per class
    confidence_scores = {
        IDX_TO_LABEL[i]: round(probs[i].item() * 100, 2)
        for i in range(len(probs))
    }

    # ── Edge importance → Influential Neighbors ───────────
    edge_mask = explanation.edge_mask.cpu().numpy()
    ei = data.edge_index.cpu().numpy()

    # Find edges connected to this node
    involved_mask = (ei[0] == node_idx) | (ei[1] == node_idx)
    involved_indices = np.where(involved_mask)[0]

    # For each connected edge, identify the neighbor and its importance
    neighbor_importance = {}
    for edge_idx in involved_indices:
        src, dst = ei[0, edge_idx], ei[1, edge_idx]
        neighbor = dst if src == node_idx else src
        importance = float(edge_mask[edge_idx])
        # Keep max importance if multiple edges to same neighbor
        if neighbor not in neighbor_importance or importance > neighbor_importance[neighbor]:
            neighbor_importance[neighbor] = importance

    # Sort by importance and pick top-K
    sorted_neighbors = sorted(
        neighbor_importance.items(), key=lambda x: x[1], reverse=True
    )[:top_k_neighbors]

    influential_neighbors = []
    for neighbor_idx, importance in sorted_neighbors:
        neighbor_info = {
            "node_idx": int(neighbor_idx),
            "importance": round(importance, 4),
            "diagnosis": data.patient_dx[neighbor_idx] if hasattr(data, 'patient_dx') else "unknown",
            "diagnosis_full": LABEL_NAMES.get(
                data.patient_dx[neighbor_idx], "Unknown"
            ) if hasattr(data, 'patient_dx') else "Unknown",
            "image_id": data.image_ids[neighbor_idx] if hasattr(data, 'image_ids') else None,
            "age": round(data.patient_age[neighbor_idx].item(), 1) if hasattr(data, 'patient_age') else None,
            "sex": data.patient_sex[neighbor_idx] if hasattr(data, 'patient_sex') else None,
            "localization": data.patient_localization[neighbor_idx] if hasattr(data, 'patient_localization') else None,
        }
        influential_neighbors.append(neighbor_info)

    # ── Feature importance ─────────────────────────────────
    node_mask = explanation.node_mask[node_idx].cpu().numpy()

    # Split into visual vs metadata
    visual_importance = float(node_mask[:VISUAL_DIM].sum())
    meta_importance = float(node_mask[VISUAL_DIM:].sum())
    total = visual_importance + meta_importance + 1e-12

    feature_summary = {
        "visual_features_pct": round(visual_importance / total * 100, 1),
        "metadata_features_pct": round(meta_importance / total * 100, 1),
    }

    # Top individual feature dimensions
    top_feat_idxs = np.argsort(node_mask)[-top_k_features:][::-1]
    feature_details = []
    for fidx in top_feat_idxs:
        group = "visual_cnn" if fidx < VISUAL_DIM else "clinical_metadata"
        feature_details.append({
            "dim": int(fidx),
            "group": group,
            "importance": float(node_mask[fidx]),
        })

    # Patient's own info
    patient_image_id = data.image_ids[node_idx] if hasattr(data, 'image_ids') else None

    return {
        "node_idx": node_idx,
        "image_id": patient_image_id,
        "predicted_class": pred_label,
        "predicted_label": LABEL_NAMES.get(pred_label, pred_label),
        "true_class": true_label,
        "true_label": LABEL_NAMES.get(true_label, true_label),
        "is_correct": pred_class == true_class,
        "confidence": round(probs[pred_class].item() * 100, 2),
        "confidence_scores": confidence_scores,
        "influential_neighbors": influential_neighbors,
        "feature_summary": feature_summary,
        "top_features": feature_details,
        "edge_mask_raw": edge_mask,
        "node_mask_raw": node_mask,
    }


# ── Visualisation ──────────────────────────────────────────

def visualise_explanation(result: dict, data, save_dir: Path = EXPLAIN_DIR):
    """
    Save intuitive visualisations:
    1. "Similar Patients" panel — the input image alongside the most
       influential neighbor images with their diagnoses.
    2. Feature importance breakdown chart.
    3. Confidence bar chart for all classes.
    """
    node_idx = result["node_idx"]
    pred = result["predicted_label"]
    true = result["true_label"]
    is_correct = result["is_correct"]
    status_color = "#2ECC40" if is_correct else "#FF4136"
    status_text = "✓ Correct" if is_correct else "✗ Incorrect"

    neighbors = result["influential_neighbors"]

    # ── 1. Similar Patients Panel ─────────────────────────
    n_cols = 1 + len(neighbors)  # input + neighbors
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle(
        "Why did the model predict " + repr(pred) + "?\n"
        "Because this lesion is most similar to these past patients:",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Input image
    input_image_id = result["image_id"]
    input_img_path = _find_image_path(input_image_id) if input_image_id else None
    ax = axes[0]
    if input_img_path and input_img_path.exists():
        img = mpimg.imread(str(input_img_path))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
    ax.set_title(f"INPUT PATIENT\nPredicted: {pred}\nTrue: {true}",
                 fontsize=10, fontweight="bold", color=status_color)
    ax.axis("off")
    # Add colored border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(status_color)
        spine.set_linewidth(3)

    # Neighbor images
    for i, neighbor in enumerate(neighbors):
        ax = axes[1 + i]
        nb_image_id = neighbor.get("image_id")
        nb_img_path = _find_image_path(nb_image_id) if nb_image_id else None

        if nb_img_path and nb_img_path.exists():
            img = mpimg.imread(str(nb_img_path))
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)

        diag_full = neighbor.get("diagnosis_full", "Unknown")
        age = neighbor.get("age", "?")
        sex = neighbor.get("sex", "?")
        loc = neighbor.get("localization", "?")
        importance = neighbor.get("importance", 0)

        ax.set_title(
            f"Similar Patient #{i+1}\n"
            f"Diagnosis: {diag_full}\n"
            f"Age: {age}, Sex: {sex}\n"
            f"Site: {loc}\n"
            f"Similarity: {importance:.3f}",
            fontsize=8, fontweight="bold",
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_dir / f"node_{node_idx}_similar_patients.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Feature importance chart ────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    summary = result["feature_summary"]
    categories = ["Visual (ViT)\nImage Features", "Clinical Metadata\n(Age, Sex, Site)"]
    values = [summary["visual_features_pct"], summary["metadata_features_pct"]]
    colors = ["#0074D9", "#2ECC40"]

    bars = ax2.barh(categories, values, color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=12, fontweight="bold")

    ax2.set_xlabel("Relative Feature Importance (%)", fontsize=12)
    ax2.set_title(
        f"What mattered more for Patient {node_idx}'s prediction?\n"
        f"Predicted: {pred} | True: {true} | {status_text}",
        fontsize=13,
    )
    ax2.set_xlim(0, 110)
    fig2.tight_layout()
    fig2.savefig(save_dir / f"node_{node_idx}_features.png", dpi=150)
    plt.close(fig2)

    # ── 3. Confidence chart ────────────────────────────────
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    scores = result["confidence_scores"]
    classes = list(scores.keys())
    class_names = [LABEL_NAMES.get(c, c) for c in classes]
    confidences = list(scores.values())

    bar_colors = []
    for c in classes:
        if c == result["predicted_class"]:
            bar_colors.append("#FF4136" if not is_correct else "#2ECC40")
        elif c == result["true_class"]:
            bar_colors.append("#FFD700")
        else:
            bar_colors.append("#AAAAAA")

    bars3 = ax3.bar(class_names, confidences, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars3, confidences):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_ylabel("Confidence (%)", fontsize=12)
    ax3.set_title(
        f"Model Confidence per Class — Patient {node_idx}\n"
        f"Green = Predicted (if correct) | Red = Predicted (if wrong) | Gold = True Label",
        fontsize=11,
    )
    ax3.set_ylim(0, max(confidences) * 1.15 + 5)
    plt.xticks(rotation=30, ha="right")
    fig3.tight_layout()
    fig3.savefig(save_dir / f"node_{node_idx}_confidence.png", dpi=150)
    plt.close(fig3)

    print(f"[INFO] Explanations saved → {save_dir / f'node_{node_idx}_*.png'}")


# ── Main Explain Pipeline ─────────────────────────────────

def run_explanations(node_indices: list[int] | None = None, num_samples: int = 5):
    """
    Load the trained model, run GNNExplainer on sample test nodes,
    and save visualisations.
    """
    print("=" * 60)
    print("  GNNExplainer — Generating Intuitive Explanations")
    print("=" * 60)

    # Load graph
    data = build_graph().to(DEVICE)
    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    # Load trained model
    model = MultimodalGAT(
        visual_channels=VISUAL_DIM,
        meta_channels=num_features - VISUAL_DIM,
        hidden_channels=256,
        out_channels=num_classes,
        heads=4,
        dropout=0.4,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "best_gat.pt"
    if not ckpt_path.exists():
        print("[ERROR] No trained model found. Run train.py first.")
        return

    try:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        )
    except RuntimeError:
        print("[WARN] Checkpoint architecture mismatch — loading compatible weights only.")
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model_dict = model.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible)
        model.load_state_dict(model_dict)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")

    # Create explainer
    explainer = create_explainer(model)

    # Pick nodes to explain
    if node_indices is None:
        test_nodes = data.test_mask.nonzero(as_tuple=True)[0]
        # Sample diverse: one per class if possible
        selected = []
        seen_classes = set()
        for nid in test_nodes.tolist():
            cls = data.y[nid].item()
            if cls not in seen_classes and len(selected) < num_samples:
                selected.append(nid)
                seen_classes.add(cls)
        # Fill remaining if needed
        for nid in test_nodes.tolist():
            if len(selected) >= num_samples:
                break
            if nid not in selected:
                selected.append(nid)
        node_indices = selected

    print(f"[INFO] Explaining {len(node_indices)} nodes: {node_indices}\n")

    for nid in node_indices:
        print(f"─── Explaining node {nid} ───")
        result = explain_node(explainer, data, nid)

        status = "✓ CORRECT" if result["is_correct"] else "✗ INCORRECT"
        print(f"  Predicted : {result['predicted_label']} ({result['confidence']:.1f}%)")
        print(f"  True      : {result['true_label']}")
        print(f"  Status    : {status}")
        print(f"  Visual importance   : {result['feature_summary']['visual_features_pct']}%")
        print(f"  Metadata importance : {result['feature_summary']['metadata_features_pct']}%")
        print(f"  Most influential similar patients:")
        for i, nb in enumerate(result["influential_neighbors"][:3]):
            print(f"    #{i+1}: {nb['diagnosis_full']} "
                  f"(age={nb['age']}, sex={nb['sex']}, "
                  f"site={nb['localization']}, "
                  f"similarity={nb['importance']:.4f})")
        print()

        visualise_explanation(result, data)

        # ── Image-level explanations (ViT Attention Rollout & IG) ──
        input_image_id = result.get("image_id")
        input_img_path = _find_image_path(input_image_id) if input_image_id else None
        if input_img_path and input_img_path.exists():
            try:
                from src.vit_explain import explain_image
                print(f"  [ViT Explain] Generating Attention Rollout & Integrated Gradients for {input_image_id}...")
                vit_res = explain_image(input_img_path, EXPLAIN_DIR, node_idx=nid, device=DEVICE)
                print(f"  [ViT Explain] Saved heatmaps: {vit_res}")
            except Exception as e:
                print(f"  [ViT Explain] Skipped image-level explanation: {e}")

    print(f"\n[DONE] All explanations saved to {EXPLAIN_DIR}")


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNNExplainer for MultimodalGAT")
    parser.add_argument(
        "--nodes", type=int, nargs="*", default=None,
        help="Specific node indices to explain. Default: auto-select from test set.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of test nodes to explain (when --nodes is not set).",
    )
    args = parser.parse_args()
    run_explanations(node_indices=args.nodes, num_samples=args.num_samples)
