#!/usr/bin/env python3
"""
dashboard/app.py
================
FastAPI backend for the GNN Skin Cancer Prediction Dashboard.

Endpoints
---------
GET  /                      → Serve the frontend
GET  /api/patients           → List all patients with metadata
GET  /api/patients/search    → Search/filter patients
GET  /api/predict/{node_idx} → Run prediction + explanation for a node
GET  /api/image/{image_id}   → Serve a raw lesion image
GET  /api/stats              → Dataset statistics
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from src.data_prep import build_graph, LABEL_MAP, LABEL_NAMES, RAW_DIR
from src.model import MultimodalGAT
from src.explain import create_explainer, explain_node

# ── Paths ──────────────────────────────────────────────────
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STATIC_DIR = Path(__file__).resolve().parent / "static"

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="GNN Skin Cancer Dashboard",
    description="Interactive dashboard for GNN-based skin lesion classification",
    version="2.0",
)

# ── Global State (loaded once at startup) ──────────────────
_state = {
    "data": None,
    "model": None,
    "explainer": None,
    "image_lookup": {},  # image_id → file path
}


@app.on_event("startup")
def load_model_and_data():
    """Load the graph and model once at startup."""
    print("[Dashboard] Loading graph data...")
    data = build_graph().to(DEVICE)
    _state["data"] = data

    num_features = data.num_node_features
    num_classes = int(data.y.max().item()) + 1

    print("[Dashboard] Loading trained model...")
    model = MultimodalGAT(
        visual_channels=2048,
        meta_channels=num_features - 2048,
        hidden_channels=256,
        out_channels=num_classes,
        heads=4,
        dropout=0.4,
    ).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / "best_gat.pt"
    if ckpt_path.exists():
        try:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            )
            print(f"[Dashboard] Model loaded from {ckpt_path}")
        except RuntimeError:
            print(f"[Dashboard] WARNING: Checkpoint architecture mismatch — "
                  f"loading compatible weights only. Retrain for best results.")
            state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model_dict = model.state_dict()
            # Filter out keys with mismatched shapes
            compatible = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(compatible)
            model.load_state_dict(model_dict)
            print(f"[Dashboard] Loaded {len(compatible)}/{len(state_dict)} compatible weights.")
        model.eval()
    else:
        print("[Dashboard] WARNING: No trained model found! Train the model first.")

    _state["model"] = model
    _state["explainer"] = create_explainer(model)

    # Build image lookup
    print("[Dashboard] Building image path lookup...")
    for p in RAW_DIR.rglob("*.jpg"):
        _state["image_lookup"][p.stem] = p
    print(f"[Dashboard] Found {len(_state['image_lookup'])} images.")
    print("[Dashboard] Ready! ✓")


# ── Static Files ──────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the main dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text()


# ── API Endpoints ─────────────────────────────────────────

@app.get("/api/stats")
def get_stats():
    """Return dataset statistics."""
    data = _state["data"]
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    # Class distribution
    y = data.y.cpu().numpy()
    class_counts = {}
    for i in range(int(y.max()) + 1):
        label = IDX_TO_LABEL.get(i, str(i))
        class_counts[LABEL_NAMES.get(label, label)] = int((y == i).sum())

    return {
        "total_patients": int(data.num_nodes),
        "total_edges": int(data.num_edges),
        "num_features": int(data.num_node_features),
        "num_classes": int(y.max()) + 1,
        "train_count": int(data.train_mask.sum()),
        "val_count": int(data.val_mask.sum()),
        "test_count": int(data.test_mask.sum()),
        "class_distribution": class_counts,
    }


@app.get("/api/patients")
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    diagnosis: str = Query(None),
    split: str = Query(None),  # "train", "val", "test"
):
    """List patients with pagination and optional filtering."""
    data = _state["data"]
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    n = data.num_nodes
    patients = []

    for i in range(n):
        # Filter by split
        if split == "train" and not data.train_mask[i]:
            continue
        if split == "val" and not data.val_mask[i]:
            continue
        if split == "test" and not data.test_mask[i]:
            continue

        dx = data.patient_dx[i] if hasattr(data, 'patient_dx') else IDX_TO_LABEL.get(data.y[i].item(), "?")

        # Filter by diagnosis
        if diagnosis and dx != diagnosis:
            continue

        image_id = data.image_ids[i] if hasattr(data, 'image_ids') else None

        patient = {
            "node_idx": i,
            "image_id": image_id,
            "diagnosis": dx,
            "diagnosis_full": LABEL_NAMES.get(dx, dx),
            "age": round(data.patient_age[i].item(), 1) if hasattr(data, 'patient_age') else None,
            "sex": data.patient_sex[i] if hasattr(data, 'patient_sex') else None,
            "localization": data.patient_localization[i] if hasattr(data, 'patient_localization') else None,
            "split": "train" if data.train_mask[i] else ("val" if data.val_mask[i] else "test"),
        }
        patients.append(patient)

    # Pagination
    total = len(patients)
    start = (page - 1) * page_size
    end = start + page_size
    page_patients = patients[start:end]

    return {
        "patients": page_patients,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@app.get("/api/predict/{node_idx}")
def predict_and_explain(node_idx: int):
    """
    Run the model on a specific patient and return the prediction,
    ground truth, correctness, and intuitive explanation.
    """
    data = _state["data"]
    model = _state["model"]
    explainer = _state["explainer"]

    if data is None or model is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded yet")

    if node_idx < 0 or node_idx >= data.num_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_idx} not found")

    # Run explanation
    result = explain_node(explainer, data, node_idx, top_k_neighbors=5)

    # Add image paths for the frontend
    patient_image_id = result.get("image_id")
    has_image = patient_image_id and patient_image_id in _state["image_lookup"]

    # Patient metadata
    patient_info = {
        "age": round(data.patient_age[node_idx].item(), 1) if hasattr(data, 'patient_age') else None,
        "sex": data.patient_sex[node_idx] if hasattr(data, 'patient_sex') else None,
        "localization": data.patient_localization[node_idx] if hasattr(data, 'patient_localization') else None,
        "split": "train" if data.train_mask[node_idx] else ("val" if data.val_mask[node_idx] else "test"),
    }

    # Clean up non-serializable numpy arrays from result
    clean_result = {
        k: v for k, v in result.items()
        if k not in ("edge_mask_raw", "node_mask_raw")
    }

    clean_result["patient_info"] = patient_info
    clean_result["has_image"] = has_image

    return clean_result


@app.get("/api/image/{image_id}")
def serve_image(image_id: str):
    """Serve a raw lesion image by image_id."""
    img_path = _state["image_lookup"].get(image_id)
    if img_path is None or not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return FileResponse(str(img_path), media_type="image/jpeg")


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
