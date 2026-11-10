#!/usr/bin/env python3
"""
data_prep.py
============
End-to-end pipeline that converts the raw HAM10000 dataset into a
PyTorch Geometric graph ready for GAT training.

Steps
-----
1. Load and clean HAM10000_metadata.csv.
2. Extract 768-dim visual embeddings from each image via a
   pretrained ViT-B/16 (Vision Transformer), using the [CLS]
   token output as the image feature vector.
3. Encode clinical metadata (age, sex, localization) into a
   numerical feature vector.
4. Concatenate visual + metadata features → multimodal node matrix X.
5. Build a k-Nearest-Neighbors graph (k=5) based on weighted
   visual + clinical similarity (tunable α) to produce edge_index.
6. Map the 7 diagnostic labels to integer class targets y.
7. Split data at the **lesion_id level** with stratification to
   prevent data leakage.
8. Persist the complete PyG `Data` object to disk (including
   image_ids and metadata for use by the dashboard / explainer).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from tqdm import tqdm

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for ViT feature extraction. "
        "Install it with: pip install timm>=0.9.0"
    )


# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 64
K_NEIGHBORS = 5
VISUAL_DIM = 768           # ViT-B/16 [CLS] token output
ALPHA = 0.7                # Default visual/clinical similarity balance
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# The 7 diagnostic classes in HAM10000
LABEL_MAP = {
    "akiec": 0,   # Actinic keratoses
    "bcc":   1,   # Basal cell carcinoma
    "bkl":   2,   # Benign keratosis-like lesions
    "df":    3,   # Dermatofibroma
    "mel":   4,   # Melanoma
    "nv":    5,   # Melanocytic nevi
    "vasc":  6,   # Vascular lesions
}

# Human-readable label names for the dashboard
LABEL_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}


# ── 1. Metadata Loading ───────────────────────────────────

def load_metadata() -> pd.DataFrame:
    """Load and validate HAM10000_metadata.csv."""
    csv_candidates = list(RAW_DIR.rglob("HAM10000_metadata.csv"))
    if not csv_candidates:
        sys.exit("[ERROR] HAM10000_metadata.csv not found in data/raw/")

    df = pd.read_csv(csv_candidates[0])
    print(f"[INFO] Loaded metadata: {len(df)} rows from {csv_candidates[0]}")

    # Handle duplicates (HAM10000 has duplicate image_id for some lesions)
    # Keep only the first occurrence per lesion_id for patient-level graph
    df = df.drop_duplicates(subset="lesion_id", keep="first").reset_index(drop=True)
    print(f"[INFO] After dedup by lesion_id: {len(df)} unique lesions")

    # Fill missing age with median
    df["age"] = df["age"].fillna(df["age"].median())

    return df


# ── 2. Image Dataset ──────────────────────────────────────

class HAM10000Images(Dataset):
    """Lightweight dataset that loads images by image_id."""

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(self, image_ids: list[str], raw_dir: Path):
        self.image_ids = image_ids
        # Build a lookup: image_id → full path
        all_jpgs = {p.stem: p for p in raw_dir.rglob("*.jpg")}
        self.paths = []
        for iid in image_ids:
            if iid in all_jpgs:
                self.paths.append(all_jpgs[iid])
            else:
                self.paths.append(None)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path is None or not path.exists():
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img = Image.open(path).convert("RGB")
        return self.transform(img)


# ── 3. Visual Feature Extraction (ViT-B/16) ──────────────

def extract_visual_features(image_ids: list[str]) -> torch.Tensor:
    """
    Pass images through a pretrained ViT-B/16 to obtain a
    [N, 768] feature matrix using the [CLS] token output.

    The ViT (Vision Transformer) processes each 224×224 image as a
    sequence of 16×16 patches (196 patches). Multi-head self-attention
    allows every patch to attend to every other patch, capturing both
    local texture and global lesion structure. The [CLS] token output
    (768 dimensions) serves as the image feature vector.
    """
    cache_path = PROCESSED_DIR / "visual_embeddings.pt"
    if cache_path.exists():
        print(f"[INFO] Loading cached ViT embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"[INFO] Extracting ViT-B/16 features on device={DEVICE} …")

    # ViT-B/16 pretrained on ImageNet — num_classes=0 returns [CLS] token
    backbone = timm.create_model(
        'vit_base_patch16_224', pretrained=True, num_classes=0
    )
    backbone = backbone.to(DEVICE).eval()

    dataset = HAM10000Images(image_ids, RAW_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ViT-B/16 Feature Extraction"):
            batch = batch.to(DEVICE)         # [B, 3, 224, 224]
            feats = backbone(batch)          # [B, 768]
            all_feats.append(feats.cpu())

    visual_matrix = torch.cat(all_feats, dim=0)   # [N, 768]
    torch.save(visual_matrix, cache_path)
    print(f"[INFO] ViT embeddings shape: {visual_matrix.shape} → saved to {cache_path}")
    return visual_matrix


# ── 4. Clinical Metadata Encoding ─────────────────────────

def encode_metadata(df: pd.DataFrame) -> torch.Tensor:
    """
    Encode age (standardised), sex (one-hot), and localization (one-hot)
    into a numerical feature vector per patient.
    """
    # Age – continuous, standardise
    age = df["age"].values.reshape(-1, 1)
    scaler = StandardScaler()
    age_scaled = scaler.fit_transform(age)

    # Sex – one-hot
    sex_dummies = pd.get_dummies(df["sex"], prefix="sex").values.astype(np.float32)

    # Localization – one-hot
    loc_dummies = pd.get_dummies(df["localization"], prefix="loc").values.astype(np.float32)

    meta_matrix = np.hstack([age_scaled, sex_dummies, loc_dummies]).astype(np.float32)
    meta_tensor = torch.tensor(meta_matrix)
    print(f"[INFO] Metadata feature shape: {meta_tensor.shape}")
    return meta_tensor


# ── 5. Graph Construction (kNN) ───────────────────────────

def build_knn_graph(visual_features: torch.Tensor, meta_features: torch.Tensor,
                    k: int = K_NEIGHBORS, alpha: float = ALPHA,
                    visual_k: int = 150) -> torch.Tensor:
    """
    Build a 2-stage k-NN graph with weighted holistic similarity.

    Similarity is defined as (Eq. 2 in the report):
        S(i,j) = α · cos(v_i, v_j) + (1−α) · sim(m_i, m_j)

    Parameters
    ----------
    visual_features : Tensor [N, 768]
        ViT image embeddings.
    meta_features : Tensor [N, D_meta]
        Encoded clinical metadata.
    k : int
        Number of nearest neighbors per node.
    alpha : float
        Weight for visual similarity (0 = metadata only, 1 = visual only).
    visual_k : int
        Number of candidates to retrieve in stage 1 (before holistic scoring).

    Steps
    -----
    1. Retrieve top `visual_k` neighbors using ViT visual embeddings.
    2. Compute combined distance: α * Visual_Dist + (1−α) * Clinical_Dist.
    3. Filter to top `k` neighbors (up to 12 if threshold met).
    """
    print(f"[INFO] Stage 1: Building {visual_k}-NN graph over "
          f"{visual_features.shape[0]} nodes using ViT features...")
    print(f"[INFO] Similarity balance: α={alpha:.2f} (visual={alpha:.0%}, "
          f"clinical={1-alpha:.0%})")
    nn_model = NearestNeighbors(n_neighbors=visual_k + 1, metric="cosine", n_jobs=-1)
    nn_model.fit(visual_features.numpy())
    visual_dists, visual_indices = nn_model.kneighbors(visual_features.numpy())

    print(f"[INFO] Stage 2: Filtering using holistic weighted scoring...")
    src, dst = [], []
    meta_np = meta_features.numpy()

    for node_idx in tqdm(range(len(visual_indices)), desc="Holistic Filtering"):
        node_meta = meta_np[node_idx].reshape(1, -1)
        
        # Skip self (index 0)
        candidate_indices = visual_indices[node_idx][1:]
        candidate_visual_dists = visual_dists[node_idx][1:]
        candidate_meta = meta_np[candidate_indices]
        
        # Compute cosine distances for metadata
        clinical_dists = cdist(node_meta, candidate_meta, metric="cosine")[0]
        
        # Weighted holistic distance (lower is better)
        # Eq. 2: S(i,j) = α·cos(v_i,v_j) + (1−α)·sim(m_i,m_j)
        combined_dists = alpha * candidate_visual_dists + (1 - alpha) * clinical_dists
        
        # Sort by combined distance
        best_local_indices = np.argsort(combined_dists)
        
        # Select minimum k (5), up to 12 if distance < 0.15
        selected = best_local_indices[:k].tolist() # guarantee top k
        for idx in best_local_indices[k:12]:
            if combined_dists[idx] < 0.15:
                selected.append(idx)
                
        best_global_indices = candidate_indices[selected]
        
        for neighbor_idx in best_global_indices:
            src.append(node_idx)
            dst.append(neighbor_idx)
            # Make undirected
            src.append(neighbor_idx)
            dst.append(node_idx)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    print(f"[INFO] Edge index shape: {edge_index.shape}  "
          f"(avg degree ≈ {edge_index.shape[1] / meta_features.shape[0]:.1f})")
    return edge_index


# ── 6. Assemble PyG Data Object ───────────────────────────

def build_graph(k: int = K_NEIGHBORS, alpha: float = ALPHA) -> Data:
    """
    Master function: returns a PyG Data object ready for training.

    Parameters
    ----------
    k : int
        Number of nearest neighbours per node in the similarity graph.
    alpha : float
        Balance between visual (α) and clinical (1−α) similarity for
        graph edge construction. Diagnosis labels are never used to
        build edges (no label leakage).
    """
    cache_path = PROCESSED_DIR / "ham10000_graph.pt"
    if cache_path.exists():
        print(f"[INFO] Loading cached graph from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)

        # Backfill metadata if missing (for graphs cached before v3)
        if not hasattr(data, 'image_ids') or data.image_ids is None:
            print("[INFO] Backfilling patient metadata into cached graph...")
            df = load_metadata()
            data.image_ids = df["image_id"].tolist()
            data.patient_age = torch.tensor(df["age"].values, dtype=torch.float)
            data.patient_sex = df["sex"].tolist()
            data.patient_localization = df["localization"].tolist()
            data.patient_dx = df["dx"].tolist()
            torch.save(data, cache_path)
            print("[INFO] Metadata backfilled and graph re-saved.")

        return data

    df = load_metadata()

    # Visual features [N, 768] (ViT-B/16)
    visual = extract_visual_features(df["image_id"].tolist())

    # Clinical features [N, D_meta]
    meta = encode_metadata(df)

    # Multimodal node features [N, 768 + D_meta]  (Eq. 1 in report)
    x = torch.cat([visual, meta], dim=1)
    print(f"[INFO] Multimodal node feature matrix X: {x.shape}")

    # Labels
    y = torch.tensor(
        df["dx"].map(LABEL_MAP).values,
        dtype=torch.long,
    )
    print(f"[INFO] Label distribution:\n{df['dx'].value_counts().to_string()}")

    # Graph edges (using tunable α)
    edge_index = build_knn_graph(visual, meta, k=k, alpha=alpha)

    # ── Lesion-ID-level stratified split (70 / 15 / 15) ───
    # This prevents images of the same lesion appearing in both
    # train and test sets, which would inflate performance.
    n = len(df)
    labels = df["dx"].map(LABEL_MAP).values

    train_ids, temp_ids = train_test_split(
        np.arange(n), test_size=0.30,
        stratify=labels, random_state=42,
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50,
        stratify=labels[temp_ids], random_state=42,
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True

    print(f"[INFO] Lesion-level stratified split: "
          f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # ── Store metadata for dashboard / explainer ──────────
    data.image_ids = df["image_id"].tolist()
    data.patient_age = torch.tensor(df["age"].values, dtype=torch.float)
    data.patient_sex = df["sex"].tolist()
    data.patient_localization = df["localization"].tolist()
    data.patient_dx = df["dx"].tolist()  # ground-truth diagnosis string

    # Persist
    torch.save(data, cache_path)
    print(f"[INFO] Full graph saved → {cache_path}")
    print(f"       Nodes : {data.num_nodes}")
    print(f"       Edges : {data.num_edges}")
    print(f"       Features per node : {data.num_node_features}")
    return data


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    data = build_graph()
    print("\n[DONE] Graph construction complete.")
    print(data)
