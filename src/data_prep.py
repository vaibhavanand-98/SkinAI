#!/usr/bin/env python3
"""
data_prep.py
============
End-to-end pipeline that converts the raw HAM10000 dataset into a
PyTorch Geometric graph ready for GAT training.

Steps
-----
1. Load and clean HAM10000_metadata.csv.
2. Extract 2048-dim visual embeddings from each image via a
   headless ResNet50, using **multi-augmentation averaging**
   (5 augmented views per image) for more robust features.
3. Encode clinical metadata (age, sex, localization) into a
   numerical feature vector.
4. Concatenate visual + metadata features → multimodal node matrix X.
5. Build a k-Nearest-Neighbors graph (k=5) based on clinical
   similarity to produce edge_index.
6. Map the 7 diagnostic labels to integer class targets y.
7. Persist the complete PyG `Data` object to disk (including
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
from torchvision import models, transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from tqdm import tqdm


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
VISUAL_DIM = 2048          # ResNet50 final-pool output
NUM_AUGMENTED_VIEWS = 5    # Number of augmented views to average
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
                # Fallback: skip (should not happen with correct data)
                self.paths.append(None)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path is None or not path.exists():
            # Return a zero tensor as placeholder
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class HAM10000AugmentedImages(Dataset):
    """
    Dataset that returns multiple augmented views of each image
    for robust multi-augmentation feature extraction.

    Each __getitem__ returns a tensor of shape [NUM_VIEWS, 3, H, W].
    """

    # Standard transform (no augmentation — the "canonical" view)
    base_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Augmented transforms for diverse views
    aug_transforms = [
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(degrees=(15, 15)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]

    def __init__(self, image_ids: list[str], raw_dir: Path):
        self.image_ids = image_ids
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
            return torch.zeros(NUM_AUGMENTED_VIEWS, 3, IMG_SIZE, IMG_SIZE)

        img = Image.open(path).convert("RGB")
        views = [self.base_transform(img)]
        for t in self.aug_transforms:
            views.append(t(img))

        return torch.stack(views)  # [NUM_VIEWS, 3, H, W]


# ── 3. Visual Feature Extraction ──────────────────────────

def extract_visual_features(image_ids: list[str]) -> torch.Tensor:
    """
    Pass images through a headless ResNet50 to obtain a
    [N, 2048] feature matrix.

    Uses multi-augmentation averaging: each image is passed through
    the CNN in NUM_AUGMENTED_VIEWS augmented forms, and the resulting
    embeddings are averaged for a more robust representation.
    """
    cache_path = PROCESSED_DIR / "visual_embeddings.pt"
    if cache_path.exists():
        print(f"[INFO] Loading cached visual embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"[INFO] Extracting visual features with {NUM_AUGMENTED_VIEWS}-view "
          f"augmentation on device={DEVICE} …")

    # Headless ResNet50
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()  # remove classification head
    backbone = backbone.to(DEVICE).eval()

    dataset = HAM10000AugmentedImages(image_ids, RAW_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CNN Feature Extraction (multi-aug)"):
            # batch shape: [B, NUM_VIEWS, 3, H, W]
            B, V, C, H, W = batch.shape
            # Flatten views into batch dimension
            flat = batch.view(B * V, C, H, W).to(DEVICE)
            feats = backbone(flat)                  # [B*V, 2048]
            feats = feats.view(B, V, -1)            # [B, V, 2048]
            # Average across augmented views
            avg_feats = feats.mean(dim=1)           # [B, 2048]
            all_feats.append(avg_feats.cpu())

    visual_matrix = torch.cat(all_feats, dim=0)   # [N, 2048]
    torch.save(visual_matrix, cache_path)
    print(f"[INFO] Visual embeddings shape: {visual_matrix.shape} → saved to {cache_path}")
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

def build_knn_graph(visual_features: torch.Tensor, meta_features: torch.Tensor, k: int = K_NEIGHBORS, visual_k: int = 150) -> torch.Tensor:
    """
    Build a 2-stage k-NN graph with weighted holistic similarity.
    1. Retrieve top `visual_k` neighbors using CNN visual embeddings.
    2. Compute combined distance: 0.7 * Visual_Dist + 0.3 * Clinical_Dist.
    3. Filter to top `k` neighbors (up to 12 if threshold met).
    """
    print(f"[INFO] Stage 1: Building {visual_k}-NN graph over {visual_features.shape[0]} nodes using CNN features...")
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
        combined_dists = 0.7 * candidate_visual_dists + 0.3 * clinical_dists
        
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

def build_graph() -> Data:
    """Master function: returns a PyG Data object ready for training."""
    cache_path = PROCESSED_DIR / "ham10000_graph.pt"
    if cache_path.exists():
        print(f"[INFO] Loading cached graph from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)

        # Backfill metadata if missing (for graphs cached before v2)
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

    # Visual features [N, 2048]
    visual = extract_visual_features(df["image_id"].tolist())

    # Clinical features [N, D_meta]
    meta = encode_metadata(df)

    # Multimodal node features [N, 2048 + D_meta]
    x = torch.cat([visual, meta], dim=1)
    print(f"[INFO] Multimodal node feature matrix X: {x.shape}")

    # Labels
    y = torch.tensor(
        df["dx"].map(LABEL_MAP).values,
        dtype=torch.long,
    )
    print(f"[INFO] Label distribution:\n{df['dx'].value_counts().to_string()}")

    # Graph edges
    edge_index = build_knn_graph(visual, meta, k=K_NEIGHBORS)

    # Train / Val / Test masks (70 / 15 / 15)
    n = len(df)
    perm = torch.randperm(n)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # ── Store metadata for dashboard / explainer ──────────
    # image_ids list — maps node index to image_id string
    data.image_ids = df["image_id"].tolist()
    # Store key metadata columns for the explainer
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
