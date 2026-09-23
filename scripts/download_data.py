#!/usr/bin/env python3
"""
download_data.py
================
Downloads the HAM10000 dataset from Kaggle and extracts it
into the project's data/ directory.

Requires:
  - `kaggle` pip package installed.
  - A valid ~/.kaggle/kaggle.json API token.
    (Get one from https://www.kaggle.com/settings → "Create New Token")

If Kaggle credentials are not set up, the script will print
instructions and exit gracefully.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


# ── Configuration ──────────────────────────────────────────
DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def check_kaggle_credentials() -> bool:
    """Return True if Kaggle API credentials are available."""
    # Environment variables take priority
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Fall back to the token file
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    return token_path.exists()


def download_dataset():
    """Download the HAM10000 dataset from Kaggle."""
    if not check_kaggle_credentials():
        print(
            "\n╔══════════════════════════════════════════════════════════╗\n"
            "║  Kaggle API credentials not found.                      ║\n"
            "║                                                         ║\n"
            "║  Option A – Token file:                                 ║\n"
            "║    1. Go to https://www.kaggle.com/settings              ║\n"
            "║    2. Click 'Create New Token'                           ║\n"
            "║    3. Save kaggle.json to ~/.kaggle/kaggle.json          ║\n"
            "║    4. chmod 600 ~/.kaggle/kaggle.json                    ║\n"
            "║                                                         ║\n"
            "║  Option B – Environment variables:                      ║\n"
            "║    export KAGGLE_USERNAME=your_username                  ║\n"
            "║    export KAGGLE_KEY=your_api_key                       ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )
        sys.exit(1)

    # Import here so we get a clear error above if creds are missing
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading dataset '{DATASET_SLUG}' → {RAW_DIR}")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), unzip=False)
    print("[INFO] Download complete.")

    # ── Unzip ──────────────────────────────────────────────
    zip_files = list(RAW_DIR.glob("*.zip"))
    for zf in zip_files:
        print(f"[INFO] Extracting {zf.name} …")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(RAW_DIR)
        zf.unlink()  # remove the zip after extraction

    print(f"[INFO] Dataset ready at {RAW_DIR}")


def verify_dataset():
    """Quick sanity check that the key files exist."""
    meta_csv = list(RAW_DIR.rglob("HAM10000_metadata.csv"))
    image_dirs = list(RAW_DIR.rglob("*.jpg"))

    if not meta_csv:
        print("[WARN] HAM10000_metadata.csv not found in data/raw/.")
        print("       If you downloaded manually, place it inside data/raw/.")
        return False

    print(f"[INFO] Found metadata CSV  : {meta_csv[0]}")
    print(f"[INFO] Found {len(image_dirs)} JPEG images.")
    return True


# ── Entrypoint ─────────────────────────────────────────────
if __name__ == "__main__":
    # Skip download if data already present
    if RAW_DIR.exists() and any(RAW_DIR.rglob("HAM10000_metadata.csv")):
        print("[INFO] Dataset already present – skipping download.")
    else:
        download_dataset()

    verify_dataset()
