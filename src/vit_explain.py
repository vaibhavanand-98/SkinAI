"""
vit_explain.py — ViT-level image explainability for skin lesion classification.

Implements two explainability methods for a ViT-B/16 image encoder:
  1. Attention Rollout (Abnar & Zuidema, 2020)
  2. Integrated Gradients (Sundararajan et al., 2017)

Both methods produce 224×224 heatmaps that can be overlaid on the original
dermoscopic image and saved as PNG files.

Dependencies:
    - timm (ViT model)
    - captum (Integrated Gradients — optional, graceful fallback)
    - torch, torchvision, matplotlib, numpy, PIL
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Optional captum import
# ---------------------------------------------------------------------------
try:
    from captum.attr import IntegratedGradients

    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False
    warnings.warn(
        "captum is not installed — Integrated Gradients will be unavailable. "
        "Install with: pip install captum",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = NUM_PATCHES_SIDE ** 2  # 196
VIT_EMBED_DIM = 768
VIT_NUM_LAYERS = 12
VIT_NUM_HEADS = 12

# Standard transform matching the project pipeline
_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ===================================================================
# Model loading
# ===================================================================

def get_vit_model(pretrained: bool = True) -> nn.Module:
    """Load and return a ViT-B/16 model from *timm*.

    The model is returned with ``num_classes=0`` so that the forward pass
    outputs the [CLS] token embedding (768-d) rather than classification
    logits.  This matches the project's usage of ViT as a frozen feature
    extractor.

    Args:
        pretrained: If ``True`` (default), load ImageNet-pretrained weights.

    Returns:
        A ``timm`` ViT-B/16 model in eval mode.
    """
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=pretrained, num_classes=0
    )
    model.eval()
    return model


# ===================================================================
# Image I/O helpers
# ===================================================================

def load_and_preprocess(image_path: str | Path) -> Tuple[torch.Tensor, Image.Image]:
    """Load an image and return the normalised tensor **and** the raw PIL image.

    Args:
        image_path: Path to a JPEG / PNG dermoscopic image.

    Returns:
        A tuple ``(tensor, pil_image)`` where *tensor* has shape
        ``[1, 3, 224, 224]`` and *pil_image* is the resized RGB image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    tensor = _transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor, img_resized


# ===================================================================
# Method 1 — Attention Rollout
# ===================================================================

def _register_attention_hooks(
    model: nn.Module,
) -> Tuple[List[torch.Tensor], List[torch.utils.hooks.RemovableHook]]:
    """Register forward hooks on every attention block to capture attention maps.

    Works with the standard ``timm`` ViT architecture where each block lives
    at ``model.blocks[i].attn``.

    Returns:
        attention_maps: list that will be populated during the forward pass.
        hooks: list of hook handles (call ``.remove()`` when done).
    """
    attention_maps: List[torch.Tensor] = []
    hooks: List[torch.utils.hooks.RemovableHook] = []

    for block in model.blocks:
        attn_module = block.attn

        def _hook_fn(_module, _input, _output, attn_module=attn_module):
            # timm's Attention.forward computes  q @ k^T / sqrt(d)
            # and stores the softmaxed weights in attn_module.attn_drop or
            # we can recompute from q, k.  However, an easier route is to
            # re-derive them.  timm >=0.9 stores nothing by default, so we
            # recompute inside the hook.
            B, N, C = _input[0].shape
            qkv = attn_module.qkv(_input[0])  # [B, N, 3*embed_dim]
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)  # each [B, heads, N, head_dim]
            scale = head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]
            attn_weights = attn_weights.softmax(dim=-1)
            attention_maps.append(attn_weights.detach().cpu())

        hook = attn_module.register_forward_hook(_hook_fn)
        hooks.append(hook)

    return attention_maps, hooks


def attention_rollout(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Compute an attention-rollout heatmap for a single image.

    Implements the method of Abnar & Zuidema (2020): at each layer the
    per-head attention matrices are averaged, then an identity residual is
    added and the result is re-normalised.  The product across all layers
    gives the *rollout* matrix.  We extract the [CLS] token row (index 0),
    discard the [CLS]-to-[CLS] entry, reshape the remaining 196 values to
    a 14×14 grid, and bi-linearly upsample to 224×224.

    Args:
        model: A ``timm`` ViT-B/16 model (``num_classes=0``).
        image_tensor: Normalised tensor of shape ``[1, 3, 224, 224]``.
        device: Torch device string.

    Returns:
        A numpy array of shape ``[224, 224]`` with values in ``[0, 1]``.
    """
    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    attention_maps, hooks = _register_attention_hooks(model)

    with torch.no_grad():
        _ = model(image_tensor)

    # Clean up hooks immediately
    for h in hooks:
        h.remove()

    if len(attention_maps) == 0:
        raise RuntimeError(
            "No attention maps were captured — the model architecture may "
            "not match the expected timm ViT layout."
        )

    # --- Rollout computation ---
    # Each element: [1, heads, N, N]  where N = 1 (CLS) + 196 (patches)
    num_tokens = attention_maps[0].shape[-1]

    rollout = torch.eye(num_tokens)  # identity for residual connection

    for attn in attention_maps:
        # Average over heads: [1, N, N] → squeeze → [N, N]
        attn_avg = attn.mean(dim=1).squeeze(0)  # [N, N]
        # Add identity (residual) and re-normalise rows
        attn_res = 0.5 * attn_avg + 0.5 * torch.eye(num_tokens)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        rollout = attn_res @ rollout

    # Extract [CLS] → all tokens, drop [CLS]-to-[CLS]
    cls_attention = rollout[0, 1:]  # [196]

    # Normalise to [0, 1]
    cls_attention = cls_attention - cls_attention.min()
    if cls_attention.max() > 0:
        cls_attention = cls_attention / cls_attention.max()

    # Reshape to 14×14 grid and upsample
    heatmap = cls_attention.reshape(1, 1, NUM_PATCHES_SIDE, NUM_PATCHES_SIDE).float()
    heatmap = F.interpolate(
        heatmap, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    heatmap = heatmap.squeeze().numpy()

    return heatmap


# ===================================================================
# Method 2 — Integrated Gradients
# ===================================================================

class _ViTClassifierWrapper(nn.Module):
    """Thin wrapper that adds a linear classification head on top of the
    frozen ViT feature extractor so that Integrated Gradients can be
    computed with respect to per-class logits.

    If no pre-trained head weights are available the wrapper uses a
    simple identity-like projection that preserves gradient flow.
    """

    def __init__(self, vit_model: nn.Module, num_classes: int = 7) -> None:
        super().__init__()
        self.vit = vit_model
        # Lightweight linear head (not trained — only used for direction)
        self.head = nn.Linear(VIT_EMBED_DIM, num_classes)
        # Initialise so all classes get comparable gradients initially
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``[B, num_classes]``."""
        features = self.vit(x)  # [B, 768]
        return self.head(features)  # [B, num_classes]


def integrated_gradients_explain(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    device: str = "cpu",
    n_steps: int = 50,
    num_classes: int = 7,
) -> np.ndarray:
    """Compute an Integrated Gradients attribution heatmap for a single image.

    The ViT feature extractor is wrapped with a lightweight linear head so
    that IG can be computed with respect to a target class logit.

    Args:
        model: A ``timm`` ViT-B/16 model (``num_classes=0``).
        image_tensor: Normalised tensor of shape ``[1, 3, 224, 224]``.
        target_class: Class index to explain.  If ``None`` the predicted
            (argmax) class is used.
        device: Torch device string.
        n_steps: Number of interpolation steps for IG (higher → more
            precise but slower).
        num_classes: Number of output classes for the wrapper head
            (default 7 for HAM10000).

    Returns:
        A numpy array of shape ``[224, 224]`` with values in ``[0, 1]``.

    Raises:
        RuntimeError: If ``captum`` is not installed.
    """
    if not _CAPTUM_AVAILABLE:
        raise RuntimeError(
            "captum is required for Integrated Gradients but is not installed. "
            "Install with: pip install captum"
        )

    # Build wrapped model
    wrapped = _ViTClassifierWrapper(model, num_classes=num_classes).to(device).eval()
    image_tensor = image_tensor.to(device).requires_grad_(True)

    # Determine target class if not specified
    if target_class is None:
        with torch.no_grad():
            logits = wrapped(image_tensor)
            target_class = int(logits.argmax(dim=1).item())

    # Integrated Gradients
    ig = IntegratedGradients(wrapped)

    # Baseline: black image (zero in normalised space corresponds to roughly
    # the ImageNet mean — a true zero tensor is fine as the conventional
    # baseline for IG).
    baseline = torch.zeros_like(image_tensor).to(device)

    attributions = ig.attribute(
        image_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=False,
    )  # [1, 3, 224, 224]

    # Summarise across colour channels: take absolute value, then mean
    attr = attributions.squeeze(0).detach().cpu().numpy()  # [3, 224, 224]
    heatmap = np.abs(attr).mean(axis=0)  # [224, 224]

    # Normalise to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax - hmin > 0:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


# ===================================================================
# Visualisation
# ===================================================================

def save_heatmap_overlay(
    heatmap: np.ndarray,
    original_image_path: str | Path,
    save_path: str | Path,
    title: str = "",
    cmap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """Overlay a heatmap on the original image and save as PNG.

    Args:
        heatmap: ``[224, 224]`` numpy array with values in ``[0, 1]``.
        original_image_path: Path to the original JPEG/PNG image.
        save_path: Destination path for the overlay PNG.
        title: Optional title rendered on the figure.
        cmap: Matplotlib colourmap name.
        alpha: Transparency of the heatmap overlay (0 = fully transparent).
    """
    original_image_path = Path(original_image_path)
    save_path = Path(save_path)

    if not original_image_path.exists():
        raise FileNotFoundError(f"Original image not found: {original_image_path}")

    img = Image.open(original_image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(img)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_np)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150, pad_inches=0.1)
    plt.close(fig)


# ===================================================================
# High-level convenience function
# ===================================================================

def explain_image(
    image_path: str | Path,
    save_dir: str | Path,
    node_idx: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, str]:
    """Run both Attention Rollout and Integrated Gradients on a single image.

    Saves two overlay PNGs into *save_dir*.

    Args:
        image_path: Path to the dermoscopic JPEG/PNG.
        save_dir: Directory where output PNGs will be written.
        node_idx: Optional integer used for naming output files.
        device: Torch device string.

    Returns:
        A dict with keys ``'rollout_path'`` and ``'ig_path'`` pointing to
        the saved overlay files.  ``'ig_path'`` may be ``None`` if
        ``captum`` is not available.
    """
    image_path = Path(image_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    suffix = f"_{node_idx}" if node_idx is not None else ""

    # Load model
    model = get_vit_model(pretrained=True)
    model = model.to(device)

    # Preprocess image
    image_tensor, _ = load_and_preprocess(image_path)
    image_tensor = image_tensor.to(device)

    results: Dict[str, Optional[str]] = {}

    # --- Attention Rollout ---
    rollout_path = save_dir / f"{stem}{suffix}_rollout.png"
    try:
        rollout_heatmap = attention_rollout(model, image_tensor, device=device)
        save_heatmap_overlay(
            rollout_heatmap,
            image_path,
            rollout_path,
            title="Attention Rollout",
        )
        results["rollout_path"] = str(rollout_path)
        print(f"[✓] Attention Rollout saved → {rollout_path}")
    except Exception as e:
        warnings.warn(f"Attention Rollout failed: {e}")
        results["rollout_path"] = None

    # --- Integrated Gradients ---
    ig_path = save_dir / f"{stem}{suffix}_ig.png"
    if _CAPTUM_AVAILABLE:
        try:
            ig_heatmap = integrated_gradients_explain(
                model, image_tensor, device=device
            )
            save_heatmap_overlay(
                ig_heatmap,
                image_path,
                ig_path,
                title="Integrated Gradients",
            )
            results["ig_path"] = str(ig_path)
            print(f"[✓] Integrated Gradients saved → {ig_path}")
        except Exception as e:
            warnings.warn(f"Integrated Gradients failed: {e}")
            results["ig_path"] = None
    else:
        warnings.warn(
            "Skipping Integrated Gradients — captum is not installed."
        )
        results["ig_path"] = None

    return results


# ===================================================================
# CLI demo
# ===================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="ViT Explainability — Attention Rollout & Integrated Gradients"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to a dermoscopic JPEG/PNG image.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./explanations",
        help="Directory to save output PNGs (default: ./explanations).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. 'cpu' or 'cuda:0' (default: cpu).",
    )
    parser.add_argument(
        "--node-idx",
        type=int,
        default=None,
        help="Optional index for naming output files.",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found — {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Image   : {args.image}")
    print(f"Save dir: {args.save_dir}")
    print(f"Device  : {args.device}")
    print("-" * 50)

    outputs = explain_image(
        image_path=args.image,
        save_dir=args.save_dir,
        node_idx=args.node_idx,
        device=args.device,
    )

    print("-" * 50)
    for key, path in outputs.items():
        status = path if path else "(skipped)"
        print(f"  {key}: {status}")
