#!/usr/bin/env python3
"""
model.py
========
Defines the Multimodal Graph Attention Network (GAT) for skin
cancer lesion classification on the HAM10000 patient graph.

Architecture (v3 — High-Accuracy Multimodal GAT with Ego-Visual Highway)
-------------------------------------------------------------------------
Input (X)  →  Visual Branch (ViT 768 → 512, BN, GELU, Dropout)
           →  Metadata Branch (Clinical 19 → 128, BN, GELU, Dropout)
           →  Fusion Projection (640 → hidden*heads, BN, GELU)
           →  GATv2Conv Layer 1 (multi-head) + Residual + LayerNorm + GELU
           →  GATv2Conv Layer 2 (multi-head) + Residual + LayerNorm + GELU
           →  Ego-Visual Highway Concatenation: [Ego_ViT, GAT_1, GAT_2]
           →  Classifier MLP (256 → 7 classes)

Key Innovations for High Accuracy:
1. **Ego-Visual Highway**: Concatenates pristine ego ViT representations directly
   with graph-refined neighbor representations, preventing over-smoothing on k-NN graphs.
2. **Two-Tower Multimodal Fusion**: Balances high-dimensional visual features with
   clinical metadata (age, sex, anatomical site).
3. **Multi-Head GATv2 Attention**: Learns dynamic attention weights over neighboring
   lesions based on holistic visual and phenotypic similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MultimodalGAT(nn.Module):
    """
    High-Accuracy Multimodal Graph Attention Network for 7-class skin lesion
    classification with an Ego-Visual Highway.

    Parameters
    ----------
    visual_channels : int
        Dimensionality of ViT visual embeddings (default 768 for ViT-B/16).
    meta_channels : int
        Dimensionality of encoded clinical metadata (default 19).
    hidden_channels : int
        Width of hidden GAT layers per attention head.
    out_channels : int
        Number of output diagnostic classes (7 for HAM10000).
    heads : int
        Number of attention heads in GAT layers (default 4).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        visual_channels: int = 768,
        meta_channels: int = 19,
        hidden_channels: int = 256,
        out_channels: int = 7,
        heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.visual_channels = visual_channels
        self.meta_channels = meta_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout

        # ── 1. Two-Tower Multimodal Encoders ───────────────────
        # Visual Branch (Ego-representation)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_channels, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata Branch (Age, Sex, Site)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_channels, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion Projection for Graph Message Passing
        self.fusion_proj = nn.Sequential(
            nn.Linear(512 + 128, hidden_channels * heads),
            nn.BatchNorm1d(hidden_channels * heads),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. Graph Attention Layers ─────────────────────────
        # GAT Layer 1: multi-head attention over similarity graph
        self.gat1 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,            # output: heads * hidden_channels = 1024
        )
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        # GAT Layer 2: multi-head attention refining neighborhood context
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,            # output: heads * hidden_channels = 1024
        )
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        # ── 3. Classifier with Ego-Visual Highway ─────────────
        # Combines:
        #   - 512-d pristine ego visual features (unblurred by neighbors)
        #   - 1024-d 1-hop graph attention features
        #   - 1024-d 2-hop graph attention features
        in_clf_dim = 512 + (hidden_channels * heads) * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_clf_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels),
        )

    def forward(self, x, edge_index):
        """
        Forward pass with Ego-Visual Highway and GAT neighbor aggregation.

        Parameters
        ----------
        x : Tensor [N, visual_channels + meta_channels]
            Multimodal node feature matrix.
        edge_index : Tensor [2, E]
            k-NN similarity graph connectivity in COO format.

        Returns
        -------
        logits : Tensor [N, out_channels]
        """
        # Split multimodal features
        x_visual = x[:, :self.visual_channels]
        x_meta = x[:, self.visual_channels:]

        # Project separate modalities
        h_visual = self.visual_proj(x_visual)
        h_meta = self.meta_proj(x_meta)

        # Fuse for graph message passing
        h_fused = torch.cat([h_visual, h_meta], dim=-1)
        x0 = self.fusion_proj(h_fused)

        # GAT Layer 1 + residual connection + LayerNorm
        x1 = self.gat1(x0, edge_index) + x0
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # GAT Layer 2 + residual connection + LayerNorm
        x2 = self.gat2(x1, edge_index) + x1
        x2 = self.norm2(x2)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Ego-Visual Highway: preserve ego visual identity alongside neighbor context
        fused_rep = torch.cat([h_visual, x1, x2], dim=-1)

        # Classification
        logits = self.classifier(fused_rep)
        return logits

    def get_attention_weights(self, x, edge_index):
        """
        Run a forward pass and return attention weights from the GAT layer
        for interpretability analysis.
        """
        x_visual = x[:, :self.visual_channels]
        x_meta = x[:, self.visual_channels:]

        h_visual = self.visual_proj(x_visual)
        h_meta = self.meta_proj(x_meta)

        h_fused = torch.cat([h_visual, h_meta], dim=-1)
        x0 = self.fusion_proj(h_fused)

        x1 = self.gat1(x0, edge_index) + x0
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)

        x2, (edge_idx, attn_weights) = self.gat2(
            x1, edge_index, return_attention_weights=True
        )
        x2 = x2 + x1
        fused_rep = torch.cat([h_visual, x1, x2], dim=-1)
        return fused_rep, edge_idx, attn_weights
