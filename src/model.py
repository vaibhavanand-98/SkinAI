#!/usr/bin/env python3
"""
model.py
========
Defines the Multimodal Graph Attention Network (GAT) for skin
cancer lesion classification on the HAM10000 patient graph.

Architecture (v2 — with residual connections & layer norm)
----------------------------------------------------------
Input (X)  →  Linear Projection (in → 512)  →  BN + ELU + Dropout
           →  GATConv (multi-head) + Residual  →  LayerNorm + ELU + Dropout
           →  GATConv (multi-head) + Residual  →  LayerNorm + ELU + Dropout
           →  GATConv (single-head)             →  LayerNorm + ELU
           →  Linear Classifier  →  7-class logits

Improvements over v1
---------------------
- Residual (skip) connections between GAT layers for better gradient flow.
- LayerNorm after each GAT layer for training stability.
- Separate `feat_dropout` (on features) and `attn_dropout` (on attention
  coefficients) for finer control.

The model expects the full multimodal node feature matrix
[N, D_visual + D_meta] and edge_index [2, E] produced by
data_prep.build_graph().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, JumpingKnowledge


class MultimodalGAT(nn.Module):
    """
    3-layer Graph Attention Network for 7-class skin lesion
    classification, with residual connections and layer normalisation.

    Parameters
    ----------
    in_channels : int
        Dimensionality of each node's multimodal feature vector.
    hidden_channels : int
        Width of hidden GAT layers (per attention head).
    out_channels : int
        Number of output classes (default 7 for HAM10000).
    heads : int
        Number of attention heads in the first two GAT layers.
    dropout : float
        Dropout probability applied after each GAT layer.
    """

    def __init__(
        self,
        visual_channels: int = 2048,
        meta_channels: int = 19,
        hidden_channels: int = 256,
        out_channels: int = 7,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.visual_channels = visual_channels
        self.meta_channels = meta_channels
        self.hidden_channels = hidden_channels
        self.heads = heads

        # ── Two-Tower Architecture for Multimodal Fusion ──
        # 1. Visual Branch
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_channels, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # 2. Metadata Branch (boosts importance of 19 features)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_channels, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # 3. Fusion Layer
        self.fusion_proj = nn.Sequential(
            nn.Linear(512 + 128, hidden_channels * heads),
            nn.BatchNorm1d(hidden_channels * heads),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── GAT layers ─────────────────────────────────────────
        # Layer 1: multi-head attention
        self.gat1 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,            # output = heads * hidden_channels
        )
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        # Layer 2: multi-head attention
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        # Layer 3: single-head (for classification)
        self.gat3 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )
        self.norm3 = nn.LayerNorm(hidden_channels)

        # ── Residual projection for skip connection from GAT block to classifier
        # gat3 output is [N, hidden_channels], we project from heads*hidden to hidden
        self.residual_proj = nn.Linear(hidden_channels * heads, hidden_channels)

        self.jk = JumpingKnowledge("cat")

        # ── Classifier head ───────────────────────────────────
        jk_dim = (hidden_channels * heads) * 2 + hidden_channels
        self.classifier = nn.Sequential(
            nn.Linear(jk_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_channels),
        )

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass with Two-Tower fusion.

        Parameters
        ----------
        x : Tensor [N, visual_channels + meta_channels]
            Multimodal node feature matrix.
        edge_index : Tensor [2, E]
            Graph connectivity in COO format.

        Returns
        -------
        logits : Tensor [N, out_channels]
        """
        # Split multimodal features
        x_visual = x[:, :self.visual_channels]
        x_meta = x[:, self.visual_channels:]

        # Project separately
        h_visual = self.visual_proj(x_visual)
        h_meta = self.meta_proj(x_meta)

        # Fuse
        h_fused = torch.cat([h_visual, h_meta], dim=-1)
        x = self.fusion_proj(h_fused)

        # GAT Layer 1 + residual
        identity = x
        x1 = self.gat1(x, edge_index)
        x1 = x1 + identity                     # residual connection
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # GAT Layer 2 + residual
        identity = x1
        x2 = self.gat2(x1, edge_index)
        x2 = x2 + identity                     # residual connection
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # GAT Layer 3 + residual (with projection for dim mismatch)
        identity = self.residual_proj(x2)      # [N, hidden]
        x3 = self.gat3(x2, edge_index)          # [N, hidden]
        x3 = x3 + identity                      # residual connection
        x3 = self.norm3(x3)
        x3 = F.elu(x3)

        # Jumping Knowledge
        x_jk = self.jk([x1, x2, x3])

        # Classification
        logits = self.classifier(x_jk)
        return logits

    def get_attention_weights(self, x, edge_index):
        """
        Run a forward pass and return attention weights from the
        final GAT layer (useful for interpretability).
        """
        # Split multimodal features
        x_visual = x[:, :self.visual_channels]
        x_meta = x[:, self.visual_channels:]

        # Project separately
        h_visual = self.visual_proj(x_visual)
        h_meta = self.meta_proj(x_meta)

        # Fuse
        h_fused = torch.cat([h_visual, h_meta], dim=-1)
        x = self.fusion_proj(h_fused)

        identity = x
        x1 = self.gat1(x, edge_index)
        x1 = x1 + identity
        x1 = self.norm1(x1)
        x1 = F.elu(x1)

        identity = x1
        x2 = self.gat2(x1, edge_index)
        x2 = x2 + identity
        x2 = self.norm2(x2)
        x2 = F.elu(x2)

        # Return attention weights from gat3
        identity = self.residual_proj(x2)
        x3, (edge_idx, attn_weights) = self.gat3(
            x2, edge_index, return_attention_weights=True
        )
        x3 = x3 + identity
        x_jk = self.jk([x1, x2, x3])
        return x_jk, edge_idx, attn_weights
