"""Attention-based edge localization models."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewAttentionFusion(nn.Module):
    """Fuse synchronized multi-view features and regress a 3D position."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 4,
        max_views: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_views = max_views

        self.view_embed = nn.Linear(feature_dim, feature_dim)
        self.position_embed = nn.Parameter(torch.randn(1, max_views, feature_dim) * 0.02)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.position_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted positions, fused features, and view attention weights.

        Args:
            features: Tensor with shape ``[batch_size, num_views, feature_dim]``.
        """

        if features.ndim != 3:
            raise ValueError("features must have shape [batch_size, num_views, feature_dim]")
        _, num_views, feature_dim = features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {feature_dim}")
        if num_views > self.max_views:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.max_views}")

        embedded = self.view_embed(features) + self.position_embed[:, :num_views, :]
        attn_output, attention_weights = self.multihead_attn(
            query=embedded,
            key=embedded,
            value=embedded,
            need_weights=True,
            average_attn_weights=True,
        )
        attn_output = self.norm(attn_output + embedded)
        fused_features = attn_output.mean(dim=1)
        predicted_positions = self.position_predictor(fused_features)
        return predicted_positions, fused_features, attention_weights


class ViewWeightPredictor(nn.Module):
    """Estimate quality weights for each view feature."""

    def __init__(self, feature_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch_size, num_views, feature_dim]")
        scores = self.quality_net(features).squeeze(-1)
        return F.softmax(scores, dim=1)


class EnhancedMultiViewFusion:
    """Inference helper that exposes numpy-friendly feature fusion."""

    def __init__(
        self,
        feature_dim: int = 512,
        num_heads: int = 4,
        max_views: int = 5,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.attention_fusion = MultiViewAttentionFusion(
            feature_dim=feature_dim,
            num_heads=num_heads,
            max_views=max_views,
        ).to(self.device)
        self.weight_predictor = ViewWeightPredictor(feature_dim=feature_dim).to(self.device)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.attention_fusion.load_state_dict(state_dict)
        self.attention_fusion.eval()
        self.weight_predictor.eval()

    @torch.no_grad()
    def fuse_features(
        self,
        features: List[np.ndarray],
        return_weights: bool = False,
    ):
        features_tensor = torch.as_tensor(
            np.stack(features),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        predicted_positions, fused_features, attention_weights = self.attention_fusion(features_tensor)
        fused_np = fused_features.squeeze(0).cpu().numpy()
        position_np = predicted_positions.squeeze(0).cpu().numpy()

        if return_weights:
            quality_weights = self.weight_predictor(features_tensor).squeeze(0).cpu().numpy()
            return (
                position_np,
                fused_np,
                attention_weights.squeeze(0).cpu().numpy(),
                quality_weights,
            )
        return position_np, fused_np
