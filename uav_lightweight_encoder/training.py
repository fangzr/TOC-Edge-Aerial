"""Training helpers for the UAV-side lightweight encoder."""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MultiViewVIB, VariationalInformationBottleneck

LOGGER = logging.getLogger("uav_lightweight_encoder")


def vib_loss_function(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


def train_single_view_vib(
    model: VariationalInformationBottleneck,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 0.1,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Training single-view VIB"):
        features = torch.tensor(batch["features"], dtype=torch.float32, device=device)
        batch_size, num_views, feat_dim = features.shape

        for view_idx in range(num_views):
            view_features = features[:, view_idx, :]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, mean, logvar, _ = model(view_features)
            loss, recon, kl = vib_loss_function(reconstructed, view_features, mean, logvar, beta)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon.item()
            running_kl += kl.item()
            steps += 1

    stats = {
        "loss": running_loss / max(steps, 1),
        "reconstruction": running_recon / max(steps, 1),
        "kl": running_kl / max(steps, 1),
    }
    LOGGER.info(
        "Single-view VIB -- loss: %.4f, recon: %.4f, kl: %.4f",
        stats["loss"],
        stats["reconstruction"],
        stats["kl"],
    )
    return stats


def train_multi_view_vib(
    model: MultiViewVIB,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 0.1,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Training multi-view VIB"):
        features = torch.tensor(batch["features"], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        reconstructed, mean, logvar, _ = model(features)
        loss, recon, kl = vib_loss_function(reconstructed, features, mean, logvar, beta)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()
        steps += 1

    stats = {
        "loss": running_loss / max(steps, 1),
        "reconstruction": running_recon / max(steps, 1),
        "kl": running_kl / max(steps, 1),
    }
    LOGGER.info(
        "Multi-view VIB -- loss: %.4f, recon: %.4f, kl: %.4f",
        stats["loss"],
        stats["reconstruction"],
        stats["kl"],
    )
    return stats
