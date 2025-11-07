"""Variational Information Bottleneck models used on the UAV."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """MLP encoder that predicts mean and log-variance."""

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev_dim = dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, latent_dim)
        self.logvar_head = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.logvar_head(h)


class DecoderNetwork(nn.Module):
    """MLP decoder that reconstructs features from latent codes."""

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 512,
        hidden_dims: Sequence[int] = (128, 256),
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.BatchNorm1d(dim)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.backbone(z)


class VariationalInformationBottleneck(nn.Module):
    """Single-view VIB module."""

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = EncoderNetwork(input_dim, latent_dim, hidden_dims)
        self.decoder = DecoderNetwork(latent_dim, input_dim, tuple(reversed(hidden_dims)))

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        z = self._reparameterize(mean, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mean, logvar, z

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encoder(x)
        return self._reparameterize(mean, logvar)

    @torch.no_grad()
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class MultiViewVIB(nn.Module):
    """Joint encoder across multiple synchronized camera views."""

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        num_views: int = 5,
        hidden_dims: Sequence[int] = (512, 256),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_views = num_views
        self.encoder = EncoderNetwork(input_dim * num_views, latent_dim, hidden_dims)
        decoder_dims = tuple(reversed(hidden_dims))
        self.decoders = nn.ModuleList(
            [DecoderNetwork(latent_dim, input_dim, decoder_dims) for _ in range(num_views)]
        )

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_views: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_views.shape[0]
        flattened = x_views.reshape(batch_size, -1)
        mean, logvar = self.encoder(flattened)
        z = self._reparameterize(mean, logvar)
        recon_views = torch.stack([decoder(z) for decoder in self.decoders], dim=1)
        return recon_views, mean, logvar, z

    @torch.no_grad()
    def compress(self, x_views: torch.Tensor) -> torch.Tensor:
        batch_size = x_views.shape[0]
        flattened = x_views.reshape(batch_size, -1)
        mean, logvar = self.encoder(flattened)
        return self._reparameterize(mean, logvar)

    @torch.no_grad()
    def decompress(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([decoder(z) for decoder in self.decoders], dim=1)
