"""CLI for training single-view or multi-view VIB encoders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import FeatureDataset
from .model import MultiViewVIB, VariationalInformationBottleneck
from .training import train_multi_view_vib, train_single_view_vib

LOGGER = logging.getLogger("uav_lightweight_encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VIB encoders for UAV-side compression.")
    parser.add_argument("--feature_dir", required=True, help="Directory that stores view_features/*/*.npz.")
    parser.add_argument("--output", required=True, help="Path to save the trained weights.")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi", help="Training mode.")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1, help="Weight for the KL term.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["Front", "Back", "Left", "Right", "Down"],
        help="Camera views to include.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on the number of frames.")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    state_dict: dict,
    model_kwargs: dict,
) -> None:
    torch.save(
        {
            "state_dict": state_dict,
            "model_kwargs": model_kwargs,
        },
        path,
    )
    LOGGER.info("Checkpoint saved to %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset = FeatureDataset(args.feature_dir, args.cameras, args.max_frames)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Ensure view_features exist.")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device(args.device)

    if args.mode == "single":
        model_kwargs = {
            "input_dim": dataset[0]["features"].shape[-1],
            "latent_dim": args.latent_dim,
            "hidden_dims": tuple(args.hidden_dims),
        }
        model = VariationalInformationBottleneck(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            LOGGER.info("Epoch %d/%d", epoch + 1, args.epochs)
            train_single_view_vib(model, dataloader, optimizer, device, beta=args.beta)

    else:
        model_kwargs = {
            "input_dim": dataset[0]["features"].shape[-1],
            "latent_dim": args.latent_dim,
            "num_views": len(args.cameras),
            "hidden_dims": tuple(args.hidden_dims),
        }
        model = MultiViewVIB(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            LOGGER.info("Epoch %d/%d", epoch + 1, args.epochs)
            train_multi_view_vib(model, dataloader, optimizer, device, beta=args.beta)

    save_checkpoint(Path(args.output), model.state_dict(), model_kwargs)


if __name__ == "__main__":
    main()
