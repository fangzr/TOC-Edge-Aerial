"""Compress multi-view features on the UAV using a trained MultiViewVIB."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FeatureDataset
from .model import MultiViewVIB

LOGGER = logging.getLogger("uav_lightweight_encoder")


def load_model(weights_path: Path, device: torch.device) -> MultiViewVIB:
    checkpoint = torch.load(weights_path, map_location=device)
    model_kwargs: Dict[str, Any] = checkpoint.get(
        "model_kwargs",
        {
            "input_dim": checkpoint.get("input_dim", 512),
            "latent_dim": checkpoint.get("latent_dim", 64),
            "num_views": checkpoint.get("num_views", 5),
            "hidden_dims": checkpoint.get("hidden_dims", (512, 256)),
        },
    )
    model = MultiViewVIB(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    LOGGER.info("Loaded model with config: %s", model_kwargs)
    return model


def compress_dataset(
    feature_dir: str,
    weights_path: str,
    output_path: str,
    cameras: Sequence[str],
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FeatureDataset(feature_dir, cameras)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    torch_device = torch.device(device)
    model = load_model(Path(weights_path), torch_device)

    for batch in tqdm(dataloader, desc="Compressing multi-view features"):
        features = torch.tensor(batch["features"], dtype=torch.float32, device=torch_device)
        latents = model.compress(features).cpu().numpy()

        positions = batch["position"]
        frame_ids = batch["frame_id"]

        for frame_id, latent, position in zip(frame_ids, latents, positions):
            np.savez_compressed(
                output_dir / f"{frame_id}.npz",
                latent=latent.astype(np.float32),
                position=np.asarray(position, dtype=np.float32),
            )

    LOGGER.info("Compressed representations stored in %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress multi-view features with MultiViewVIB.")
    parser.add_argument("--feature_dir", required=True, help="Directory that contains view_features/*/*.npz.")
    parser.add_argument("--weights", required=True, help="Path to the trained MultiViewVIB checkpoint.")
    parser.add_argument("--output", required=True, help="Directory used to store latent codes.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["Front", "Back", "Left", "Right", "Down"],
        help="Camera views expected in the dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    arguments = parse_args()
    compress_dataset(
        feature_dir=arguments.feature_dir,
        weights_path=arguments.weights,
        output_path=arguments.output,
        cameras=arguments.cameras,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
