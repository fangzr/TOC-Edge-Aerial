"""Train the edge-side multi-view localization fusion model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from edge_database_encoder.feature_database import FeatureDatabase
from edge_localization.attention_fusion import MultiViewAttentionFusion
from edge_localization.data import DEFAULT_CAMERAS, MultiViewFeatureDataset

LOGGER = logging.getLogger("edge_localization.train_fusion")


class PositionFusionLoss(nn.Module):
    """Position regression plus feature-consistency regularization."""

    def __init__(
        self,
        position_weight: float = 1.0,
        feature_weight: float = 0.1,
        attention_entropy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.position_weight = position_weight
        self.feature_weight = feature_weight
        self.attention_entropy_weight = attention_entropy_weight

    def forward(
        self,
        predicted_positions: torch.Tensor,
        true_positions: torch.Tensor,
        fused_features: torch.Tensor,
        view_features: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        position_loss = torch.norm(predicted_positions - true_positions, dim=1).mean()
        mean_view_feature = view_features.mean(dim=1)
        feature_loss = torch.norm(fused_features - mean_view_feature, dim=1).mean()
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(dim=-1).mean()
        total_loss = (
            self.position_weight * position_loss
            + self.feature_weight * feature_loss
            + self.attention_entropy_weight * attention_entropy
        )
        return total_loss, {
            "position_loss": float(position_loss.detach().cpu()),
            "feature_loss": float(feature_loss.detach().cpu()),
            "attention_entropy": float(attention_entropy.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }


def setup_logging(log_dir: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / "train_fusion.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def split_dataset(dataset: MultiViewFeatureDataset, train_ratio: float, val_ratio: float):
    total_size = len(dataset)
    if total_size == 0:
        raise RuntimeError("No complete multi-view feature samples found.")
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_ratio and val_ratio must leave non-empty train and validation splits.")
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )


def build_retrieval_database(
    dataset: MultiViewFeatureDataset,
    subsets: Sequence[Subset],
    output_path: str,
    feature_dim: int,
) -> None:
    database = FeatureDatabase(feature_dim=feature_dim)
    for subset in subsets:
        for dataset_idx in tqdm(subset.indices, desc="Building retrieval database"):
            sample = dataset[dataset_idx]
            mean_feature = np.asarray(sample["features"], dtype=np.float32).mean(axis=0)
            database.add_feature(mean_feature, sample["position"])
    database.save(output_path)
    LOGGER.info("Saved retrieval database to %s", output_path)


def save_test_split(dataset: MultiViewFeatureDataset, test_subset: Subset, output_path: str) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = [dataset.frame_ids[idx] for idx in test_subset.indices]
    with open(output_dir / "test_frame_ids.json", "w", encoding="utf-8") as handle:
        json.dump(frame_ids, handle, indent=2)
    LOGGER.info("Saved %d test frame ids to %s", len(frame_ids), output_dir)


def run_epoch(
    model: MultiViewAttentionFusion,
    dataloader: DataLoader,
    criterion: PositionFusionLoss,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "position_error": 0.0, "feature_loss": 0.0, "attention_entropy": 0.0}
    batches = 0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="Train" if is_train else "Validate"):
            features = batch["features"].to(device=device, dtype=torch.float32)
            positions = batch["position"].to(device=device, dtype=torch.float32)

            predicted_positions, fused_features, attention_weights = model(features)
            loss, components = criterion(
                predicted_positions,
                positions,
                fused_features,
                features,
                attention_weights,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            position_error = torch.norm(predicted_positions - positions, dim=1).mean()
            totals["loss"] += float(loss.detach().cpu())
            totals["position_error"] += float(position_error.detach().cpu())
            totals["feature_loss"] += components["feature_loss"]
            totals["attention_entropy"] += components["attention_entropy"]
            batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_fusion_model(
    feature_dir: str,
    model_save_path: str,
    database_output_path: str | None = None,
    test_split_output: str | None = None,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_dim: int = 512,
    num_heads: int = 4,
    max_views: int = 5,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    max_frames: int | None = None,
    device: str = "cuda",
) -> Dict[str, object]:
    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", torch_device)

    dataset = MultiViewFeatureDataset(feature_dir=feature_dir, cameras=cameras, max_frames=max_frames)
    train_subset, val_subset, test_subset = split_dataset(dataset, train_ratio, val_ratio)
    LOGGER.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_subset),
        len(val_subset),
        len(test_subset),
    )

    if database_output_path:
        build_retrieval_database(dataset, [train_subset, val_subset], database_output_path, feature_dim)
    if test_split_output:
        save_test_split(dataset, test_subset, test_split_output)

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch_device.type == "cuda",
    }
    train_loader = DataLoader(train_subset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **dataloader_kwargs)

    model = MultiViewAttentionFusion(
        feature_dim=feature_dim,
        num_heads=num_heads,
        max_views=max_views,
    ).to(torch_device)
    criterion = PositionFusionLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, torch_device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, torch_device)
        scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        LOGGER.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_pos_err=%.4fm",
            epoch,
            num_epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["position_error"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kwargs": {
                        "feature_dim": feature_dim,
                        "num_heads": num_heads,
                        "max_views": max_views,
                    },
                    "cameras": list(cameras),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                model_path,
            )
            LOGGER.info("Saved best model to %s", model_path)

    results = {"best_epoch": best_epoch, "best_val_loss": best_val_loss, "history": history}
    with open(model_path.parent / "training_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train edge-side multi-view localization fusion.")
    parser.add_argument("--feature_dir", required=True, help="Directory containing view_features/<camera>/<frame>.npz.")
    parser.add_argument("--model_save_path", required=True, help="Output path for the best fusion model checkpoint.")
    parser.add_argument("--database_output_path", default=None, help="Optional path prefix for a train/val FAISS database.")
    parser.add_argument("--test_split_output", default=None, help="Optional directory for test_frame_ids.json.")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_views", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_dir", default=None, help="Optional directory for train_fusion.log.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_dir)
    train_fusion_model(
        feature_dir=args.feature_dir,
        model_save_path=args.model_save_path,
        database_output_path=args.database_output_path,
        test_split_output=args.test_split_output,
        cameras=args.cameras,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        max_views=args.max_views,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_frames=args.max_frames,
        device=args.device,
    )
