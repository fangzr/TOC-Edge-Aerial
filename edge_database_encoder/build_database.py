"""Build a geo-referenced feature database on the edge server."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .clip_encoder import CLIPFeatureExtractor
from .dataset_stats import DatasetStatistics
from .feature_database import FeatureDatabase


LOGGER = logging.getLogger("edge_database_encoder")


def setup_logging(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "database_building.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def _load_coordinate(metadata_path: Path) -> Sequence[float]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    position = metadata["uav_state"]["position"]
    if isinstance(position, dict):
        return [position["x"], position["y"], position["z"]]
    return position


def extract_and_save_features(
    dataset_path: str,
    output_path: str,
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    batch_size: int = 64,
    max_frames: Optional[int] = None,
    cameras: Optional[Iterable[str]] = None,
) -> Path:
    """Extract CLIP features per frame and build a FAISS database."""

    dataset_root = Path(dataset_path)
    output_root = Path(output_path)

    setup_logging(output_root)
    LOGGER.info("Building database from %s", dataset_root)

    camera_names = list(cameras) if cameras is not None else [
        "Front",
        "Back",
        "Left",
        "Right",
        "Down",
    ]

    extractor = CLIPFeatureExtractor(model_name=model_name, device=device)
    database = FeatureDatabase(feature_dim=512)

    stats = DatasetStatistics(dataset_path).compute_rgb_stats(camera_names)
    LOGGER.info("RGB statistics: mean=%s std=%s", stats["mean"], stats["std"])

    metadata_dir = dataset_root / "metadata"
    frame_ids = sorted(path.stem for path in metadata_dir.glob("*.json"))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
        LOGGER.info("Limiting to first %d frames", len(frame_ids))
    else:
        LOGGER.info("Processing %d frames", len(frame_ids))

    view_feature_dir = output_root / "view_features"
    for camera in camera_names:
        (view_feature_dir / camera).mkdir(parents=True, exist_ok=True)

    batch_features: list[np.ndarray] = []
    batch_coordinates: list[Sequence[float]] = []

    for frame_id in tqdm(frame_ids, desc="Extracting features"):
        metadata_path = metadata_dir / f"{frame_id}.json"
        if not metadata_path.exists():
            LOGGER.warning("Missing metadata for frame %s, skipping", frame_id)
            continue

        coordinate = _load_coordinate(metadata_path)
        frame_features: dict[str, np.ndarray] = {}

        for camera in camera_names:
            rgb_path = dataset_root / "rgb" / camera / f"{frame_id}.png"
            if not rgb_path.exists():
                LOGGER.warning("Missing RGB image for frame %s camera %s", frame_id, camera)
                frame_features = {}
                break

            image = Image.open(rgb_path)
            feature = extractor.extract_features(image)
            frame_features[camera] = feature

            np.savez_compressed(
                view_feature_dir / camera / f"{frame_id}.npz",
                feature=feature.astype(np.float32),
                position=np.asarray(coordinate, dtype=np.float32),
            )

        if len(frame_features) != len(camera_names):
            continue

        avg_feature = np.mean(list(frame_features.values()), axis=0)
        batch_features.append(avg_feature)
        batch_coordinates.append(coordinate)

        if len(batch_features) >= batch_size:
            database.batch_add(np.stack(batch_features), batch_coordinates)
            batch_features.clear()
            batch_coordinates.clear()

    if batch_features:
        database.batch_add(np.stack(batch_features), batch_coordinates)

    database_path = output_root / "feature_database"
    database.save(str(database_path))

    config_data = {
        "model_name": model_name,
        "feature_dim": 512,
        "num_frames": len(frame_ids),
        "cameras": camera_names,
        "rgb_stats": stats,
    }
    with open(output_root / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config_data, handle, indent=2)

    LOGGER.info("Database saved to %s", database_path)
    return database_path


def validate_database(database_path: str, sample_size: int = 32) -> bool:
    """Sample queries to ensure the persisted FAISS index is consistent."""

    LOGGER.info("Validating database at %s", database_path)
    database = FeatureDatabase.load(database_path)
    total = len(database)
    if total == 0:
        raise RuntimeError("Cannot validate an empty database.")

    sample_indices = np.random.choice(total, min(sample_size, total), replace=False)

    for idx in sample_indices:
        feature = database.index.reconstruct(int(idx))
        distances, coords = database.search(feature, k=1)
        if distances[0] > 1e-5:
            LOGGER.error(
                "Feature %d failed validation: distance %.6f for coordinate %s",
                idx,
                float(distances[0]),
                coords[0],
            )
            return False

    LOGGER.info("Database validation passed.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the edge-side feature database.")
    parser.add_argument("--dataset_path", required=True, help="Path to the collected dataset.")
    parser.add_argument("--output_path", required=True, help="Directory used to store the database.")
    parser.add_argument("--model_name", default="ViT-B/32", help="CLIP vision backbone.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of averaged frames per FAISS update.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame budget for quick experiments.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    db_path = extract_and_save_features(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
    )
    validate_database(str(db_path))
