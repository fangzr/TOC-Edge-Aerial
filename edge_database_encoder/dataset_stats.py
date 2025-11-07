"""Dataset statistics utilities for RGB streams."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


class DatasetStatistics:
    """Compute basic RGB statistics that are stored alongside the database."""

    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = Path(dataset_path)

    def compute_rgb_stats(
        self,
        cameras: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[float]]:
        cameras = list(cameras) if cameras is not None else [
            "Front",
            "Back",
            "Left",
            "Right",
            "Down",
        ]

        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []

        for camera in cameras:
            camera_dir = self.dataset_path / "rgb" / camera
            if not camera_dir.exists():
                continue

            for image_path in camera_dir.glob("*.png"):
                img = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
                means.append(img.mean(axis=(0, 1)))
                stds.append(img.std(axis=(0, 1)))

        if not means:
            raise ValueError("No RGB frames were found for the requested cameras.")

        mean_vector = np.mean(means, axis=0).tolist()
        std_vector = np.mean(stds, axis=0).tolist()
        return {"mean": mean_vector, "std": std_vector}
