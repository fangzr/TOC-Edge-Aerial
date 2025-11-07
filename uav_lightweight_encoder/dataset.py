"""Dataset utilities for training the lightweight encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Loads per-frame multi-view features stored in npz files."""

    def __init__(
        self,
        data_dir: str,
        cameras: Sequence[str],
        max_frames: int | None = None,
    ) -> None:
        super().__init__()
        self.cameras = list(cameras)
        self.base_dir = Path(data_dir) / "view_features"
        self.samples: List[Tuple[str, np.ndarray, np.ndarray]] = []

        frame_ids = sorted({path.stem for path in self.base_dir.glob("*/*.npz")})
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]

        for frame_id in frame_ids:
            features_per_view = []
            position = None
            valid = True

            for camera in self.cameras:
                npz_path = self.base_dir / camera / f"{frame_id}.npz"
                if not npz_path.exists():
                    valid = False
                    break
                data = np.load(npz_path)
                if "feature" not in data or "position" not in data:
                    valid = False
                    break
                features_per_view.append(np.asarray(data["feature"], dtype=np.float32))
                if position is None:
                    position = np.asarray(data["position"], dtype=np.float32)

            if valid and features_per_view and position is not None:
                stacked = np.stack(features_per_view)
                self.samples.append((frame_id, stacked, position))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame_id, features, position = self.samples[idx]
        return {
            "frame_id": frame_id,
            "features": features,
            "position": position,
        }
