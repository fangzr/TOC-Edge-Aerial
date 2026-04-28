"""Shared data loading helpers for edge localization."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

DEFAULT_CAMERAS = ["Front", "Back", "Left", "Right", "Down"]


def _extract_position(raw_position) -> np.ndarray:
    if isinstance(raw_position, dict):
        return np.asarray(
            [raw_position["x"], raw_position["y"], raw_position["z"]],
            dtype=np.float32,
        )
    return np.asarray(raw_position, dtype=np.float32)


def load_feature_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one feature file from the public ``.npz`` or legacy ``.npy`` format."""

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            feature = np.asarray(data["feature"], dtype=np.float32)
            position = _extract_position(data["position"])
            return feature, position

    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()
        if isinstance(data, dict):
            return np.asarray(data["feature"], dtype=np.float32), _extract_position(data["position"])
        raise ValueError(f"Unsupported .npy feature payload in {path}")

    raise ValueError(f"Unsupported feature file extension: {path.suffix}")


def resolve_view_features_dir(feature_dir: str | Path) -> Path:
    root = Path(feature_dir)
    view_dir = root / "view_features"
    return view_dir if view_dir.exists() else root


def find_feature_file(view_features_dir: Path, camera: str, frame_id: str) -> Path | None:
    for suffix in (".npz", ".npy"):
        candidate = view_features_dir / camera / f"{frame_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_frame_ids(view_features_dir: Path, cameras: Sequence[str]) -> List[str]:
    first_camera_dir = view_features_dir / cameras[0]
    frame_ids = {path.stem for path in first_camera_dir.glob("*.npz")}
    frame_ids.update(path.stem for path in first_camera_dir.glob("*.npy"))

    valid_frame_ids: List[str] = []
    for frame_id in sorted(frame_ids):
        if all(find_feature_file(view_features_dir, camera, frame_id) for camera in cameras):
            valid_frame_ids.append(frame_id)
    return valid_frame_ids


class MultiViewFeatureDataset(Dataset):
    """Dataset of complete synchronized multi-view feature tensors."""

    def __init__(
        self,
        feature_dir: str,
        cameras: Sequence[str] | None = None,
        frame_ids: Sequence[str] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.cameras = list(cameras or DEFAULT_CAMERAS)
        self.view_features_dir = resolve_view_features_dir(feature_dir)
        discovered = list(frame_ids) if frame_ids is not None else discover_frame_ids(self.view_features_dir, self.cameras)
        if max_frames is not None:
            discovered = discovered[:max_frames]
        self.frame_ids = discovered

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        frame_id = self.frame_ids[idx]
        features = []
        position = None

        for camera in self.cameras:
            feature_path = find_feature_file(self.view_features_dir, camera, frame_id)
            if feature_path is None:
                raise FileNotFoundError(f"Missing feature for camera={camera}, frame={frame_id}")
            feature, camera_position = load_feature_file(feature_path)
            features.append(feature)
            if position is None:
                position = camera_position

        if position is None:
            raise RuntimeError(f"No position found for frame {frame_id}")

        return {
            "frame_id": frame_id,
            "features": np.stack(features).astype(np.float32),
            "position": position.astype(np.float32),
        }
