"""Lightweight FAISS-backed feature database."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np


class FeatureDatabase:
    """Store CLIP features with geo-coordinates for fast retrieval."""

    def __init__(self, feature_dim: int = 512) -> None:
        self.feature_dim = feature_dim
        self.index = faiss.IndexFlatL2(feature_dim)
        self.coordinates: List[Tuple[float, float, float]] = []

    def add_feature(self, feature: np.ndarray, coordinate: Sequence[float]) -> None:
        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        self.index.add(feature)
        self.coordinates.append(tuple(map(float, coordinate)))

    def batch_add(
        self,
        features: np.ndarray,
        coordinates: Sequence[Sequence[float]],
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        self.index.add(features)
        self.coordinates.extend([tuple(map(float, coord)) for coord in coordinates])

    def search(
        self,
        query_feature: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
        query_feature = np.asarray(query_feature, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_feature, k)
        coords = [self.coordinates[idx] for idx in indices[0]]
        return distances[0], coords

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path_obj.with_suffix(".index")))
        with open(path_obj.with_suffix(".coords"), "wb") as handle:
            pickle.dump(self.coordinates, handle)

    @classmethod
    def load(cls, path: str) -> "FeatureDatabase":
        instance = cls()
        path_obj = Path(path)
        instance.index = faiss.read_index(str(path_obj.with_suffix(".index")))
        with open(path_obj.with_suffix(".coords"), "rb") as handle:
            instance.coordinates = pickle.load(handle)
        return instance

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.index.ntotal
