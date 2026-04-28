"""Optional bird's-eye-view visualization for localization results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


class BEVPathVisualizer:
    """Project CARLA world coordinates onto a precomputed BEV map."""

    def __init__(self, bev_image_path: str, coords_path: str, vis_scale: float = 1.0) -> None:
        self.bev_image = cv2.imread(str(bev_image_path))
        if self.bev_image is None:
            raise ValueError(f"Could not load BEV image: {bev_image_path}")

        self.xy_coords = np.load(str(coords_path))
        if self.xy_coords.ndim != 3 or self.xy_coords.shape[2] != 2:
            raise ValueError(f"Invalid coordinate map shape in {coords_path}")

        self.image_height, self.image_width = self.bev_image.shape[:2]
        self.vis_scale = vis_scale
        self.x_min = float(np.min(self.xy_coords[:, :, 0]))
        self.x_max = float(np.max(self.xy_coords[:, :, 0]))
        self.y_min = float(np.min(self.xy_coords[:, :, 1]))
        self.y_max = float(np.max(self.xy_coords[:, :, 1]))

    @staticmethod
    def _xy(position) -> tuple[float, float]:
        if isinstance(position, dict):
            return float(position["x"]), float(position["y"])
        return float(position[0]), float(position[1])

    def world_to_image(self, position) -> tuple[int, int]:
        x, y = self._xy(position)
        x_norm = (x - self.x_min) / max(self.x_max - self.x_min, 1e-6)
        y_norm = (y - self.y_min) / max(self.y_max - self.y_min, 1e-6)
        x_img = int(y_norm * (self.image_width - 1))
        y_img = int((1.0 - x_norm) * (self.image_height - 1))
        return (
            int(np.clip(x_img, 0, self.image_width - 1)),
            int(np.clip(y_img, 0, self.image_height - 1)),
        )

    def _scaled(self, value: int) -> int:
        return max(1, int(value * self.vis_scale))

    def draw_position(self, image, position, color, label: str, radius: int) -> None:
        x_img, y_img = self.world_to_image(position)
        cv2.circle(image, (x_img, y_img), self._scaled(radius), color, -1)
        cv2.putText(
            image,
            label,
            (x_img + self._scaled(8), y_img - self._scaled(8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * self.vis_scale,
            color,
            self._scaled(2),
            cv2.LINE_AA,
        )

    def visualize_frame(self, gt_position, predicted_positions: Sequence, title: str | None = None):
        image = self.bev_image.copy()
        self.draw_position(image, gt_position, (0, 255, 0), "GT", 8)
        for idx, position in enumerate(predicted_positions):
            label = "Pred" if idx == 0 else f"Top{idx + 1}"
            color = (0, 0, 255) if idx == 0 else (128, 128, 255)
            self.draw_position(image, position, color, label, 8 if idx == 0 else 5)

        if title:
            cv2.putText(
                image,
                title,
                (self._scaled(24), self._scaled(36)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * self.vis_scale,
                (255, 255, 255),
                self._scaled(2),
                cv2.LINE_AA,
            )
        return image


def build_visualizer(bev_image_path: str | None, bev_coords_path: str | None, vis_scale: float):
    if not bev_image_path and not bev_coords_path:
        return None
    if not bev_image_path or not bev_coords_path:
        raise ValueError("Both --bev_image_path and --bev_coords_path are required for visualization.")
    if not Path(bev_image_path).exists():
        raise FileNotFoundError(bev_image_path)
    if not Path(bev_coords_path).exists():
        raise FileNotFoundError(bev_coords_path)
    return BEVPathVisualizer(bev_image_path, bev_coords_path, vis_scale=vis_scale)
