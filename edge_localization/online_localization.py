"""Run edge-side online localization from multi-view feature files."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from edge_database_encoder.feature_database import FeatureDatabase
from edge_localization.attention_fusion import MultiViewAttentionFusion
from edge_localization.data import DEFAULT_CAMERAS, find_feature_file, load_feature_file, resolve_view_features_dir
from edge_localization.visualization import build_visualizer

LOGGER = logging.getLogger("edge_localization.online_localization")


def position_to_dict(position) -> Dict[str, float]:
    if isinstance(position, dict):
        return {"x": float(position["x"]), "y": float(position["y"]), "z": float(position["z"])}
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: json_ready(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_ready(value) for value in obj]
    if isinstance(obj, tuple):
        return [json_ready(value) for value in obj]
    return obj


class OnlineLocalizer:
    """Load an edge fusion model and estimate positions from feature dictionaries."""

    def __init__(
        self,
        model_path: str,
        database_path: str | None = None,
        cameras: Sequence[str] | None = None,
        feature_dim: int = 512,
        num_heads: int = 4,
        max_views: int = 5,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_kwargs = checkpoint.get(
            "model_kwargs",
            {"feature_dim": feature_dim, "num_heads": num_heads, "max_views": max_views},
        )
        self.cameras = list(cameras or checkpoint.get("cameras", DEFAULT_CAMERAS))
        self.fusion_model = MultiViewAttentionFusion(**model_kwargs).to(self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.fusion_model.load_state_dict(state_dict)
        self.fusion_model.eval()

        self.database = FeatureDatabase.load(database_path) if database_path else None
        if self.database is not None:
            LOGGER.info("Loaded retrieval database with %d entries", len(self.database))

    @torch.no_grad()
    def process_frame(self, features_dict: Dict[str, np.ndarray], k: int = 3, return_attention: bool = False):
        features = []
        for camera in self.cameras:
            if camera not in features_dict:
                raise ValueError(f"Missing feature for camera {camera}")
            features.append(np.asarray(features_dict[camera], dtype=np.float32))

        features_tensor = torch.as_tensor(
            np.stack(features),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        predicted_positions, fused_features, attention_weights = self.fusion_model(features_tensor)
        predicted_position = position_to_dict(predicted_positions.squeeze(0).cpu().numpy())

        estimated_positions = [predicted_position]
        distances = [0.0]
        if self.database is not None and k > 1:
            db_distances, db_coordinates = self.database.search(fused_features.squeeze(0).cpu().numpy(), k=k - 1)
            estimated_positions.extend(position_to_dict(coord) for coord in db_coordinates)
            distances.extend(float(distance) for distance in db_distances)

        if return_attention:
            return estimated_positions, distances, attention_weights.squeeze(0).cpu().numpy()
        return estimated_positions, distances, None


def load_frame_features(view_features_dir: Path, frame_id: str, cameras: Sequence[str]):
    features: Dict[str, np.ndarray] = {}
    gt_position = None
    for camera in cameras:
        feature_path = find_feature_file(view_features_dir, camera, frame_id)
        if feature_path is None:
            raise FileNotFoundError(f"Missing feature for camera={camera}, frame={frame_id}")
        feature, position = load_feature_file(feature_path)
        features[camera] = feature
        if gt_position is None:
            gt_position = position
    if gt_position is None:
        raise RuntimeError(f"No ground-truth position for frame {frame_id}")
    return features, position_to_dict(gt_position)


def load_frame_ids(test_data_path: Path, view_features_dir: Path, cameras: Sequence[str], max_frames: int | None) -> list[str]:
    ids_path = test_data_path / "test_frame_ids.json"
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as handle:
            frame_ids = json.load(handle)
    else:
        first_camera_dir = view_features_dir / cameras[0]
        frame_ids = sorted({path.stem for path in first_camera_dir.glob("*.npz")})
        frame_ids.extend(path.stem for path in first_camera_dir.glob("*.npy") if path.stem not in frame_ids)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def evaluate_localization(
    localizer: OnlineLocalizer,
    test_data_path: str,
    output_path: str | None = None,
    max_frames: int | None = None,
    k: int = 3,
    bev_image_path: str | None = None,
    bev_coords_path: str | None = None,
    vis_scale: float = 1.0,
):
    test_root = Path(test_data_path)
    view_features_dir = resolve_view_features_dir(test_root)
    output_dir = Path(output_path) if output_path else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = build_visualizer(bev_image_path, bev_coords_path, vis_scale)
    visualization_dir = None
    if visualizer is not None and output_dir is not None:
        visualization_dir = output_dir / "visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = load_frame_ids(test_root, view_features_dir, localizer.cameras, max_frames)
    results = []
    position_errors = []

    for frame_id in tqdm(frame_ids, desc="Evaluating localization"):
        try:
            features, gt_position = load_frame_features(view_features_dir, frame_id, localizer.cameras)
            start_time = time.perf_counter()
            estimated_positions, distances, attention_weights = localizer.process_frame(
                features,
                k=k,
                return_attention=True,
            )
            processing_time = time.perf_counter() - start_time

            gt_array = np.asarray([gt_position["x"], gt_position["y"], gt_position["z"]], dtype=np.float32)
            pred = estimated_positions[0]
            pred_array = np.asarray([pred["x"], pred["y"], pred["z"]], dtype=np.float32)
            position_error = float(np.linalg.norm(gt_array - pred_array))
            position_errors.append(position_error)

            if visualizer is not None and visualization_dir is not None:
                image = visualizer.visualize_frame(gt_position, estimated_positions, title=f"Frame {frame_id}")
                cv2.imwrite(str(visualization_dir / f"{frame_id}_viz.png"), image)

            results.append(
                {
                    "frame_id": frame_id,
                    "ground_truth": gt_position,
                    "estimated_positions": estimated_positions,
                    "position_error": position_error,
                    "confidence_scores": distances,
                    "processing_time": processing_time,
                    "attention_weights": attention_weights,
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue evaluating remaining frames.
            LOGGER.warning("Skipping frame %s: %s", frame_id, exc)

    if position_errors:
        thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = {
            "mean_position_error": float(np.mean(position_errors)),
            "median_position_error": float(np.median(position_errors)),
            "std_position_error": float(np.std(position_errors)),
            "total_frames": len(results),
            "total_frames_attempted": len(frame_ids),
            "success_rate": float(len(results) / len(frame_ids)) if frame_ids else 0.0,
            **{f"accuracy_{threshold}m": float(np.mean(np.asarray(position_errors) <= threshold)) for threshold in thresholds},
        }
    else:
        summary = {
            "mean_position_error": float("nan"),
            "median_position_error": float("nan"),
            "std_position_error": float("nan"),
            "total_frames": 0,
            "total_frames_attempted": len(frame_ids),
            "success_rate": 0.0,
        }

    if output_dir:
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(summary), handle, indent=2)
        with open(output_dir / "full_results.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(results), handle, indent=2)

    return summary, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate edge-side online UAV localization.")
    parser.add_argument("--model_path", required=True, help="Path to the trained fusion model checkpoint.")
    parser.add_argument("--test_data_path", required=True, help="Directory with view_features and optional test_frame_ids.json.")
    parser.add_argument("--database_path", default=None, help="Optional FAISS database path prefix for retrieval candidates.")
    parser.add_argument("--output_path", default=None, help="Optional directory for JSON results and visualizations.")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_views", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--bev_image_path", default=None, help="Optional BEV image path for visualization.")
    parser.add_argument("--bev_coords_path", default=None, help="Optional BEV coordinate .npy path for visualization.")
    parser.add_argument("--vis_scale", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = parse_args()
    localizer = OnlineLocalizer(
        model_path=args.model_path,
        database_path=args.database_path,
        cameras=args.cameras,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        max_views=args.max_views,
        device=args.device,
    )
    summary, _ = evaluate_localization(
        localizer=localizer,
        test_data_path=args.test_data_path,
        output_path=args.output_path,
        max_frames=args.max_frames,
        k=args.top_k,
        bev_image_path=args.bev_image_path,
        bev_coords_path=args.bev_coords_path,
        vis_scale=args.vis_scale,
    )
    print(json.dumps(json_ready(summary), indent=2))
