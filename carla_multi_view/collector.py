"""Synchronous CARLA multi-view data collection pipeline."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, Iterator, Tuple

import carla  # type: ignore
import cv2
import numpy as np

LOGGER = logging.getLogger("carla_multi_view")


def _semantic_palette() -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[1] = [70, 70, 70]
    palette[2] = [190, 153, 153]
    palette[3] = [180, 220, 135]
    palette[5] = [153, 153, 153]
    palette[6] = [255, 255, 255]
    palette[7] = [128, 64, 128]
    palette[8] = [244, 35, 232]
    palette[9] = [107, 142, 35]
    palette[11] = [102, 102, 156]
    palette[12] = [220, 220, 0]
    palette[13] = [70, 130, 180]
    palette[14] = [81, 0, 81]
    palette[15] = [150, 100, 100]
    palette[16] = [230, 150, 140]
    palette[17] = [180, 165, 180]
    palette[18] = [250, 170, 30]
    palette[19] = [110, 190, 160]
    palette[20] = [111, 74, 0]
    palette[21] = [45, 60, 150]
    palette[22] = [152, 251, 152]
    palette[40] = [220, 20, 60]
    palette[41] = [255, 0, 0]
    palette[100] = [0, 0, 142]
    palette[101] = [0, 0, 70]
    palette[102] = [0, 60, 100]
    palette[103] = [0, 80, 100]
    palette[104] = [0, 0, 230]
    palette[105] = [119, 11, 32]
    return palette


@dataclass
class CameraConfig:
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    fov: float = 90.0
    image_size: Tuple[int, int] = (400, 300)
    modalities: Tuple[str, ...] = ("rgb", "depth", "semantic")


def default_camera_configs() -> Dict[str, CameraConfig]:
    return {
        "Front": CameraConfig(position=(2.0, 0.0, 0.0), rotation=(-15, 0.0, 0.0)),
        "Back": CameraConfig(position=(-2.0, 0.0, 0.0), rotation=(-15, 180.0, 0.0)),
        "Left": CameraConfig(position=(0.0, -2.0, 0.0), rotation=(-15, -90.0, 0.0)),
        "Right": CameraConfig(position=(0.0, 2.0, 0.0), rotation=(-15, 90.0, 0.0)),
        "Down": CameraConfig(position=(0.0, 0.0, -1.0), rotation=(-90, 0.0, 0.0)),
    }


@dataclass
class CollectorSettings:
    server_host: str = "localhost"
    server_port: int = 2000
    map_name: str = "Town12"
    altitude: float = 30.0
    frequency_hz: float = 10.0
    sensor_timeout: float = 0.4
    output_dir: Path = field(default_factory=lambda: Path("./datasets"))
    sample_distance: float = 15.0
    cameras: Dict[str, CameraConfig] = field(default_factory=default_camera_configs)


class CameraRig:
    """Manage CARLA sensors for all camera views."""

    def __init__(self, world: carla.World, settings: CollectorSettings) -> None:
        self.world = world
        self.settings = settings
        self.sensor_queues: Dict[Tuple[str, str], Queue] = {}
        self.sensors = defaultdict(dict)
        self.palette = _semantic_palette()
        self._spawn_sensors()

    def _spawn_sensors(self) -> None:
        library = self.world.get_blueprint_library()
        for name, cfg in self.settings.cameras.items():
            for modality in cfg.modalities:
                blueprint = self._blueprint_for_modality(library, modality)
                width, height = cfg.image_size
                blueprint.set_attribute("image_size_x", str(width))
                blueprint.set_attribute("image_size_y", str(height))
                blueprint.set_attribute("fov", str(cfg.fov))
                blueprint.set_attribute("sensor_tick", str(1.0 / self.settings.frequency_hz))

                transform = carla.Transform(
                    carla.Location(*cfg.position),
                    carla.Rotation(*cfg.rotation),
                )
                sensor = self.world.spawn_actor(blueprint, transform)
                queue: Queue = Queue(maxsize=2)
                self.sensor_queues[(name, modality)] = queue
                sensor.listen(
                    lambda data, cam=name, mod=modality: self._sensor_callback(cam, mod, data)
                )

                self.sensors[name][modality] = sensor

    def _blueprint_for_modality(self, library, modality: str) -> carla.ActorBlueprint:
        if modality == "rgb":
            return library.find("sensor.camera.rgb")
        if modality == "depth":
            return library.find("sensor.camera.depth")
        if modality == "semantic":
            return library.find("sensor.camera.semantic_segmentation")
        raise ValueError(f"Unknown modality: {modality}")

    def _sensor_callback(self, camera: str, modality: str, image: carla.Image) -> None:
        array: np.ndarray

        if modality == "rgb":
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))[:, :, :3]
        elif modality == "depth":
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            normalized = (
                raw[:, :, 2].astype(np.float32)
                + raw[:, :, 1].astype(np.float32) * 256.0
                + raw[:, :, 0].astype(np.float32) * 65536.0
            ) / (256**3 - 1)
            array = normalized * 1000.0
        else:
            raw = np.frombuffer(image.raw_data, dtype=np.uint8)
            raw = raw.reshape((image.height, image.width, 4))
            labels = raw[:, :, 2]
            array = self.palette[labels]

        payload = {
            "frame": image.frame,
            "timestamp": image.timestamp,
            "array": array,
        }

        queue = self.sensor_queues[(camera, modality)]
        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass
        queue.put(payload)

    def update_pose(self, location: carla.Location, yaw: float) -> None:
        for name, cfg in self.settings.cameras.items():
            base_rotation = carla.Rotation(*cfg.rotation)
            world_rotation = carla.Rotation(
                pitch=base_rotation.pitch,
                yaw=base_rotation.yaw + yaw,
                roll=base_rotation.roll,
            )

            offset = carla.Location(*cfg.position)
            yaw_rad = math.radians(yaw)
            rotated_x = offset.x * math.cos(yaw_rad) - offset.y * math.sin(yaw_rad)
            rotated_y = offset.x * math.sin(yaw_rad) + offset.y * math.cos(yaw_rad)
            world_location = carla.Location(
                x=location.x + rotated_x,
                y=location.y + rotated_y,
                z=location.z + offset.z,
            )
            transform = carla.Transform(world_location, world_rotation)

            for modality, sensor in self.sensors[name].items():
                sensor.set_transform(transform)

    def fetch_frame(self, frame_id: int, timeout: float) -> dict | None:
        frame_data: Dict[str, Dict[str, dict]] = {name: {} for name in self.settings.cameras}
        for name, cfg in self.settings.cameras.items():
            for modality in cfg.modalities:
                sample = self._pop_until_frame((name, modality), frame_id, timeout)
                if sample is None:
                    return None
                frame_data[name][modality] = sample
        return frame_data

    def _pop_until_frame(
        self,
        key: Tuple[str, str],
        target_frame: int,
        timeout: float,
    ) -> dict | None:
        deadline = time.time() + timeout
        queue = self.sensor_queues[key]
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                sample = queue.get(timeout=remaining)
            except Empty:
                return None

            if sample["frame"] == target_frame:
                return sample
        return None

    def cleanup(self) -> None:
        for sensors in self.sensors.values():
            for sensor in sensors.values():
                sensor.stop()
                sensor.destroy()


class RoadWaypointPlanner:
    """Iterate through road-aligned waypoints at a fixed altitude."""

    def __init__(self, world: carla.World, altitude: float, sample_distance: float) -> None:
        self.world = world
        self.map = world.get_map()
        self.altitude = altitude
        self.sample_distance = sample_distance
        self.waypoints = self.map.generate_waypoints(sample_distance)
        if not self.waypoints:
            raise RuntimeError("No waypoints generated; check the sample distance.")

    def iter_transforms(self) -> Iterator[carla.Transform]:
        while True:
            for waypoint in self.waypoints:
                base_transform = waypoint.transform
                location = carla.Location(
                    x=base_transform.location.x,
                    y=base_transform.location.y,
                    z=self.altitude,
                )
                rotation = carla.Rotation(
                    pitch=0.0,
                    yaw=base_transform.rotation.yaw,
                    roll=0.0,
                )
                yield carla.Transform(location, rotation)


class FrameWriter:
    """Persist sensor data and metadata to disk."""

    def __init__(self, settings: CollectorSettings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.output_dir)
        self.rgb_root = self.output_dir / "rgb"
        self.depth_root = self.output_dir / "depth"
        self.semantic_root = self.output_dir / "semantic"
        self.metadata_root = self.output_dir / "metadata"

        for root in [self.rgb_root, self.depth_root, self.semantic_root]:
            for camera in settings.cameras:
                (root / camera).mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)

    def save(self, frame_idx: int, state: dict, frame_data: dict) -> None:
        frame_name = f"{frame_idx:06d}"
        for camera, modalities in frame_data.items():
            for modality, payload in modalities.items():
                array = payload["array"]
                if modality == "rgb":
                    path = self.rgb_root / camera / f"{frame_name}.png"
                    cv2.imwrite(str(path), array)
                elif modality == "depth":
                    raw_path = self.depth_root / camera / frame_name
                    np.save(str(raw_path), array)
                    vis = self._depth_to_png(array)
                    vis_path = raw_path.parent / f"{raw_path.name}.png"
                    cv2.imwrite(str(vis_path), vis)
                elif modality == "semantic":
                    path = self.semantic_root / camera / f"{frame_name}.png"
                    cv2.imwrite(str(path), array)

        metadata = {
            "frame_id": frame_name,
            "timestamp": state["timestamp"],
            "uav_state": {
                "position": state["position"],
                "rotation": state["rotation"],
            },
            "map_name": state["map_name"],
        }
        with open(self.metadata_root / f"{frame_name}.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @staticmethod
    def _depth_to_png(depth: np.ndarray) -> np.ndarray:
        normalized = np.clip(depth / 1000.0, 0.0, 1.0)
        log_depth = np.clip(1 + np.log(normalized + 1e-6) / 5.70378, 0.0, 1.0)
        return (log_depth * 255).astype(np.uint8)


class CarlaMultiViewCollector:
    """High-level orchestrator that ties together CARLA world, sensors, and storage."""

    def __init__(self, settings: CollectorSettings) -> None:
        self.settings = settings
        self.client = carla.Client(settings.server_host, settings.server_port)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(settings.map_name)
        self._configure_world()
        self.camera_rig = CameraRig(self.world, settings)
        self.writer = FrameWriter(settings)
        self.planner = RoadWaypointPlanner(
            self.world, altitude=settings.altitude, sample_distance=settings.sample_distance
        )

    def _configure_world(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.settings.frequency_hz
        self.world.apply_settings(settings)
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

    def collect(self, max_frames: int | None = None) -> None:
        frame_idx = 0
        for transform in self.planner.iter_transforms():
            if max_frames is not None and frame_idx >= max_frames:
                break

            self.camera_rig.update_pose(transform.location, transform.rotation.yaw)
            frame_id = self.world.tick()
            frame_data = self.camera_rig.fetch_frame(frame_id, self.settings.sensor_timeout)
            if not frame_data:
                LOGGER.warning("Timed out waiting for sensors at frame %d", frame_id)
                continue

            state = {
                "timestamp": time.time(),
                "position": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z,
                },
                "rotation": {
                    "pitch": transform.rotation.pitch,
                    "yaw": transform.rotation.yaw,
                    "roll": transform.rotation.roll,
                },
                "map_name": self.world.get_map().name,
            }

            self.writer.save(frame_idx, state, frame_data)
            frame_idx += 1

        LOGGER.info("Collected %d frames.", frame_idx)

    def cleanup(self) -> None:
        self.camera_rig.cleanup()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    settings = CollectorSettings()
    collector = CarlaMultiViewCollector(settings)
    try:
        collector.collect()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
