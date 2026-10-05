from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_lidar(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR missing: {path}")
    if path.stat().st_size == 0 or path.stat().st_size % 16:
        raise ValueError(f"LiDAR must contain float32 x,y,z,reflectance records: {path}")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError("LiDAR contains NaN or Inf")
    return points


def load_gt_boxes(path: Path) -> dict:
    """Valid 2D GT boxes from the MMDetection3D KITTI info pickle (DontCare entries are dropped)."""
    import pickle

    if not path.is_file():
        raise FileNotFoundError(f"Annotation pickle missing: {path}")
    with path.open("rb") as handle:
        info = pickle.load(handle)["data_list"][0]
    valid = [i for i in info["instances"] if i["bbox_label"] >= 0 and i["num_lidar_pts"] > 0]
    return {
        "boxes": np.asarray([i["bbox"] for i in valid], dtype=np.float64).reshape(-1, 4),
        "depth": np.asarray([i["depth"] for i in valid], dtype=np.float64),
        "num_lidar_pts": np.asarray([i["num_lidar_pts"] for i in valid], dtype=np.int64),
    }
