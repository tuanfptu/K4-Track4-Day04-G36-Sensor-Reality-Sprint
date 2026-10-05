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
