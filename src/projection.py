import numpy as np


def project(points: np.ndarray, p: np.ndarray, rect: np.ndarray, transform: np.ndarray,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray, int]:
    xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
    camera = (transform @ xyz1.T).T
    rectified = (rect @ camera.T).T
    front = np.isfinite(rectified).all(axis=1) & (rectified[:, 2] > 0)
    front_count = int(front.sum())
    rect1 = np.column_stack((rectified[front], np.ones(front_count)))
    projected = (p @ rect1.T).T
    denominator = projected[:, 2]
    valid = np.isfinite(projected).all(axis=1) & (denominator > 0)
    pixels = np.full((front_count, 2), np.nan)
    pixels[valid] = projected[valid, :2] / denominator[valid, None]
    inside = valid & np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels[inside], rectified[front, 2][inside], front_count
