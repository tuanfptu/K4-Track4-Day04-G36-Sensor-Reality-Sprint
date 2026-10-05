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


def project_indexed(points: np.ndarray, p: np.ndarray, rect: np.ndarray,
                    transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project every point, keeping index alignment with `points`.

    Returns pixels (N, 2) and depth (N,); both are NaN where the point is not in front of the camera.
    Pixels are not clipped to the image so reference/perturbed runs can be compared point by point.
    """
    xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
    rectified = (rect @ (transform @ xyz1.T)).T
    depth = rectified[:, 2]
    front = np.isfinite(rectified).all(axis=1) & (depth > 0)
    pixels = np.full((len(points), 2), np.nan)
    projected = (p @ np.column_stack((rectified[front], np.ones(int(front.sum())))).T).T
    pixels[front] = projected[:, :2] / projected[:, 2:3]
    return pixels, np.where(front, depth, np.nan)
