"""Frame-aligned geometry and explicitly limited metrics for the visual demo.

These are demo proxies, not the team's official benchmark: displacement is
relative to the baseline projection, and association is fixed 2D box membership.
Neither represents true GT point-object association or detector mAP.
"""

from pathlib import Path

import numpy as np

from scripts.export_kitti_calib import SampleUnpickler


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def project_aligned(points, p, rect, transform, width, height):
    """Return pixels, rectified camera depth, and in-image mask in point-row order.

    Points with invalid/nonpositive depth or projection denominator have NaN UV.
    Finite offscreen UV are retained; clipping never changes row identity.
    Camera matrices use the same P2 @ R0_rect @ Tr chain as the legacy baseline.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    p = _matrix(p, (3, 4), "p")
    rect = _matrix(rect, (3, 3), "rect")
    transform = _matrix(transform, (3, 4), "transform")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive and finite")

    xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        camera = (transform @ xyz1.T).T
        rectified = (rect @ camera.T).T
        depth = rectified[:, 2].copy()
        front = np.isfinite(rectified).all(axis=1) & (depth > 0)
        projected = (p @ np.column_stack((rectified, np.ones(len(points)))).T).T
        valid = front & np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0)
        pixels = np.full((len(points), 2), np.nan)
        pixels[valid] = projected[valid, :2] / projected[valid, 2, None]
    pixels[~np.isfinite(pixels).all(axis=1)] = np.nan
    inside = np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, depth, inside


def camera_yaw_transform(transform, yaw_deg):
    """Apply camera-frame yaw as R_y(yaw_deg) @ original unrectified 3x4 T.

    Camera axes: +x right, +y down, +z forward. Positive yaw rotates +z
    toward +x (right); rectification is subsequently applied by projection.
    Both rotation and translation in T are left-multiplied. This standalone
    demo convention is not an implicit replacement for the team's perturb.py.
    """
    transform = _matrix(transform, (3, 4), "transform")
    yaw_deg = float(yaw_deg)
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw_deg must be finite")
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rotation @ transform


def load_sample_boxes(path):
    """Read valid labeled 2D boxes from the restricted frame-000008 sample.

    Uses the export script's constructor allowlist, verifies sample/image/LiDAR
    pairing, and excludes negative labels (DontCare). Returns finite Nx4 xyxy.
    Invalid labeled boxes are rejected rather than silently used in a metric.
    """
    with Path(path).open("rb") as stream:
        data = SampleUnpickler(stream).load()
    samples = data.get("data_list") if isinstance(data, dict) else None
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError("Expected exactly one frame in sample metadata")
    sample = samples[0]
    if not isinstance(sample, dict) or sample.get("sample_id") != 8:
        raise ValueError("Expected KITTI sample_id 8")
    try:
        image_name = str(sample["images"]["CAM2"]["img_path"]).replace("\\", "/").rsplit("/", 1)[-1]
        lidar_name = str(sample["lidar_points"]["lidar_path"]).replace("\\", "/").rsplit("/", 1)[-1]
        instances = sample["instances"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Sample missing frame pairing or instances") from exc
    if image_name != "000008.png" or lidar_name != "000008.bin":
        raise ValueError("Image and LiDAR must both match frame 000008")
    if not isinstance(instances, list):
        raise ValueError("Sample instances must be a list")
    boxes = []
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError("Each instance must be a dictionary")
        label = instance.get("bbox_label")
        if not isinstance(label, (int, float, np.integer, np.floating)) or not np.isfinite(label):
            raise ValueError("bbox_label must be finite and numeric")
        if label < 0:
            continue
        box = _matrix(instance.get("bbox"), (4,), "bbox")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Labeled bbox must have positive width and height")
        boxes.append(box)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def measure_drift(baseline_pixels, corrupted_pixels, baseline_inside, corrupted_inside, boxes):
    """Measure displacement and fixed-box association proxy using same point IDs.

    Median/P90 displacement includes only points visible in BOTH projections;
    no such points gives None. Association counts point-box pairs, including
    overlaps once per box. Retention is the fraction of baseline pairs still
    in that same box and in the image; new memberships never offset losses.
    Retention is in [0,1], drop is percentage points, and both are None when
    baseline has no associations. These are not GT associations or mAP.
    """
    baseline = np.asarray(baseline_pixels, dtype=np.float64)
    corrupted = np.asarray(corrupted_pixels, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[1] != 2 or corrupted.shape != baseline.shape:
        raise ValueError("Projection arrays must have matching shape (N, 2)")
    masks = []
    for mask, pixels in ((baseline_inside, baseline), (corrupted_inside, corrupted)):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(baseline),):
            raise ValueError("Visibility masks must have shape (N,)")
        masks.append(mask & np.isfinite(pixels).all(axis=1))
    baseline_inside, corrupted_inside = masks
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("boxes must have finite shape (M, 4)")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("boxes must have positive width and height")

    matched = baseline_inside & corrupted_inside
    errors = np.linalg.norm(corrupted[matched] - baseline[matched], axis=1)

    def memberships(pixels, inside):
        # Inclusive annotation boundaries; image visibility remains half-open.
        return (inside[:, None]
                & (pixels[:, None, 0] >= boxes[None, :, 0])
                & (pixels[:, None, 0] <= boxes[None, :, 2])
                & (pixels[:, None, 1] >= boxes[None, :, 1])
                & (pixels[:, None, 1] <= boxes[None, :, 3]))

    before = memberships(baseline, baseline_inside)
    after = memberships(corrupted, corrupted_inside)
    count = int(before.sum())
    retained = int((before & after).sum())
    retention = retained / count if count else None
    return {
        "median_error_px": float(np.median(errors)) if len(errors) else None,
        "p90_error_px": float(np.percentile(errors, 90)) if len(errors) else None,
        "association_retention": retention,
        "association_drop_pp": 100.0 * (1.0 - retention) if retention is not None else None,
        "baseline_visible": int(baseline_inside.sum()),
        "corrupted_visible": int(corrupted_inside.sum()),
        "matched_visible": int(matched.sum()),
        "baseline_associations": count,
        "retained_associations": retained,
    }
