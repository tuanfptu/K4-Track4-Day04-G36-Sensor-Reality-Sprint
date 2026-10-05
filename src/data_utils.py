"""KITTI demo annotations and geometric reference data for association.

Camera boxes: (x, y, z, length, height, width, rotation_y), bottom centre,
metres/radians, rectified camera axes right/down/forward. Pixel boxes: xyxy,
continuous coordinates (no +1 in areas), left/top inclusive, right/bottom
exclusive for point membership. See docs/DINH_VALIDATION.md for the contract.
"""

import pickle
from pathlib import Path

import numpy as np


KITTI_CLASSES = ("Pedestrian", "Cyclist", "Car", "Van", "Truck",
                 "Person_sitting", "Tram", "Misc")
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


class SampleUnpickler(pickle.Unpickler):
    """Restricted constructors for the inspected bundled MMDetection3D pickle."""

    ALLOWED = {("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar"),
               ("numpy", "dtype"), ("_codecs", "encode")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")
        return super().find_class(module, name)


def load_sample_info(path: Path, sample_id: int = 8) -> dict:
    """Select exactly one sample; reject mismatched source frame identifiers."""
    with Path(path).open("rb") as stream:
        dataset = SampleUnpickler(stream).load()
    if not isinstance(dataset, dict) or not isinstance(dataset.get("data_list"), list):
        raise ValueError("Expected MMDetection3D dict with data_list")
    matches = [s for s in dataset["data_list"] if s.get("sample_id", s.get("sample_idx")) == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one sample {sample_id}, found {len(matches)}")
    sample = matches[0]
    cam = sample["images"]["CAM2"]
    if Path(cam["img_path"]).stem != f"{sample_id:06d}":
        raise ValueError("CAM2 source frame ID mismatch")
    if Path(sample["lidar_points"]["lidar_path"]).stem != f"{sample_id:06d}":
        raise ValueError("LiDAR source frame ID mismatch")
    if sample["lidar_points"]["num_pts_feats"] != 4:
        raise ValueError("Expected four LiDAR features: x, y, z, reflectance")
    return sample


def extract_objects(sample: dict) -> list[dict]:
    """Preserve every raw annotation, including ignored sentinel boxes.

    Label mapping follows the official KITTI v2 converter, not a detector's
    configurable training class order. -1 has lost its original class name.
    """
    objects = []
    for row, instance in enumerate(sample["instances"]):
        obj = dict(instance)
        label = int(obj["bbox_label"])
        if label < -1 or label >= len(KITTI_CLASSES):
            raise ValueError(f"Unknown KITTI label {label} at row {row}")
        if int(obj["bbox_label_3d"]) != label:
            raise ValueError(f"2D/3D label mismatch at row {row}")
        bbox = np.asarray(obj["bbox"], dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
            raise ValueError(f"Invalid 2D bbox at row {row}")
        box = np.asarray(obj["bbox_3d"], dtype=float)
        if box.shape != (7,) or not np.isfinite(box).all():
            raise ValueError(f"Invalid 3D bbox at row {row}")
        if label >= 0 and (np.any(box[3:6] <= 0) or box[2] <= 0):
            raise ValueError(f"Nonpositive dimensions/depth at row {row}")
        obj.update(object_id=row, class_name=KITTI_CLASSES[label] if label >= 0 else "Ignore",
                   ignore=label == -1, valid_3d=label >= 0)
        objects.append(obj)
    return objects


def calibration_matrices(p, rect, transform):
    """Return rectified LiDAR-to-camera and LiDAR-to-image matrices."""
    p, rect, transform = (np.asarray(v, dtype=float) for v in (p, rect, transform))
    if p.shape != (3, 4) or rect.shape != (3, 3) or transform.shape != (3, 4):
        raise ValueError("Expected P2 3x4, R0_rect 3x3, Tr_velo_to_cam 3x4")
    if not all(np.isfinite(v).all() for v in (p, rect, transform)):
        raise ValueError("Calibration matrices must be finite")
    lidar2cam = np.eye(4)
    lidar2cam[:3] = rect @ transform
    return lidar2cam, p @ lidar2cam


def project_with_indices(points, p, rect, transform, width, height) -> dict:
    """Keep one row per input point, including invisible points; never reorder.

    Pixels are NaN when depth/division is invalid. in_image additionally checks
    0 <= u < width and 0 <= v < height. point_indices index the original .bin.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or width <= 0 or height <= 0:
        raise ValueError("Expected Nx3+ points and positive image dimensions")
    lidar2cam, _ = calibration_matrices(p, rect, transform)
    xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
    camera = (xyz1 @ lidar2cam.T)[:, :3]
    q = np.column_stack((camera, np.ones(len(points)))) @ np.asarray(p).T
    front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    valid = front & np.isfinite(q).all(axis=1) & (q[:, 2] > 0)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = q[valid, :2] / q[valid, 2, None]
    inside = valid & np.isfinite(uv).all(axis=1)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return dict(point_indices=np.arange(len(points)), camera_xyz=camera, pixels=uv,
                depth=camera[:, 2], positive_depth=front, in_image=inside)


def rotation_y(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def camera_box_corners(box):
    box = np.asarray(box, dtype=float)
    if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
        raise ValueError("Expected finite camera box with positive l,h,w")
    template = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                         [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]])
    local = (template - [0.5, 1, 0.5]) * box[3:6]
    return local @ rotation_y(box[6]).T + box[:3]


def project_camera_box(box, p, width, height):
    """Project eight corners; return None if any corner is behind the camera.

    Near-plane clipping is intentionally unsupported: do not produce a
    misleading enclosing rectangle for boxes crossing that plane.
    """
    corners = camera_box_corners(box)
    q = np.column_stack((corners, np.ones(8))) @ np.asarray(p).T
    if not np.isfinite(q).all() or np.any(corners[:, 2] <= 0) or np.any(q[:, 2] <= 0):
        return None
    uv = q[:, :2] / q[:, 2, None]
    raw = np.concatenate((uv.min(axis=0), uv.max(axis=0)))
    clipped = np.clip(raw, [0, 0, 0, 0], [width, height, width, height])
    return dict(corners_uv=uv, bbox_raw=raw, bbox_clipped=clipped)


def bbox_iou(a, b) -> np.ndarray:
    """Pairwise continuous xyxy IoU (N,M); degenerate boxes give zero."""
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bboxes must be finite")
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def points_in_camera_boxes(camera_xyz, boxes):
    """NxM geometric membership, inclusive 3D faces, no invented point labels."""
    points = np.asarray(camera_xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 7)
    result = np.zeros((len(points), len(boxes)), dtype=bool)
    for j, box in enumerate(boxes):
        camera_box_corners(box)  # validate dimensions before inverse rotation
        local = (points - box[:3]) @ rotation_y(box[6])
        lower = np.array([-box[3] / 2, -box[4], -box[5] / 2])
        upper = np.array([box[3] / 2, 0, box[5] / 2])
        result[:, j] = np.isfinite(local).all(axis=1) & (local >= lower - 1e-7).all(axis=1) & (local <= upper + 1e-7).all(axis=1)
    return result


def points_in_bboxes(pixels, boxes):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return ((pixels[:, None] >= boxes[None, :, :2]).all(axis=2) &
            (pixels[:, None] < boxes[None, :, 2:]).all(axis=2) &
            np.isfinite(pixels).all(axis=1)[:, None])


def points_in_kitti_lidar_boxes(points, camera_boxes, lidar2cam):
    """Reproduce KITTI converter's upright LiDAR box counting convention.

    Centre uses the full inverse transform, dimensions become l,w,h, and yaw
    becomes -rotation_y-pi/2. This drops camera/LiDAR roll and pitch from box
    orientation; it is NOT identical to transforming points into camera boxes.
    Strict faces match the converter's convex-polyhedron membership test.
    """
    points = np.asarray(points, dtype=float)
    boxes = np.asarray(camera_boxes, dtype=float).reshape(-1, 7)
    inverse = np.linalg.inv(lidar2cam)
    result = np.zeros((len(points), len(boxes)), dtype=bool)
    for j, box in enumerate(boxes):
        camera_box_corners(box)
        centre = (inverse @ np.append(box[:3], 1))[:3]
        yaw = -box[6] - np.pi / 2
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        local = (points[:, :3] - centre) @ rotation
        lower = [-box[3] / 2, -box[5] / 2, 0]
        upper = [box[3] / 2, box[5] / 2, box[4]]
        result[:, j] = (local > lower).all(axis=1) & (local < upper).all(axis=1)
    return result


def prepare_association(points, projection, objects) -> dict:
    """Produce arrays usable with np.savez_compressed, without object dtype.

    Columns correspond to object_ids, not source labels or raw KITTI index.
    Reference = membership in GT 3D cuboids. Candidates = projection in GT 2D
    rectangles. This is an oracle geometry diagnostic, not a detector score.
    """
    valid = [o for o in objects if o["valid_3d"]]
    ignored = [o for o in objects if o["ignore"]]
    boxes2d = np.array([o["bbox"] for o in valid], dtype=float).reshape(-1, 4)
    boxes3d = np.array([o["bbox_3d"] for o in valid], dtype=float).reshape(-1, 7)
    ignore_boxes = np.array([o["bbox"] for o in ignored], dtype=float).reshape(-1, 4)
    membership = points_in_camera_boxes(projection["camera_xyz"], boxes3d)
    candidates = points_in_bboxes(projection["pixels"], boxes2d) & projection["in_image"][:, None]
    in_ignore = points_in_bboxes(projection["pixels"], ignore_boxes).any(axis=1) & projection["in_image"]
    ambiguous = membership.sum(axis=1) > 1
    eligible = projection["in_image"] & ~in_ignore & ~ambiguous
    reference = np.full(len(points), -1, dtype=np.int64)  # -1 = background
    for column, obj in enumerate(valid):
        reference[membership[:, column]] = obj["object_id"]
    reference[ambiguous] = -2
    return dict(points_lidar=np.asarray(points), **projection,
                object_ids=np.array([o["object_id"] for o in valid], dtype=np.int64),
                labels=np.array([o["bbox_label"] for o in valid], dtype=np.int64),
                boxes_2d=boxes2d, boxes_3d_camera=boxes3d, ignore_boxes_2d=ignore_boxes,
                membership_3d=membership, candidate_2d=candidates,
                in_ignore_region=in_ignore, ambiguous_3d=ambiguous,
                metric_eligible=eligible, reference_object_id=reference)


def association_diagnostics(data) -> list[dict]:
    """Per-object retrieval counts; undefined precision/recall are None."""
    rows = []
    for j, object_id in enumerate(data["object_ids"]):
        gt = data["membership_3d"][:, j] & data["metric_eligible"]
        pred = data["candidate_2d"][:, j] & data["metric_eligible"]
        tp, fp, fn = int((gt & pred).sum()), int((~gt & pred).sum()), int((gt & ~pred).sum())
        rows.append(dict(object_id=int(object_id), points_in_3d=int(data["membership_3d"][:, j].sum()),
                         points_in_2d=int(data["candidate_2d"][:, j].sum()), tp=tp, fp=fp, fn=fn,
                         precision=tp / (tp + fp) if tp + fp else None,
                         recall=tp / (tp + fn) if tp + fn else None))
    return rows
