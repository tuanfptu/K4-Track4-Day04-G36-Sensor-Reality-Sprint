import numpy as np

from src.data_utils import points_in_bboxes


def measure_yaw_drift(reference: dict, corrupted: dict, objects: list[dict]) -> tuple[dict, list[dict]]:
    """Measure matched original LiDAR rows and fixed 2D-box pair retention.

    Both projections have one row per original point. Displacement uses rows
    visible in both images; out-of-frame uses all baseline-visible rows.
    Each bbox contributes its own baseline point IDs, including overlaps.
    """
    base_uv, bad_uv = reference["pixels"], corrupted["pixels"]
    base_visible, bad_visible = reference["in_image"], corrupted["in_image"]
    if base_uv.shape != bad_uv.shape or base_visible.shape != bad_visible.shape:
        raise ValueError("Projections must preserve the same point IDs")
    if not base_visible.any():
        raise ValueError("No baseline-visible points")
    common = base_visible & bad_visible
    shift = np.linalg.norm(bad_uv[common] - base_uv[common], axis=1)
    if not np.isfinite(shift).all():
        raise ValueError("Matched projection contains nonfinite displacement")
    boxes = np.asarray([obj["bbox"] for obj in objects], dtype=float).reshape(-1, 4)
    before = points_in_bboxes(base_uv, boxes) & base_visible[:, None]
    after = points_in_bboxes(bad_uv, boxes) & bad_visible[:, None]
    baseline_pairs = before.sum(axis=0)
    retained_pairs = (before & after).sum(axis=0)
    total_pairs = int(baseline_pairs.sum())
    total_retained = int(retained_pairs.sum())
    retention = 100 * total_retained / total_pairs if total_pairs else None
    summary = {
        "mean_reprojection_px": float(np.mean(shift)) if len(shift) else None,
        "median_reprojection_px": float(np.median(shift)) if len(shift) else None,
        "p90_reprojection_px": float(np.percentile(shift, 90)) if len(shift) else None,
        "max_reprojection_px": float(np.max(shift)) if len(shift) else None,
        "association_retention_pct": retention,
        "association_drop_pct": 100 - retention if retention is not None else None,
        "out_of_frame_pct": 100 * int((base_visible & ~bad_visible).sum()) / int(base_visible.sum()),
        "gt_5px_pct": 100 * float(np.mean(shift > 5)) if len(shift) else None,
        "gt_10px_pct": 100 * float(np.mean(shift > 10)) if len(shift) else None,
        "gt_20px_pct": 100 * float(np.mean(shift > 20)) if len(shift) else None,
        "baseline_visible_points": int(base_visible.sum()),
        "matched_visible_points": int(common.sum()),
        "baseline_associated_pairs": total_pairs,
        "retained_associated_pairs": total_retained,
    }
    per_object = []
    for j, obj in enumerate(objects):
        n, kept = int(baseline_pairs[j]), int(retained_pairs[j])
        rate = 100 * kept / n if n else None
        per_object.append({
            "object_id": int(obj["object_id"]), "class": obj["class_name"],
            "bbox": ",".join(f"{float(value):.2f}" for value in obj["bbox"]),
            "baseline_associated_points": n, "retained_points": kept,
            "association_retention_pct": rate,
            "association_drop_pct": 100 - rate if rate is not None else None,
        })
    return summary, per_object


def reprojection_displacement(pix_ref: np.ndarray, pix_pert: np.ndarray) -> dict:
    """Per-point pixel displacement between reference and perturbed projections (index-aligned).

    Only points in front of the camera in both runs are compared.
    """
    ref_ok = np.isfinite(pix_ref).all(axis=1)
    both = ref_ok & np.isfinite(pix_pert).all(axis=1)
    if not both.any():
        return {"median_px": np.nan, "p90_px": np.nan, "n_compared": 0, "valid_ratio": 0.0}
    shift = np.linalg.norm(pix_pert[both] - pix_ref[both], axis=1)
    return {
        "median_px": float(np.median(shift)),
        "p90_px": float(np.percentile(shift, 90)),
        "n_compared": int(both.sum()),
        "valid_ratio": float(both.sum() / max(int(ref_ok.sum()), 1)),
    }


def assign_to_boxes(pixels: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Box index per point (-1 = none). Overlaps go to the smallest containing box (usually the nearer, occluding object)."""
    x, y = pixels[:, 0:1], pixels[:, 1:2]
    inside = (x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    assigned = np.argmin(np.where(inside, area, np.inf), axis=1)
    assigned[~inside.any(axis=1)] = -1
    return assigned


def association_retention(pix_ref: np.ndarray, pix_pert: np.ndarray, boxes: np.ndarray) -> dict:
    """Fraction of points assigned to a GT box at baseline that stay assigned to the SAME box after perturbation.

    Returns overall retention plus per-box retention and baseline point counts.
    """
    ref = assign_to_boxes(np.nan_to_num(pix_ref, nan=-1e9), boxes)
    pert = assign_to_boxes(np.nan_to_num(pix_pert, nan=-1e9), boxes)
    base = ref >= 0
    kept = base & (pert == ref)
    per_box_n = np.bincount(ref[base], minlength=len(boxes))
    per_box_kept = np.bincount(ref[kept], minlength=len(boxes))
    return {
        "retention": float(kept.sum() / base.sum()) if base.any() else np.nan,
        "n_associated": int(base.sum()),
        "per_box_retention": np.where(per_box_n > 0, per_box_kept / np.maximum(per_box_n, 1), np.nan),
        "per_box_n": per_box_n,
    }


def inside_image_ratio(pixels: np.ndarray, width: int, height: int) -> float:
    ok = np.isfinite(pixels).all(axis=1)
    if not ok.any():
        return 0.0
    u, v = pixels[ok, 0], pixels[ok, 1]
    return float(((u >= 0) & (u < width) & (v >= 0) & (v < height)).sum() / ok.sum())
