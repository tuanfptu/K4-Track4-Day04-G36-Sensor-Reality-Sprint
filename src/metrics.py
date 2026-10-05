import numpy as np


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
