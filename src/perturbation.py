"""Inject yaw calibration drift into a KITTI LiDAR-to-camera extrinsic."""

import numpy as np


def perturb_extrinsic(transform: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    """Return a new float64 3x4 extrinsic with yaw in the LiDAR frame.

    ``transform`` maps LiDAR coordinates to the unrectified camera frame:
    ``p_camera = R @ p_lidar + t``. KITTI LiDAR axes are X forward, Y left,
    Z up. Positive yaw rotates X toward Y about LiDAR Z (right-hand rule).

    Composition is ``T_drift = T_original @ delta_T_lidar``, where delta_T
    contains only Rz(yaw_deg). Thus ``R_drift = R @ Rz`` and translation is
    unchanged: the rotation pivot is the LiDAR origin. This is an active
    perturbation of the transform; correcting it uses the opposite angle.
    Camera intrinsics and rectification must remain outside this function.

    Angles are in degrees. The benchmark uses 0, 0.5, 1, 1.5, 2 degrees;
    any finite scalar angle is supported for signed drift and verification.
    The input must be a finite, real 3x4 rigid extrinsic. Shape and finiteness
    are validated; the caller supplies the original rigid rotation matrix.
    The input is never modified and the output does not share its storage.

    Raises:
        ValueError: If the transform is not a finite real 3x4 matrix, or
            yaw_deg is not a finite real scalar.
    """
    if np.iscomplexobj(transform):
        raise ValueError("transform must be a finite real 3x4 matrix")
    try:
        original = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("transform must be a finite real 3x4 matrix") from error
    if original.shape != (3, 4) or not np.isfinite(original).all():
        raise ValueError("transform must be a finite real 3x4 matrix")

    if np.ndim(yaw_deg) != 0 or np.iscomplexobj(yaw_deg):
        raise ValueError("yaw_deg must be a finite real scalar in degrees")
    try:
        angle = float(yaw_deg)
    except (TypeError, ValueError) as error:
        raise ValueError("yaw_deg must be a finite real scalar in degrees") from error
    if not np.isfinite(angle):
        raise ValueError("yaw_deg must be a finite real scalar in degrees")

    theta = np.deg2rad(angle)
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    drifted = original.copy()
    drifted[:, :3] = original[:, :3] @ rotation
    return drifted
