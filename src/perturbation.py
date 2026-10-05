import numpy as np


def rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Rotation about camera x (pitch), y (yaw), z (roll); camera frame is x right, y down, z forward."""
    ax, ay, az = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def perturb_extrinsics(transform: np.ndarray, rot_deg=(0.0, 0.0, 0.0), trans_m=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Apply a small error dT in the camera frame to the 3x4 LiDAR->camera extrinsic: Tr_new = dT @ Tr."""
    delta_r = rotation_matrix(*rot_deg)
    rotation, translation = transform[:, :3], transform[:, 3]
    return np.column_stack((delta_r @ rotation, delta_r @ translation + np.asarray(trans_m, dtype=np.float64)))
