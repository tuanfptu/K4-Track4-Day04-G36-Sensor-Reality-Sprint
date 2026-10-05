from pathlib import Path

import numpy as np


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration missing: {path}")
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, values = line.split(":", 1)
            fields[key.strip()] = values.split()

    def matrix(names, shape):
        for name in names:
            if name in fields:
                values = np.asarray([float(v) for v in fields[name]], dtype=np.float64)
                if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name}: expected {shape[0]}x{shape[1]} finite matrix")
                return values.reshape(shape)
        raise ValueError(f"Calibration missing field: {' or '.join(names)}")

    p = matrix(("P2", "P_rect_02"), (3, 4))
    rect = matrix(("R0_rect", "R_rect", "R_rect_00"), (3, 3))
    transform = matrix(("Tr_velo_to_cam", "Tr_velo_cam", "Tr"), (3, 4))
    return p, rect, transform
