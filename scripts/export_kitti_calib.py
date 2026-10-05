"""Export KITTI text calibration from the trusted MMDetection3D frame 000008 pickle."""

import argparse
import pickle
from pathlib import Path

import numpy as np


class SampleUnpickler(pickle.Unpickler):
    """Allow only the constructors used by the inspected OpenMMLab sample."""

    ALLOWED = {
        ("numpy.core.multiarray", "scalar"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"Unexpected pickle class: {module}.{name}")
        return super().find_class(module, name)


def matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        array = array[: shape[0], : shape[1]]
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite {shape[0]}x{shape[1]}")
    return array


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/sample/000008.pkl")
    parser.add_argument("--output", type=Path, default=root / "data/sample/calib.txt")
    args = parser.parse_args()
    with args.input.open("rb") as stream:
        sample = SampleUnpickler(stream).load()["data_list"][0]
    if sample["sample_id"] != 8:
        raise ValueError("Expected KITTI sample_id 8")
    images = sample["images"]
    if Path(images["CAM2"]["img_path"]).name != "000008.png":
        raise ValueError("CAM2 image does not match frame 000008")
    if Path(sample["lidar_points"]["lidar_path"]).name != "000008.bin":
        raise ValueError("LiDAR does not match frame 000008")

    p = {f"P{n}": matrix(images[f"CAM{n}"]["cam2img"], (3, 4), f"P{n}") for n in range(4)}
    rect = matrix(images["R0_rect"], (3, 3), "R0_rect")
    # CAM2.lidar2cam in this pickle is already rectified. The baseline applies
    # R0_rect itself, so export the original unrectified Velodyne transform.
    transform = matrix(sample["lidar_points"]["Tr_velo_to_cam"], (3, 4), "Tr_velo_to_cam")
    rect4 = np.eye(4)
    rect4[:3, :3] = rect
    transform4 = np.eye(4)
    transform4[:3, :] = transform
    reference = matrix(images["CAM2"]["lidar2img"], (3, 4), "CAM2.lidar2img")
    if not np.allclose(p["P2"] @ rect4 @ transform4, reference, rtol=1e-5, atol=1e-4):
        raise ValueError("Exported calibration disagrees with source lidar2img")

    fields = {**p, "R0_rect": rect, "Tr_velo_to_cam": transform}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(
        f"{name}: " + " ".join(f"{value:.12g}" for value in array.flat)
        for name, array in fields.items()
    ) + "\n", encoding="utf-8")
    print(f"Exported frame 000008 calibration: {args.output}")


if __name__ == "__main__":
    main()
