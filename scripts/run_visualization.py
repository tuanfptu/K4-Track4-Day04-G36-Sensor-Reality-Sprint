"""Generate KITTI drift overlays, plots and an offline slider demo."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration
from src.drift_demo import camera_yaw_transform, load_sample_boxes, measure_drift, project_aligned
from src.kitti_loader import load_image, load_lidar
from src.visualization import comparison_overlay, plot_benchmark, read_benchmark_csv, write_demo_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/visualization")
    parser.add_argument("--benchmark-csv", type=Path,
                        help="Plot a team CSV using the documented schema; skip the sample demo")
    parser.add_argument("--yaws", type=float, nargs="+", default=list(np.linspace(0, 2, 9)),
                        help="Demo camera-Y yaw values in degrees, within [0,2], including 0")
    args = parser.parse_args()
    try:
        if args.benchmark_csv:
            rows = read_benchmark_csv(args.benchmark_csv)
            for path in plot_benchmark(rows, args.output_dir):
                print(f"Plot: {path}")
            return

        yaws = sorted(set(args.yaws))
        if not yaws or 0.0 not in yaws or any(not np.isfinite(y) or not 0 <= y <= 2 for y in yaws):
            raise ValueError("Demo yaws must be finite, within [0,2], and include baseline 0")
        image_path = ROOT / "data/sample/image.png"
        lidar_path = ROOT / "data/sample/points.bin"
        calib_path = ROOT / "data/sample/calib.txt"
        labels_path = ROOT / "data/sample/000008.pkl"
        image = load_image(image_path)
        points = load_lidar(lidar_path)
        p, rect, transform = load_calibration(calib_path)
        boxes = load_sample_boxes(labels_path)
        baseline, _, baseline_inside = project_aligned(points, p, rect, transform, *image.size)

        frames, rows = [], []
        for yaw in yaws:
            corrupted, _, inside = project_aligned(
                points, p, rect, camera_yaw_transform(transform, yaw), *image.size)
            metrics = measure_drift(baseline, corrupted, baseline_inside, inside, boxes)
            rows.append({"condition": "camera_yaw", "drift_value": yaw, "drift_unit": "deg",
                         **metrics, "map": None, "map_protocol": ""})
            frames.append({"yaw_deg": yaw, "pixels": corrupted, "metrics": metrics})

        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "demo_benchmark.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plots = plot_benchmark(read_benchmark_csv(csv_path), args.output_dir)
        selected = frames[-1]
        comparison_path = args.output_dir / "comparison.png"
        comparison_overlay(image, baseline, selected["pixels"], boxes,
                           f"camera +Y yaw = {selected['yaw_deg']:g} deg").save(comparison_path)
        demo_path = write_demo_html(image, baseline, frames, boxes,
                                    args.output_dir / "demo.html", plots[0])
        metadata = {
            "frame_id": "000008", "dataset": "KITTI",
            "sample_source": "OpenMMLab MMDetection3D demo/data/kitti",
            "total_lidar_points": len(points), "valid_2d_boxes": len(boxes),
            "yaw_deg": yaws, "selected_overlay_yaw_deg": selected["yaw_deg"],
            "perturbation": "T_corrupt = R_y_camera(yaw) @ T_original; P2 and R0_rect fixed",
            "camera_axes": "x right, y down, z forward; rotation before rectification",
            "error_definition": "L2 pixel displacement vs baseline, same row IDs visible in both images",
            "association_definition": "retained baseline point-box pairs / baseline point-box pairs; fixed 2D boxes",
            "association_drop_definition": "100 * (1 - association_retention), percentage points",
            "map": None, "map_status": "not evaluated; no detector predictions or evaluation set",
            "limits": ["Single demo frame, not a dataset benchmark",
                       "Point cloud is prefiltered to camera FOV",
                       "Pixel displacement is not independent ground-truth reprojection error",
                       "2D bbox membership is an association proxy, not 3D object ownership",
                       "Viewer draws at most 3500 fixed sampled point IDs; metrics use full cloud"],
            "input_sha256": {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (image_path, lidar_path, calib_path, labels_path)},
        }
        metadata_path = args.output_dir / "demo_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for path in [comparison_path, *plots, csv_path, metadata_path, demo_path]:
            print(f"Output: {path}")
        print(f"Frame 000008: {len(points)} points, {len(boxes)} boxes, {len(yaws)} yaw levels")
        print(f"At {selected['yaw_deg']:g} deg: {json.dumps(selected['metrics'], allow_nan=False)}")
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
