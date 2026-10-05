"""Reproducible single-frame LiDAR-frame yaw benchmark (KITTI 000008)."""

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration
from src.data_utils import extract_objects, load_sample_info, project_with_indices
from src.kitti_loader import load_image, load_lidar
from src.metrics import measure_yaw_drift
from src.perturbation import perturb_extrinsic
from src.visualization import overlay

YAW_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path, rows, fields, labels, ylabel):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 4), dpi=140)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    yaw = [row["yaw_deg"] for row in rows]
    for field, label in zip(fields, labels):
        axis.plot(yaw, [row[field] for row in rows], marker="o", linewidth=2, label=label)
    axis.set(xlabel="LiDAR-frame yaw drift (degrees)", ylabel=ylabel, xlim=(0, 2))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)


def draw_comparison(image, baseline, drifted):
    width, height = image.size
    output = Image.new("RGB", (width * 2, height + 32), "#142031")
    output.paste(baseline, (0, 32))
    output.paste(drifted, (width, 32))
    draw = ImageDraw.Draw(output)
    draw.text((12, 9), "BASELINE 0 deg", fill="white")
    draw.text((width + 12, 9), "LIDAR-FRAME YAW 2 deg", fill="white")
    return output


def run(output_dir: Path):
    sample_dir = ROOT / "data/sample"
    image = load_image(sample_dir / "image.png")
    points = load_lidar(sample_dir / "points.bin")
    p, rect, original = load_calibration(sample_dir / "calib.txt")
    sample = load_sample_info(sample_dir / "000008.pkl")
    if image.size != (int(sample["images"]["CAM2"]["width"]),
                      int(sample["images"]["CAM2"]["height"])):
        raise ValueError("Image dimensions do not match frame 000008 metadata")
    objects = [obj for obj in extract_objects(sample) if obj["valid_3d"]]
    baseline = project_with_indices(points, p, rect, original, *image.size)
    if not baseline["in_image"].any():
        raise ValueError("Baseline has no visible points")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, object_rows = [], []
    baseline_image = None
    last_image = None
    for yaw in YAW_LEVELS:
        transform = perturb_extrinsic(original, yaw)
        projection = project_with_indices(points, p, rect, transform, *image.size)
        metrics, per_object = measure_yaw_drift(baseline, projection, objects)
        summary_rows.append({"yaw_deg": yaw, **metrics})
        object_rows.extend({"yaw_deg": yaw, **row} for row in per_object)
        visible = projection["in_image"]
        rendered = overlay(image, projection["pixels"][visible], projection["depth"][visible])
        rendered.save(output_dir / f"yaw_{str(yaw).replace('.', '_')}.png")
        if yaw == 0:
            baseline_image = rendered
        if yaw == 2:
            last_image = rendered
    zero = summary_rows[0]
    if any(abs(zero[key]) > 1e-8 for key in ("median_reprojection_px", "p90_reprojection_px",
                                               "association_drop_pct", "out_of_frame_pct")):
        raise ValueError("Zero-drift sanity check failed")
    if zero["association_retention_pct"] != 100:
        raise ValueError("Zero-drift association retention is not 100%")
    draw_comparison(image, baseline_image, last_image).save(output_dir / "baseline_vs_yaw_2_0.png")
    write_csv(output_dir / "benchmark_summary.csv", summary_rows)
    write_csv(output_dir / "benchmark_per_object.csv", object_rows)
    plot_curve(output_dir / "reprojection_vs_yaw.png", summary_rows,
               ("median_reprojection_px", "p90_reprojection_px"), ("Median", "P90"),
               "Reprojection displacement relative to baseline (px)")
    plot_curve(output_dir / "association_vs_yaw.png", summary_rows,
               ("association_retention_pct", "association_drop_pct"), ("Retention", "Drop"),
               "Fixed-bbox association (%)")
    plot_curve(output_dir / "out_of_frame_vs_yaw.png", summary_rows,
               ("out_of_frame_pct",), ("Out of frame",), "Baseline-visible points lost (%)")
    metadata = {
        "dataset": "KITTI", "sample_source": "OpenMMLab MMDetection3D demo/data/kitti",
        "frame_id": "000008", "image_dimensions": list(image.size),
        "total_lidar_points": len(points), "baseline_visible_points": int(baseline["in_image"].sum()),
        "yaw_protocol_deg": list(YAW_LEVELS),
        "coordinate_convention": "LiDAR X forward, Y left, Z up; right-compose Rz(yaw) with original extrinsic independently at every level",
        "metric_definitions": {
            "reprojection": "L2 pixel displacement relative to baseline for the same original LiDAR IDs visible in both images",
            "association": "Baseline point-bbox pairs retained in the same fixed 2D bbox; overlapping boxes counted separately",
            "out_of_frame": "Baseline-visible point IDs invisible after drift / baseline-visible point IDs",
            "threshold_exceedance": "Fraction of points visible in both images whose displacement exceeds 5, 10, or 20 px",
        },
        "known_limitations": ["single-frame experiment", "MMDetection3D demo point cloud appears camera-FOV filtered",
                              "association metric is bbox-based proxy", "not object ownership verified by 3D ground truth",
                              "no multimodal detector mAP"],
        "map_status": "NOT EVALUATED",
    }
    (output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/final")
    args = parser.parse_args()
    rows = run(args.output_dir)
    for row in rows:
        print(f"yaw={row['yaw_deg']:.1f} median={row['median_reprojection_px']:.2f}px "
              f"p90={row['p90_reprojection_px']:.2f}px "
              f"retention={row['association_retention_pct']:.2f}% "
              f"out-of-frame={row['out_of_frame_pct']:.2f}%")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
