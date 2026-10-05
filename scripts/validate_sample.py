"""Extract frame 000008 GT, validate baseline, export association reference data."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration
from src.data_utils import (BOX_EDGES, association_diagnostics, bbox_iou,
                            calibration_matrices, extract_objects, load_sample_info,
                            points_in_kitti_lidar_boxes, prepare_association,
                            project_camera_box, project_with_indices)
from src.kitti_loader import load_image, load_lidar
from src.projection import project
from src.visualization import overlay


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=json_default) + "\n", encoding="utf-8")


def draw_validation(image, objects, projections, pixels, depth):
    """Two panels: readable 2D GT; baseline points with 3D wireframes."""
    banner = 38
    canvas = Image.new("RGB", (image.width, 2 * (image.height + banner)), (20, 25, 35))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image, (0, banner))
    lower_y = image.height + 2 * banner
    canvas.paste(overlay(image, pixels, depth), (0, lower_y))
    draw.text((12, 6), "KITTI 000008 | GREEN: annotated 2D Car boxes | ORANGE: ignored regions (-1)", fill="white")
    draw.text((12, image.height + banner + 6),
              "BASELINE | CYAN: projected 3D GT cuboids | GREEN: 2D GT | points: near red / far blue", fill="white")
    for obj in objects:
        color = (255, 175, 40) if obj["ignore"] else (70, 255, 100)
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle((x1, y1 + banner, x2, y2 + banner), outline=color, width=2)
        text = f"#{obj['object_id']} {obj['class_name']}"
        # Stack small distant ignore labels so adjacent labels do not obscure one another.
        label_y = max(0, y1 - 14) if not obj["ignore"] else 32 + 15 * (obj["object_id"] % 4)
        text_width = draw.textbbox((0, 0), text)[2] + 6
        label_x = min(max(0, x1), image.width - text_width)
        draw.rectangle((label_x, label_y + banner, label_x + text_width, label_y + banner + 13), fill=(20, 25, 35))
        draw.text((label_x + 2, label_y + banner), text, fill=color)
        if obj["ignore"]:
            draw.line((label_x, label_y + banner + 13, x1, y1 + banner), fill=color)
            continue
        draw.rectangle((x1, y1 + lower_y, x2, y2 + lower_y), outline=color, width=2)
        projected = projections[obj["object_id"]]
        if projected is not None:
            corners = projected["corners_uv"]
            # Draw on a separate image to clip lines strictly to this panel.
            panel = canvas.crop((0, lower_y, image.width, lower_y + image.height))
            panel_draw = ImageDraw.Draw(panel)
            for a, b in BOX_EDGES:
                panel_draw.line((*corners[a], *corners[b]), fill=(35, 230, 255), width=2)
            canvas.paste(panel, (0, lower_y))
    return canvas


def run(sample_dir, output_dir):
    sample = load_sample_info(sample_dir / "000008.pkl")
    image = load_image(sample_dir / "image.png")
    points = load_lidar(sample_dir / "points.bin")
    p, rect, transform = load_calibration(sample_dir / "calib.txt")
    objects = extract_objects(sample)
    valid = [o for o in objects if o["valid_3d"]]
    cam = sample["images"]["CAM2"]
    if image.size != (int(cam["width"]), int(cam["height"])):
        raise ValueError("Image dimensions disagree with source metadata")
    projection = project_with_indices(points, p, rect, transform, *image.size)
    association = prepare_association(points, projection, objects)
    diagnostics = association_diagnostics(association)
    pixels, depth, front_count = project(points, p, rect, transform, *image.size)
    if not len(pixels):
        raise ValueError("Baseline has no points in image")
    lidar2cam, lidar2img = calibration_matrices(p, rect, transform)
    association["membership_kitti_lidar"] = points_in_kitti_lidar_boxes(
        points, association["boxes_3d_camera"], lidar2cam)
    source_counts = association["membership_kitti_lidar"].sum(axis=0)
    source_lidar2cam = np.asarray(cam["lidar2cam"])
    source_lidar2img = np.asarray(cam["lidar2img"])[:3]
    xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
    source_camera = (xyz1 @ source_lidar2cam.T)[:, :3]
    source_q = xyz1 @ source_lidar2img.T
    source_valid = (source_camera[:, 2] > 0) & (source_q[:, 2] > 0) & np.isfinite(source_q).all(axis=1)
    source_uv = np.full((len(points), 2), np.nan)
    source_uv[source_valid] = source_q[source_valid, :2] / source_q[source_valid, 2, None]
    source_inside = source_valid & (source_uv[:, 0] >= 0) & (source_uv[:, 0] < image.width)
    source_inside &= (source_uv[:, 1] >= 0) & (source_uv[:, 1] < image.height)
    common = source_valid & np.isfinite(projection["pixels"]).all(axis=1)
    source_pixel_error = float(np.linalg.norm(source_uv[common] - projection["pixels"][common], axis=1).max()) if common.any() else None
    selected = projection["in_image"]
    same_count = len(pixels) == int(selected.sum())
    baseline_error = float(np.abs(pixels - projection["pixels"][selected]).max()) if same_count else None
    checks = {
        "rectified_transform_matches_source": bool(np.allclose(lidar2cam, source_lidar2cam, atol=1e-6, rtol=0)),
        "projection_matrix_matches_source": bool(np.allclose(lidar2img, source_lidar2img, atol=1e-4, rtol=1e-5)),
        "source_visibility_mask_matches": bool(np.array_equal(source_inside, selected)),
        "source_pixel_error_below_0_001px": source_pixel_error is not None and source_pixel_error < 0.001,
        "baseline_pixels_match": baseline_error is not None and baseline_error < 1e-8,
        "baseline_depth_matches": bool(same_count and np.allclose(depth, projection["depth"][selected], atol=1e-10, rtol=0)),
        "baseline_front_count_matches": front_count == int(projection["positive_depth"].sum()),
    }
    projected_boxes = {o["object_id"]: project_camera_box(o["bbox_3d"], p, *image.size) for o in valid}
    box_rows = []
    for column, (obj, diag) in enumerate(zip(valid, diagnostics)):
        box = np.asarray(obj["bbox_3d"])
        centre = box[:3].copy()
        centre[1] -= box[4] / 2
        q = p @ np.append(centre, 1)
        centre_error = float(np.linalg.norm(q[:2] / q[2] - obj["center_2d"]))
        depth_error = float(abs(q[2] - obj["depth"]))
        projected = projected_boxes[obj["object_id"]]
        checks[f"object_{obj['object_id']}_center_matches"] = centre_error < 0.001 and depth_error < 1e-4
        checks[f"object_{obj['object_id']}_corners_in_front"] = projected is not None
        # Compare like-for-like with the source's upright LiDAR box convention.
        checks[f"object_{obj['object_id']}_source_point_count_matches"] = int(source_counts[column]) == obj["num_lidar_pts"]
        box_rows.append(dict(**diag, source_num_lidar_pts=obj["num_lidar_pts"],
                             recomputed_kitti_lidar_points=int(source_counts[column]),
                             center_error_px=centre_error, depth_error_m=depth_error,
                             projected_bbox=projected["bbox_clipped"] if projected else None,
                             projected_bbox_iou=float(bbox_iou([obj["bbox"]], [projected["bbox_clipped"]])[0, 0]) if projected else None))
    # Negative controls: verify this comparison actually detects common mistakes.
    no_rect = p @ np.vstack((transform, [0, 0, 0, 1]))
    rect4 = np.eye(4)
    rect4[:3, :3] = rect
    twice_rect = p @ rect4 @ lidar2cam
    controls = {}
    for name, matrix in (("omit_rectification", no_rect), ("double_rectification", twice_rect)):
        q = xyz1 @ matrix.T
        eligible = common & (q[:, 2] > 0)
        error = np.linalg.norm(q[eligible, :2] / q[eligible, 2, None] - source_uv[eligible], axis=1)
        controls[name] = dict(mean_pixel_error=float(error.mean()), max_pixel_error=float(error.max()))
        checks[f"negative_control_{name}_detected"] = bool(error.max() > 0.001)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "objects_000008.json", dict(frame_id="000008", label_mapping_source="MMDetection3D KITTI v2 converter",
                bbox_2d_format="xyxy, continuous pixels", bbox_3d_format="x,y,z,l,h,w,rotation_y; rectified camera; bottom center; m,rad",
                ignored_label_note="-1 original class name is not stored; sentinel geometry is consistent with KITTI DontCare",
                objects=objects))
    csv_fields = ["object_id", "class_name", "bbox_label", "ignore", "valid_3d", "bbox", "bbox_3d", "center_2d",
                  "depth", "num_lidar_pts", "difficulty", "truncated", "occluded", "alpha", "score", "index", "group_id"]
    with (output_dir / "objects_000008.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for obj in objects:
            writer.writerow({key: json.dumps(obj[key], default=json_default) if isinstance(obj[key], list) else obj[key] for key in csv_fields})
    np.savez_compressed(output_dir / "association_000008.npz", **association)
    schema = {key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in association.items()}
    write_json(output_dir / "association_schema.json", dict(frame_id="000008", arrays=schema,
                reference="3D cuboid membership; geometric proxy, not point-level semantic ground truth",
                source_count_reference="membership_kitti_lidar reproduces upright LiDAR boxes used by the converter; diagnostics use membership_3d in camera coordinates",
                point_order="Original points.bin row order; point_indices are zero-based",
                column_order="object_ids; six valid Car boxes; excludes all label=-1 rows",
                eligibility="in_image AND NOT in_ignore_region AND NOT ambiguous_3d; background retained for false positives",
                reference_id_values={"-1": "background", "-2": "multiple 3D boxes; excluded from metrics"},
                bbox_membership="2D left/top inclusive, right/bottom exclusive; 3D inclusive faces with 1e-7m tolerance",
                metric="Per-object 2D point retrieval versus 3D geometric membership; precision=TP/(TP+FP), recall=TP/(TP+FN); undefined=null"))
    draw_validation(image, objects, projected_boxes, pixels, depth).save(output_dir / "baseline_bbox.png")
    report = dict(frame_id="000008", status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  input_sha256={name: hashlib.sha256((sample_dir / name).read_bytes()).hexdigest()
                                for name in ("000008.pkl", "image.png", "points.bin", "calib.txt")},
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__),
                  image_size=list(image.size), total_points=len(points), positive_depth=front_count,
                  inside_image=len(pixels), projection_ratio=len(pixels) / len(points),
                  annotations=len(objects), valid_objects=len(valid), ignored_regions=len(objects) - len(valid),
                  source_matrix_max_abs_error=float(np.abs(lidar2img - source_lidar2img).max()),
                  source_pixel_max_error=source_pixel_error, baseline_pixel_max_error=baseline_error,
                  ignored_points=int(association["in_ignore_region"].sum()),
                  ambiguous_3d_points=int(association["ambiguous_3d"].sum()),
                  metric_eligible_points=int(association["metric_eligible"].sum()),
                  overlapping_2d_candidates=int((association["candidate_2d"].sum(axis=1) > 1).sum()),
                  negative_controls=controls, objects=box_rows,
                  limitations=["Single frame; reduced FOV cloud, not a full Velodyne scan",
                               "Projection agreement is internal consistency with source calibration, not independent calibration accuracy",
                               "3D cuboid membership is a geometric reference, not per-point semantic annotation",
                               "2D GT candidates are an oracle diagnostic, not detector or KITTI AP performance",
                               "GT projected 3D envelope and annotated 2D bbox need not coincide under occlusion/truncation"])
    write_json(output_dir / "projection_validation.json", report)
    print(f"{report['status']}: {len(objects)} annotations, {len(valid)} objects, {len(pixels)}/{len(points)} projected points")
    print(f"Source pixel max error: {source_pixel_error:.8f} px")
    print(f"Outputs: {output_dir}")
    if not all(checks.values()):
        print("Failed checks: " + ", ".join(k for k, v in checks.items() if not v))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "data/sample")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report = run(args.sample_dir, args.output_dir)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
