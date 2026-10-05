import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration
from src.kitti_loader import load_gt_boxes, load_image, load_lidar
from src.metrics import association_retention, inside_image_ratio, reprojection_displacement
from src.perturbation import perturb_extrinsics
from src.projection import project_indexed

ROT_LEVELS_DEG = [0.5, 1.0, 1.5, 2.0]
TRANS_LEVELS_M = [0.02, 0.05, 0.10]
COMBINED_LEVELS = [(0.5, 0.02), (1.0, 0.05), (2.0, 0.10)]  # (degrees, metres)
AXES = {"pitch": (0, "rot"), "yaw": (1, "rot"), "roll": (2, "rot"),
        "trans_x": (0, "trans"), "trans_y": (1, "trans"), "trans_z": (2, "trans")}


def configs(rng):
    """Yield (type, level, unit, rot_deg, trans_m); sign / direction is random per seed."""
    yield "baseline", "0", "-", (0, 0, 0), (0, 0, 0)
    for name, (axis, kind) in AXES.items():
        for level in (ROT_LEVELS_DEG if kind == "rot" else TRANS_LEVELS_M):
            vec = np.zeros(3)
            vec[axis] = level * rng.choice([-1, 1])
            if kind == "rot":
                yield name, str(level), "deg", tuple(vec), (0, 0, 0)
            else:
                yield name, str(level), "m", (0, 0, 0), tuple(vec)
    for deg, metres in COMBINED_LEVELS:
        r, t = rng.normal(size=3), rng.normal(size=3)
        yield ("combined", f"{deg}deg+{int(metres * 100)}cm", "deg+m",
               tuple(deg * r / np.linalg.norm(r)), tuple(metres * t / np.linalg.norm(t)))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-drift benchmark on the KITTI 000008 sample")
    parser.add_argument("--seeds", type=int, default=20, help="random sign/direction repeats per level")
    parser.add_argument("--out", type=Path, default=ROOT / "results/benchmark.csv")
    args = parser.parse_args()

    image = load_image(ROOT / "data/sample/image.png")
    points = load_lidar(ROOT / "data/sample/points.bin")
    p, rect, transform = load_calibration(ROOT / "data/sample/calib.txt")
    gt = load_gt_boxes(ROOT / "data/sample/000008.pkl")
    pix_ref, _ = project_indexed(points, p, rect, transform)

    runs, per_object, order = [], [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for name, level, unit, rot, trans in configs(rng):
            if (name, level) not in order:
                order.append((name, level))
            pix, _ = project_indexed(points, p, rect, perturb_extrinsics(transform, rot, trans))
            disp = reprojection_displacement(pix_ref, pix)
            assoc = association_retention(pix_ref, pix, gt["boxes"])
            runs.append({
                "perturb_type": name, "level": level, "unit": unit, "seed": seed,
                "n_compared": disp["n_compared"], "valid_ratio": round(disp["valid_ratio"], 4),
                "median_px": round(disp["median_px"], 3), "p90_px": round(disp["p90_px"], 3),
                "retention": round(assoc["retention"], 4),
                "inside_image_ratio": round(inside_image_ratio(pix, *image.size), 4),
            })
            for k in range(len(gt["boxes"])):
                per_object.append({"perturb_type": name, "level": level, "box": k,
                                   "depth_m": round(float(gt["depth"][k]), 1),
                                   "n_points": int(assoc["per_box_n"][k]),
                                   "retention": float(assoc["per_box_retention"][k])})

    write_csv(args.out, runs)

    summary = []
    for name, level in order:
        sel = [r for r in runs if r["perturb_type"] == name and r["level"] == level]
        summary.append({"perturb_type": name, "level": level, "unit": sel[0]["unit"], "n_runs": len(sel),
                        **{m: round(float(np.mean([r[m] for r in sel])), 3)
                           for m in ("median_px", "p90_px", "retention", "valid_ratio", "inside_image_ratio")}})
    write_csv(args.out.with_name("benchmark_summary.csv"), summary)

    obj_rows = []
    for name, level in order:
        for k in range(len(gt["boxes"])):
            sel = [r for r in per_object if r["perturb_type"] == name and r["level"] == level and r["box"] == k]
            obj_rows.append({"perturb_type": name, "level": level, "box": k, "depth_m": sel[0]["depth_m"],
                             "n_points": sel[0]["n_points"],
                             "retention": round(float(np.nanmean([r["retention"] for r in sel])), 4)})
    write_csv(args.out.with_name("benchmark_per_object.csv"), obj_rows)

    print(f"{len(points)} points, {len(gt['boxes'])} GT boxes, {args.seeds} seeds, {len(runs)} runs")
    print(f"{'type':10s} {'level':>12s} {'median_px':>10s} {'p90_px':>8s} {'retention':>10s}")
    for r in summary:
        print(f"{r['perturb_type']:10s} {r['level']:>12s} {r['median_px']:10.2f} {r['p90_px']:8.2f} {r['retention']:10.3f}")
    print(f"Wrote {args.out}, benchmark_summary.csv, benchmark_per_object.csv")


if __name__ == "__main__":
    main()
