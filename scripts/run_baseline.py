import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import load_calibration
from src.kitti_loader import load_image, load_lidar
from src.projection import project
from src.visualization import overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Project KITTI Velodyne points onto camera 2")
    parser.add_argument("--image", type=Path, default=ROOT / "data/sample/image.png")
    parser.add_argument("--lidar", type=Path, default=ROOT / "data/sample/points.bin")
    parser.add_argument("--calib", type=Path, default=ROOT / "data/sample/calib.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline_projection.png")
    parser.add_argument("--frame-id", default="000008", help="Original KITTI frame ID (all three inputs must match)")
    args = parser.parse_args()
    try:
        image = load_image(args.image)
        points = load_lidar(args.lidar)
        p, rect, transform = load_calibration(args.calib)
        pixels, depth, front_count = project(points, p, rect, transform, *image.size)
        if not len(pixels):
            raise ValueError("No LiDAR points projected inside the image; check matching inputs and calibration")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        overlay(image, pixels, depth).save(args.output)
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise RuntimeError("Output image was not created")
        metadata_path = args.output.parent / "baseline_metadata.json"
        metadata = {
            "dataset": "KITTI",
            "sample_source": "OpenMMLab MMDetection3D demo/data/kitti",
            "frame_id": args.frame_id,
            "image_file": str(args.image.relative_to(ROOT)) if args.image.is_relative_to(ROOT) else str(args.image),
            "lidar_file": str(args.lidar.relative_to(ROOT)) if args.lidar.is_relative_to(ROOT) else str(args.lidar),
            "calibration_file": str(args.calib.relative_to(ROOT)) if args.calib.is_relative_to(ROOT) else str(args.calib),
            "total_lidar_points": len(points),
            "positive_depth_points": front_count,
            "projected_inside_image": len(pixels),
            "projection_ratio": len(pixels) / len(points),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Image size: {image.width}x{image.height}")
        print(f"Total LiDAR points: {len(points)}")
        print(f"Points with positive camera depth: {front_count}")
        print(f"Points inside image bounds: {len(pixels)}")
        print(f"Projection ratio: {100 * len(pixels) / len(points):.2f}%")
        print(f"Output: {args.output}")
        print(f"Metadata: {metadata_path}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
