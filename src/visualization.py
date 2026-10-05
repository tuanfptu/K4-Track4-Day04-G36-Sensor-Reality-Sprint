"""Projection overlays and a portable, offline calibration-drift viewer."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASELINE_COLOR = (36, 211, 238)
CORRUPTED_COLOR = (255, 131, 92)


def overlay(image: Image.Image, pixels: np.ndarray, depth: np.ndarray) -> Image.Image:
    result = image.copy()
    if len(pixels) != len(depth):
        raise ValueError("Pixels and depth must have the same number of points")
    if not len(pixels):
        return result
    draw = ImageDraw.Draw(result)
    low, high = np.percentile(depth, [5, 95])
    span = max(float(high - low), 1e-9)
    for (u, v), z in zip(pixels, depth):
        t = float(np.clip((z - low) / span, 0, 1))
        color = (int(255 * (1 - t)), int(220 * (1 - abs(2 * t - 1))), int(255 * t))
        x, y = int(round(u)), int(round(v))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return result


def _font(size: int):
    # DejaVu is bundled by Matplotlib on every supported platform.
    try:
        from matplotlib import get_data_path
        return ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)
    except (ImportError, OSError):
        return ImageFont.load_default()


def _visible(pixels, image):
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Pixels must have shape (N, 2)")
    return (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image.width) & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image.height))


def comparison_overlay(image, baseline_pixels, corrupted_pixels, boxes=(),
                       label="Corrupted calibration") -> Image.Image:
    """Three panels with fixed colors; each row keeps the original image size.

    Point rows must identify the same LiDAR returns in both arrays. NaN or
    off-image projections are omitted, never paired with a different point.
    """
    baseline_pixels = np.asarray(baseline_pixels, dtype=float)
    corrupted_pixels = np.asarray(corrupted_pixels, dtype=float)
    if baseline_pixels.shape != corrupted_pixels.shape:
        raise ValueError("Baseline and corrupted pixels must share point IDs and shape")
    masks = [_visible(p, image) for p in (baseline_pixels, corrupted_pixels)]
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    header = 52
    result = Image.new("RGB", (image.width, 3 * (image.height + header)), (15, 23, 42))
    headings = ["BASELINE | original calibration", f"CORRUPTED | {label}",
                "COMPARISON | cyan: baseline   orange: corrupted   white: fixed 2D boxes"]
    for index, heading in enumerate(headings):
        panel = image.convert("RGB").copy()
        draw = ImageDraw.Draw(panel)
        for box in boxes:
            draw.rectangle(tuple(box), outline=(245, 245, 245), width=2)
        series = [0] if index == 0 else [1] if index == 1 else [0, 1]
        for number in series:
            pixels = (baseline_pixels, corrupted_pixels)[number]
            color = (BASELINE_COLOR, CORRUPTED_COLOR)[number]
            for u, v in pixels[masks[number]]:
                draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
        y = index * (image.height + header)
        result.paste(panel, (0, y + header))
        ImageDraw.Draw(result).text((18, y + 15), heading, fill="white", font=_font(17))
    return result


def read_benchmark_csv(path):
    """Read the documented team CSV contract without loading Matplotlib."""
    from src.benchmark_plot import read_benchmark_csv as read
    return read(path)


def plot_benchmark(rows, output_dir):
    """Plot measured displacement, association and optional detector mAP."""
    from src.benchmark_plot import plot_benchmark as plot
    return plot(rows, output_dir)


def write_demo_html(image, baseline_pixels, frames, boxes, output, plot_path,
                    max_display_points=3500):
    """Write a self-contained viewer; full-cloud metrics accompany sampled dots.

    frames contains {yaw_deg, pixels (N,2), metrics}; baseline_pixels and each
    pixels array must preserve original LiDAR row IDs. No HTTP server is needed.
    """
    if not frames or max_display_points < 1:
        raise ValueError("The viewer needs frames and a positive point display limit")
    baseline_pixels = np.asarray(baseline_pixels, dtype=float)
    _visible(baseline_pixels, image)
    count = len(baseline_pixels)
    indices = np.linspace(0, count - 1, min(count, max_display_points), dtype=int) if count else np.array([], dtype=int)

    def pack(pixels):
        pixels = np.asarray(pixels, dtype=float)
        if pixels.shape != baseline_pixels.shape:
            raise ValueError("Every frame must preserve baseline point IDs")
        return [[round(float(u), 2), round(float(v), 2)] if np.isfinite([u, v]).all()
                else None for u, v in pixels[indices]]

    jpeg = io.BytesIO()
    image.convert("RGB").save(jpeg, format="JPEG", quality=92)
    data = {"width": image.width, "height": image.height, "point_count": count,
            "display_count": len(indices), "baseline": pack(baseline_pixels),
            "boxes": np.asarray(boxes, dtype=float).reshape(-1, 4).tolist(),
            "frames": [{"yaw_deg": float(f["yaw_deg"]), "pixels": pack(f["pixels"]),
                        "metrics": f["metrics"]} for f in frames]}
    template = Path(__file__).with_name("drift_viewer.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # JSON is data even when a future caller supplies text containing HTML.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = (template.replace("__DEMO_DATA__", payload)
            .replace("__IMAGE_DATA__", base64.b64encode(jpeg.getvalue()).decode("ascii"))
            .replace("__PLOT_DATA__", base64.b64encode(Path(plot_path).read_bytes()).decode("ascii")))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
