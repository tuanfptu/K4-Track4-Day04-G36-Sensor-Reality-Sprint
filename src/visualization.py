import numpy as np
from PIL import Image, ImageDraw


def overlay(image: Image.Image, pixels: np.ndarray, depth: np.ndarray) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    low, high = np.percentile(depth, [5, 95])
    span = max(float(high - low), 1e-9)
    for (u, v), z in zip(pixels, depth):
        t = float(np.clip((z - low) / span, 0, 1))
        color = (int(255 * (1 - t)), int(220 * (1 - abs(2 * t - 1))), int(255 * t))
        x, y = int(round(u)), int(round(v))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return result
