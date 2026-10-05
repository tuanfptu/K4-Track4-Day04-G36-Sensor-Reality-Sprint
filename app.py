"""CalibGuard: live Camera–LiDAR yaw-drift classroom demonstration."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import ImageDraw

from src.calibration import load_calibration
from src.data_utils import extract_objects, load_sample_info, project_with_indices
from src.kitti_loader import load_image, load_lidar
from src.metrics import measure_yaw_drift
from src.perturbation import perturb_extrinsic
from src.visualization import overlay

ROOT = Path(__file__).resolve().parent
DEMO_HEALTH_THRESHOLDS = {"warning_median_px": 5.0, "recalibrate_median_px": 20.0}


@st.cache_data(show_spinner="Loading KITTI frame 000008…")
def load_static():
    folder = ROOT / "data/sample"
    image = load_image(folder / "image.png")
    points = load_lidar(folder / "points.bin")
    p, rect, original = load_calibration(folder / "calib.txt")
    try:
        sample = load_sample_info(folder / "000008.pkl")
        objects = [item for item in extract_objects(sample) if item["valid_3d"]]
    except FileNotFoundError:
        objects = []
    baseline = project_with_indices(points, p, rect, original, *image.size)
    return image, points, p, rect, original, objects, baseline


@st.cache_data(show_spinner=False)
def load_benchmark():
    path = ROOT / "results/final/benchmark_summary.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def with_boxes(image, objects):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for item in objects:
        draw.rectangle(tuple(item["bbox"]), outline=(245, 245, 245), width=2)
    return result


st.set_page_config(page_title="CalibGuard", page_icon="📐", layout="wide")
st.title("CalibGuard")
st.caption("Camera–LiDAR Calibration Failure Lab · KITTI frame 000008")
st.write("Camera and LiDAR are individually operational. We intentionally perturb only their extrinsic alignment and measure how fusion quality degrades.")

if "yaw" not in st.session_state:
    st.session_state.yaw = 0.0
with st.sidebar:
    st.header("Control panel")
    for value, label in [(0.0, "Baseline"), (0.5, "0.5°"), (1.0, "1.0°"), (1.5, "1.5°"), (2.0, "2.0°")]:
        if st.button(label, key=f"preset_{value}"):
            st.session_state.yaw = value
            st.rerun()
    st.slider("Yaw drift (degrees)", 0.0, 2.0, step=0.1, key="yaw")
    st.caption("LiDAR-frame yaw; each value starts from the original calibration.")
    st.caption("Team-defined demonstration thresholds. Not an automotive safety standard.")

try:
    image, points, p, rect, original, objects, baseline = load_static()
    yaw = float(st.session_state.yaw)
    if not np.isfinite(yaw) or not 0 <= yaw <= 2:
        raise ValueError("Yaw must be between 0° and 2°")
    current = project_with_indices(points, p, rect, perturb_extrinsic(original, yaw), *image.size)
    metrics, per_object = measure_yaw_drift(baseline, current, objects)
    current_visible = current["in_image"]
    current_image = overlay(image, current["pixels"][current_visible], current["depth"][current_visible])
    baseline_visible = baseline["in_image"]
    baseline_image = overlay(image, baseline["pixels"][baseline_visible], baseline["depth"][baseline_visible])
except (OSError, ValueError, KeyError) as error:
    st.error(f"Unable to load or project the bundled sample: {error}")
    st.stop()

st.info("VERIFIED BASELINE" if yaw == 0 else "CONTROLLED CALIBRATION DRIFT")
median = metrics["median_reprojection_px"]
if median is None:
    health = "WARNING · no shared visible points"
elif median < DEMO_HEALTH_THRESHOLDS["warning_median_px"]:
    health = "HEALTHY"
elif median < DEMO_HEALTH_THRESHOLDS["recalibrate_median_px"]:
    health = "WARNING"
else:
    health = "RECALIBRATION NEEDED"
st.badge(health)

current_tab, compare_tab, benchmark_tab, object_tab, method_tab = st.tabs(
    ["Current projection", "Baseline vs drift", "Benchmark", "Per-object analysis", "Method / limitations"]
)
with current_tab:
    picture, figures = st.columns([3, 2])
    with picture:
        st.image(current_image, alt=f"KITTI camera image with LiDAR projection at {yaw:.1f} degrees")
        st.caption(f"{int(current_visible.sum()):,} visible LiDAR points · yaw {yaw:.1f}°")
    with figures:
        st.metric("Median displacement", f"{median:.2f} px" if median is not None else "—", border=True)
        st.metric("P90 displacement", f"{metrics['p90_reprojection_px']:.2f} px" if metrics["p90_reprojection_px"] is not None else "—", border=True)
        st.metric("Association retention", f"{metrics['association_retention_pct']:.2f}%" if metrics["association_retention_pct"] is not None else "—", border=True)
        st.metric("Association drop", f"{metrics['association_drop_pct']:.2f}%" if metrics["association_drop_pct"] is not None else "—", border=True)
        st.metric("Out of frame", f"{metrics['out_of_frame_pct']:.2f}%", border=True)
with compare_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Original calibration · 0°")
        st.image(baseline_image, alt="Baseline LiDAR points aligned to KITTI image")
    with right:
        st.subheader(f"Current yaw · {yaw:.1f}°")
        st.image(current_image, alt="Drifted LiDAR points on the same KITTI image")
    st.caption("Same image and point cloud; only the LiDAR-to-camera extrinsic changes.")
with benchmark_tab:
    fixed = load_benchmark()
    if fixed is None:
        st.warning("Fixed benchmark CSV is missing. Run `python scripts/benchmark.py`.")
    else:
        st.caption("Fixed protocol: 0°, 0.5°, 1°, 1.5°, 2°; these values are read from results/final/benchmark_summary.csv.")
        st.line_chart(fixed, x="yaw_deg", y=["median_reprojection_px", "p90_reprojection_px"],
                      alt="Median and P90 displacement rise with yaw")
        st.line_chart(fixed, x="yaw_deg", y=["association_retention_pct", "association_drop_pct"],
                      alt="Association retention and drop across yaw levels")
        st.line_chart(fixed, x="yaw_deg", y="out_of_frame_pct", alt="Out-of-frame rate across yaw levels")
        st.dataframe(fixed, hide_index=True, alt="Five-row fixed yaw benchmark summary")
with object_tab:
    if not per_object:
        st.info("No labeled camera boxes are available for this frame.")
    else:
        labels = [f"#{item['object_id']} · {item['class']}" for item in per_object]
        selected = st.selectbox("Camera bbox", labels)
        row = per_object[labels.index(selected)]
        st.image(with_boxes(current_image, [objects[labels.index(selected)]]),
                 alt="Current LiDAR projection with selected fixed camera box")
        st.write(f"BBox: {row['bbox']}")
        st.write(f"Baseline associations: {row['baseline_associated_points']}; retained: {row['retained_points']}")
        if row["association_retention_pct"] is None:
            st.info("This bbox has no baseline LiDAR associations; retention is undefined.")
        else:
            st.metric("Object retention", f"{row['association_retention_pct']:.2f}%")
            st.metric("Object drop", f"{row['association_drop_pct']:.2f}%")
        object_path = ROOT / "results/final/benchmark_per_object.csv"
        if object_path.is_file():
            table = pd.read_csv(object_path)
            subset = table[table["object_id"] == row["object_id"]]
            st.line_chart(subset, x="yaw_deg", y="association_retention_pct",
                          alt="Fixed benchmark retention for selected bbox")
with method_tab:
    st.subheader("What changes")
    st.write("Input: RGB camera + LiDAR + original extrinsic calibration. Perturbation: right-composed LiDAR-frame yaw at the LiDAR origin. Output: matched-point reprojection displacement, fixed-bbox association retention, and visibility loss.")
    st.write("Reprojection displacement is relative to the 0° baseline, using the same original point IDs visible in both views. It is not independent calibration ground-truth error.")
    st.write("Association is a fixed 2D-bbox proxy; overlapping boxes contribute separate point-box pairs. It does not verify 3D object ownership.")
    st.warning("Limitations: one frame; demo cloud appears camera-FOV filtered; bbox association is a proxy; downstream multimodal detector mAP is NOT EVALUATED.")
    st.caption("Research context: Galibr, CalibRefine, DF-Calib / UniCalib. This sprint does not reproduce those methods.")
    st.caption("Team-defined demonstration thresholds. Not an automotive safety standard.")
