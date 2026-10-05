# CalibGuard — Camera–LiDAR calibration drift demo

Team G36's reproducible classroom benchmark projects a KITTI LiDAR frame onto its synchronized camera image, then measures how injected extrinsic yaw changes pixel alignment and fixed 2D box association. The small demo frame is included; no separate KITTI download is required.

## Clone and run

From the repository root, with Python and `pip` installed:

```bash
git clone https://github.com/tuanfptu/K4-Track4-Day04-G36-Sensor-Reality-Sprint.git
cd K4-Track4-Day04-G36-Sensor-Reality-Sprint
pip install -r requirements.txt
python scripts/run_baseline.py
```

The baseline writes `results/baseline_projection.png` and `results/baseline_metadata.json`. To regenerate the final experiment and launch the interactive demo:

```bash
python scripts/benchmark.py
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. On Windows, `py` can replace `python` if it is the configured Python command. A virtual environment is recommended but optional.

## Demo data and provenance

| Item | Detail |
| --- | --- |
| Dataset origin | KITTI |
| Demo sample distribution | [OpenMMLab MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti) |
| Frame | `000008` |
| Purpose | Small reproducible classroom benchmark |

The synchronized sample is committed as `data/sample/image.png`, `data/sample/points.bin`, `data/sample/000008.pkl`, and `data/sample/calib.txt`. Team G36 did not create the KITTI data. See [source and license notes](references/SOURCES.md).

## Experiment protocol

The baseline applies `Tr_velo_to_cam`, `R0_rect`, and `P2`, then keeps positive-depth pixels inside the image. The final benchmark injects **yaw about the LiDAR Z axis** at 0°, 0.5°, 1°, 1.5°, and 2°. Every trial starts from the original calibration: `T_drift = T_original @ Rz_lidar(yaw)`. Translation, camera projection, rectification, image, point cloud, and 2D boxes remain fixed.

Metrics match points by their original LiDAR indices. Reprojection displacement compares points visible in both projections. Association retention is the share of baseline point–box pairs still inside the same fixed 2D box; overlapping boxes contribute separate pairs. Out-of-frame rate measures baseline-visible points that leave the image. These are geometry and 2D association proxies, **not** an independent calibration ground truth or detector mAP.

Measured on the bundled frame (17,238 points):

| LiDAR yaw | Median displacement | P90 displacement | Association retention | Association drop | Out of frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0° | 0.00 px | 0.00 px | 100.00% | 0.00% | 0.00% |
| 0.5° | 7.19 px | 9.94 px | 96.56% | 3.44% | 0.58% |
| 1° | 14.37 px | 19.79 px | 92.90% | 7.10% | 1.19% |
| 1.5° | 21.53 px | 29.56 px | 89.59% | 10.41% | 1.76% |
| 2° | 28.68 px | 39.20 px | 86.16% | 13.84% | 2.34% |

The source of record is [`results/final/benchmark_summary.csv`](results/final/benchmark_summary.csv); [`benchmark_metadata.json`](results/final/benchmark_metadata.json) records the protocol and limits. `python scripts/benchmark.py` regenerates those files, per-object results, yaw overlays, a 0°/2° comparison, and metric plots under `results/final/`.

## Presenting the demo

In a 3–5 minute walkthrough, show the 0° projection, increase yaw in CalibGuard, compare point positions on the road and vehicles, and read the median/P90 displacement and association retention. The app's health thresholds are team-defined presentation cues, not an automotive safety standard.

This is one camera-field-of-view demo frame; its point cloud appears prefiltered. The 100% baseline in-image count is not representative of a full KITTI scan. No multimodal detector was run, and detector mAP was not evaluated. The measured association does not establish 3D object ownership.

## Verification and project files

```bash
python scripts/validate_sample.py
python -m unittest discover -s tests -v
```

Sample validation writes `results/baseline_bbox.png` and `results/association_000008.npz`. [Validation and association details](docs/DINH_VALIDATION.md) explain the checks and data contract. The implementation is in `src/`; baseline, validation, and benchmark entry points are in `scripts/`.

The earlier [visualization handoff](docs/VISUALIZATION_HANDOFF.md) documents `python scripts/run_visualization.py`, an offline **camera-frame yaw** teaching demo in `results/visualization/`. Its coordinate convention differs from the final LiDAR-frame benchmark, so its numbers should be read separately. Research links and provenance are in [SOURCES.md](references/SOURCES.md). The five-person roster is in [TEAMMATES.md](TEAMMATES.md).
