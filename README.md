# Camera–LiDAR projection baseline

## Problem

Camera–LiDAR fusion requires an accurate transform between sensor coordinate systems.

## Baseline status

**VERIFIED PASS** with bundled KITTI frame `000008`, distributed as [OpenMMLab MMDetection3D demo data](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). Dataset origin: KITTI. Demo sample distribution: OpenMMLab MMDetection3D. Purpose: small reproducible classroom benchmark. Team G36 did not create the KITTI sample. This baseline uses the original calibration without intentionally injected drift.

The pipeline transforms each Velodyne point with `Tr_velo_to_cam`, applies `R0_rect`, then projects into camera 2 with `P2`. It retains positive-depth pixels inside the RGB image and colors them by distance. The exporter checks the matrix composition against the source metadata.

## Ground truth and association handoff

For frame `000008`, run `python scripts/validate_sample.py` after installing the
dependencies. This extracts every `.pkl` object and ignored region, checks the
existing projection against the source matrices, creates
`results/baseline_bbox.png`, and exports `results/association_000008.npz` with
point indices and 2D/3D membership arrays. See [Dinh's validation and data
contract](docs/DINH_VALIDATION.md) for results, metric definitions, and limits.
Run `python -m unittest discover -s tests -v` for the geometry and frame checks.

## Input and output

Inputs are bundled: `data/sample/image.png`, `data/sample/points.bin`, `data/sample/calib.txt`. The source `data/sample/000008.pkl` is included for calibration provenance. Outputs: `results/baseline_projection.png` and `results/baseline_metadata.json`.

## Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

After cloning and installing dependencies, run:

```powershell
python scripts/run_baseline.py
```

On the verification machine, `python` and `py` were unavailable as shell commands. The **exact verified commands** used the Codex bundled Python with NumPy 2.3.5 and Pillow 12.3.0:

```powershell
& 'C:\Users\TUAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_baseline.py
```

Measured: 1242×375 RGB image, 17,238 LiDAR points, 17,238 positive-depth points, and 17,238 points inside the image. This MMDetection3D demo point cloud appears prefiltered to the camera field of view, so its 100% projection ratio should not be interpreted as a full-scan KITTI statistic. The saved overlay was opened and visually checked: points follow the road and visible vehicles, without obvious mirroring or global shift.
