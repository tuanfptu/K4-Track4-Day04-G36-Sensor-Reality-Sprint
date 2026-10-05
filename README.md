# Camera–LiDAR projection baseline

## Problem

Camera–LiDAR fusion requires an accurate transform between sensor coordinate systems.

## Baseline status

**VERIFIED PASS** with KITTI frame `000008`, distributed as [OpenMMLab MMDetection3D demo data](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). This baseline uses the original calibration without intentionally injected drift.

The pipeline transforms each Velodyne point with `Tr_velo_to_cam`, applies `R0_rect`, then projects into camera 2 with `P2`. It retains positive-depth pixels inside the RGB image and colors them by distance. The exporter checks the matrix composition against the source metadata.

## Input and output

Inputs: `data/sample/image.png`, `data/sample/points.bin`, `data/sample/calib.txt`. See [data/README.md](data/README.md) for the exact three-file download and calibration export. Outputs: `results/baseline_projection.png` and `results/baseline_metadata.json`.

## Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the sample and run `python scripts/export_kitti_calib.py` as shown in [data/README.md](data/README.md). Then run:

```powershell
python scripts/run_baseline.py
```

On the verification machine, `python` and `py` were unavailable as shell commands. The **exact verified commands** used the Codex bundled Python with NumPy 2.3.5 and Pillow 12.3.0:

```powershell
& 'C:\Users\TUAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/export_kitti_calib.py
& 'C:\Users\TUAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_baseline.py
```

Measured: 1242×375 RGB image, 17,238 LiDAR points, 17,238 positive-depth points, and 17,238 points inside the image. This MMDetection3D demo point cloud appears prefiltered to the camera field of view, so its 100% projection ratio should not be interpreted as a full-scan KITTI statistic. The saved overlay was opened and visually checked: points follow the road and visible vehicles, without obvious mirroring or global shift.
