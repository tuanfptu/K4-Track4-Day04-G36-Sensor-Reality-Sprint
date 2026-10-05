# Camera–LiDAR projection baseline

## Problem

Camera–LiDAR fusion requires an accurate transform between sensor coordinate systems.

## Baseline status

**VERIFIED PASS** with bundled KITTI frame `000008`, distributed as [OpenMMLab MMDetection3D demo data](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). Dataset origin: KITTI. Demo sample distribution: OpenMMLab MMDetection3D. Purpose: small reproducible classroom benchmark. Team G36 did not create the KITTI sample. This baseline uses the original calibration without intentionally injected drift.

The pipeline transforms each Velodyne point with `Tr_velo_to_cam`, applies `R0_rect`, then projects into camera 2 with `P2`. It retains positive-depth pixels inside the RGB image and colors them by distance. The exporter checks the matrix composition against the source metadata.

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

## Visualization + Research — Lâm

Phần visualization minh họa calibration drift trên mẫu `000008` đã có trong repo. Chạy từ repo root sau khi cài `requirements.txt`:

```powershell
python scripts/run_visualization.py
Start-Process .\results\visualization\demo.html
```

Demo HTML chạy offline, có slider camera yaw **0–2°**, overlay cyan/orange, chế độ cạnh nhau, bbox cố định và vector dịch chuyển. Slider chọn 9 mức đã tính trước, cách nhau 0.25°. Chỉ lấy mẫu tối đa 3,500 point IDs khi vẽ trên trình duyệt; metric dùng toàn bộ 17,238 điểm.

Các output nằm trong `results/visualization/`:

| File | Nội dung |
| --- | --- |
| `comparison.png` | Baseline, corrupted và chồng lớp tại mức yaw lớn nhất |
| `metrics.png` | Median/P90 displacement, association retention và association drop |
| `demo.html` | Demo tự chứa ảnh, dữ liệu và chart; mở trực tiếp trên trình duyệt |
| `demo_benchmark.csv` | Metric đo trên từng mức yaw, dùng để tái tạo plot |
| `demo_metadata.json` | Protocol, giới hạn và SHA-256 của input |

**Diễn giải:** error ở đây là khoảng cách pixel so với calibration baseline của cùng điểm còn trong ảnh ở cả hai phép chiếu. Association là proxy giữ lại cặp điểm–bbox trong cùng bbox 2D, chưa xác nhận ownership vật thể 3D. Demo một frame và point cloud đã lọc FOV không thay thế benchmark trên tập dữ liệu. **Chưa đánh giá detector mAP**; cột `map` để trống.

Kết quả demo đã chạy local cho frame `000008`, 6 bbox hợp lệ:

| Camera yaw | Median / P90 displacement | Association retention | Association drop | Điểm trong ảnh |
| --- | --- | --- | --- | --- |
| 0° | 0 / 0 px | 100% | 0 pp | 17,238 |
| 2° | 27.88 / 37.44 px | 84.86% | 15.14 pp | 16,697 |

Ở baseline có 10,390 cặp điểm–bbox; ở 2° giữ lại 8,817 cặp. Số cặp tính riêng cho mỗi bbox khi có chồng lấp. Các giá trị đầy đủ nằm trong `demo_benchmark.csv`.

Để vẽ CSV của benchmark nhóm sau khi chuẩn hóa theo [CSV contract và cách tích hợp](docs/VISUALIZATION_HANDOFF.md):

```powershell
python scripts/run_visualization.py --benchmark-csv results/benchmark.csv --output-dir results/team_plots
```

CSV có kết quả mAP thật và `map_protocol` sẽ sinh thêm biểu đồ mAP/drop; dữ liệu thiếu không được điền thành 0. Script chỉ đọc CSV, không chạy detector hoặc sinh mAP.

Research Galibr, CalibRefine và DF-Calib/UniCalib, cùng link paper/code và giới hạn áp dụng: [references/SOURCES.md](references/SOURCES.md). Các phương pháp này được tra cứu, chưa được tái lập trong sprint.

Kiểm tra phần visualization:

```powershell
python -m unittest discover -s tests -v
```
