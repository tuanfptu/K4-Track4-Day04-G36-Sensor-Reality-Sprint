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

## Mô phỏng sai lệch hiệu chuẩn ngoại tại (yaw)

Module `src/perturbation.py` cung cấp hàm `perturb_extrinsic(transform, yaw_deg=0.0)`
để mô phỏng sai lệch góc yaw của ma trận biến đổi từ LiDAR sang camera.
Đầu vào là ma trận `Tr_velo_to_cam` gốc kích thước 3×4; đầu ra là một ma trận
3×4 mới, có kiểu dữ liệu `float64`. Hàm không sửa đổi ma trận đầu vào.

Phép quay được định nghĩa trong **hệ tọa độ LiDAR**: X hướng về phía trước,
Y hướng sang trái, Z hướng lên trên. Yaw quay quanh trục Z của LiDAR theo
quy tắc bàn tay phải: góc dương quay trục X về phía trục Y. Tâm quay là
gốc tọa độ LiDAR.

Khi biểu diễn bằng ma trận đồng nhất 4×4, thứ tự kết hợp là
`T_drift = T_original @ delta_T_lidar`. Do đó,
`R_drift = R_original @ Rz(yaw)` và cột tịnh tiến được giữ nguyên.
Đây là phép chủ động thêm sai lệch vào ma trận biến đổi; để đảo ngược sai lệch,
áp dụng góc yaw đối dấu. Phép quay này dùng trục Z của LiDAR;
các ma trận nội tại `P2` và hiệu chỉnh `R0_rect` được giữ nguyên.

```python
from pathlib import Path
from src.calibration import load_calibration
from src.perturbation import perturb_extrinsic

p, rect, transform = load_calibration(Path("data/sample/calib.txt"))
drifted_transform = perturb_extrinsic(transform, yaw_deg=2.0)
# Dùng drifted_transform thay cho transform khi gọi hàm project.
```

Các mức yaw đề xuất cho thí nghiệm: **0°, 0.5°, 1°, 1.5°, 2°**.
Mỗi lần gọi hàm cần dùng ma trận gốc để tránh cộng dồn sai lệch giữa các mức.
Tham số `yaw_deg` có đơn vị là độ; hàm hỗ trợ mọi góc vô hướng hữu hạn.
Các trường hợp kiểm thử ±90° giúp kiểm tra rõ trục quay và dấu của góc;
chúng không phải mức sai lệch dùng trong benchmark 0–2°.

Pitch và tịnh tiến là các phần mở rộng tùy chọn, hiện chưa được triển khai.
Phạm vi của module là tạo ma trận bị lệch; việc tính metric và chạy benchmark
do pipeline thí nghiệm đảm nhiệm.

Bộ kiểm thử trong `tests/test_perturbation.py` kiểm tra góc 0°, trục và dấu
của phép quay, thứ tự nhân ma trận, tính trực giao của ma trận quay, cột
tịnh tiến, việc giữ nguyên đầu vào và xử lý dữ liệu không hợp lệ.
Các kiểm thử tích hợp với KITTI xác nhận yaw 0° tái tạo phép chiếu baseline
và yaw 2° làm thay đổi vị trí pixel khi giữ nguyên nội tại và hiệu chỉnh.

Chạy kiểm thử từ thư mục gốc của dự án bằng lệnh dưới đây.
Chỉ cần các thư viện hiện có trong `requirements.txt`; không cần cài thêm
thư viện kiểm thử:

```powershell
python -m unittest discover -s tests -v
```

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
