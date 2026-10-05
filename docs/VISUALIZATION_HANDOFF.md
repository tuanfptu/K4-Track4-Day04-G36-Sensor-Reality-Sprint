# Visualization handoff

Owner: Lâm. Scope: baseline/corrupted overlay, plots, offline visual demo, research.

## Chạy demo

```powershell
python -m pip install -r requirements.txt
python scripts/run_visualization.py
Start-Process .\results\visualization\demo.html
```

Mặc định 9 mức camera yaw từ 0 đến 2°, bước 0.25°. Có thể chọn grid khác trong cùng khoảng, luôn gồm 0:

```powershell
python scripts/run_visualization.py --yaws 0 0.5 1 2 --output-dir results/custom_demo
```

Ảnh static chọn mức yaw lớn nhất trong grid. HTML tự chứa input để mở offline; thay grid cần chạy lại script. CLI demo chỉ dùng mẫu KITTI `000008` đồng bộ trong repo.

## Ghép với perturbation và metrics của nhóm

`src/visualization.py` cung cấp:

```python
from src.visualization import comparison_overlay, read_benchmark_csv, plot_benchmark

# baseline_uv và corrupted_uv: (N,2), cùng row = cùng LiDAR point ID.
# Có thể giữ NaN cho điểm không chiếu được và UV ngoài ảnh cho điểm rơi khỏi FOV.
panel = comparison_overlay(image, baseline_uv, corrupted_uv, boxes, label="yaw = 2 deg")
panel.save("results/comparison.png")
plots = plot_benchmark(read_benchmark_csv("results/benchmark.csv"), "results/plots")
```

`src/projection.py:project()` hiện trả điểm đã lọc. Không ghép hai mảng trả về theo index nếu tập điểm visible thay đổi: mỗi index có thể trở thành point ID khác. `src/drift_demo.py:project_aligned()` giữ N row của point cloud ban đầu, dùng riêng cho adapter demo này. Nhóm có thể cung cấp projector khác nếu giữ mapping point IDs.

Trước benchmark chung, thống nhất với Khánh về hệ tọa độ perturbation. Demo dùng trục **camera +Y**, camera x phải/y xuống/z trước, với:

```text
T_corrupt = R_y_camera(yaw) @ T_original
uv = project(P2, R0_rect, T_corrupt, points)
```

Phép xoay tác động lên extrinsic gốc **trước rectification**; P2/R0_rect cố định. Đây là protocol minh họa. Nếu `perturbation.py` dùng yaw quanh LiDAR +Z hoặc phép nhân phía phải, hãy ghi condition khác và dùng output của module đó, tránh gộp các sweep khác nghĩa.

## CSV contract

Một row cho một mức drift của một condition. Header tối thiểu:

```csv
condition,drift_value,drift_unit,median_error_px,p90_error_px,association_retention
```

| Cột | Quy ước |
| --- | --- |
| `condition` | Tên sweep, ví dụ `camera_yaw`, `lidar_yaw`, `camera_tx`; không trộn các thí nghiệm trong cùng tên |
| `drift_value` | Giá trị hữu hạn; có thể có dấu trong CSV benchmark |
| `drift_unit` | Đơn vị, ví dụ `deg`, `m` |
| `median_error_px`, `p90_error_px` | Displacement pixel không âm; blank nếu undefined |
| `association_retention` | Tỷ lệ [0,1], **không dùng phần trăm 0–100**; blank nếu không có mẫu số |
| `association_drop_pp` | Optional; `100 * (1 - retention)`, đơn vị điểm phần trăm; plotter có thể suy ra |
| `map` | Optional; mAP thật trong [0,1]; blank nếu chưa đánh giá |
| `map_protocol` | Bắt buộc khi có `map`: tên task/evaluator/IoU/classes/split, dùng cùng protocol trong mỗi series |

Plotter tách series theo `(condition, drift_unit)`, không nối yaw với translation. NaN/Inf, trùng mức drift, metric ngoài miền, drop không khớp retention và mAP thiếu protocol/baseline sẽ báo lỗi. Metric thiếu giữ khoảng trống trên curve.

mAP drop tính bằng `100 * (mAP_at_zero_drift - mAP_at_current_drift)`. Nếu mAP tăng thì drop có thể âm. Cần row zero-drift có mAP thật trong mỗi series có mAP; không suy ra mAP từ bbox retention.

Các count bổ sung trong `demo_benchmark.csv` giải thích mẫu số và dropout: `baseline_visible`, `corrupted_visible`, `matched_visible`, `baseline_associations`, `retained_associations`. Đây là số đo một frame, chưa phải aggregate dataset.

## Metric demo

- **Displacement:** L2 giữa baseline và corrupted UV của cùng point ID, chỉ trên intersection visible. Median/P90 undefined nếu intersection rỗng. Cần công bố dropout để tránh bias khi chỉ giữ điểm còn visible.
- **Association retention proxy:** baseline membership là cặp `(point_id, box_id)` có UV baseline visible và nằm trong bbox. Sau drift giữ cặp nếu cùng điểm vẫn visible và nằm trong cùng bbox cố định. Box biên dùng inclusive; image bounds dùng `[0,width) × [0,height)`. Bbox chồng lấp đếm từng cặp riêng, không chọn tùy ý một box. Mẫu số 0 → undefined.
- **GT bbox:** lấy 6 bbox có `bbox_label >= 0` từ metadata sample; bỏ 4 DontCare. Đây là bbox 2D được cung cấp, chưa kiểm chứng correspondence điểm–vật thể 3D. Loader chỉ dùng các constructor đã allowlist của bundled sample pickle.
- **mAP:** chưa có detector predictions/evaluator nên không đánh giá trong demo.

## Research và pitch

[SOURCES.md](../references/SOURCES.md) ghi nguồn primary và tóm tắt ba hướng: Galibr (ground/cạnh), CalibRefine (object association/homography), DF-Calib/UniCalib (depth flow). DF-Calib là tên arXiv v1; bản WACV 2026 mang tên UniCalib. Những mũi tên trên demo là displacement hình học từ perturbation đã biết, chưa phải flow do mạng dự đoán.

Demo khi pitch: mở HTML ở 0°, kéo tới 2°, chuyển cạnh nhau, bật vector, chỉ vào chart và diễn giải error/association. Kết thúc bằng giới hạn một frame và nhu cầu benchmark detector để đo mAP drop.
