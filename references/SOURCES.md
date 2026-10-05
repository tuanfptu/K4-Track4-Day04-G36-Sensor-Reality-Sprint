# Sources and attribution

- **Dataset origin:** [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/), Geiger, Lenz, and Urtasun, CVPR 2012. KITTI publishes the data under [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/). The overlay is derived from a KITTI image and should be shared under compatible terms with attribution.
- **Demo sample distribution:** [OpenMMLab MMDetection3D, `demo/data/kitti/000008.png`, `000008.bin`, `000008.pkl`](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). These are synchronized frame `000008`. The repository's [license](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE) covers its software; the underlying KITTI data retain KITTI terms.
- **Calibration source:** `000008.pkl` from the same OpenMMLab demo. It stores `P0`–`P3`, `R0_rect`, `lidar_points.Tr_velo_to_cam`, and `CAM2.lidar2img`. `scripts/export_kitti_calib.py` exports the original unrectified transform and verifies the composition against `CAM2.lidar2img`.
- **Implementation:** Team G36 baseline projection. No external projection code was copied.

The four small frame `000008` files are bundled for a reproducible classroom benchmark. They retain the KITTI dataset's CC BY-NC-SA 3.0 terms; see [data/README.md](../data/README.md) for their source and regeneration details.

## Research: calibration drift và hướng hiệu chỉnh

Đối chiếu nguồn gốc ngày **2026-10-05**. Phần này phục vụ research/pitch của Lâm; repo hiện minh họa tác động của extrinsic perturbation, **chưa chạy hoặc tái lập** ba phương pháp dưới đây. Số đo của bài báo không phải kết quả của G36.

### Galibr

- **Bài:** *Galibr: Targetless LiDAR-Camera Extrinsic Calibration Method via Ground Plane Initialization* — Wonho Song, Minho Oh, Jaeyoung Lee, Hyun Myung; **IV 2024 Workshop**, theo thông tin tác giả trên [arXiv:2406.11599v1](https://arxiv.org/abs/2406.11599v1).
- **Ý tưởng:** GP-init dùng mặt phẳng đất để khởi tạo tư thế; sau đó ghép cạnh LiDAR/ảnh để tinh chỉnh ngoại tham. Thí nghiệm gồm KITTI và KAIST quadruped. Đọc [phương pháp và thí nghiệm](https://arxiv.org/html/2406.11599v1).
- **Liên hệ demo — suy luận của nhóm:** overlay ở mặt đường và biên vật thể giúp giải thích vì sao lệch góc làm mất khớp hình học; có thể dùng làm trực quan trước/sau nếu sau này tích hợp hiệu chỉnh.
- **Giới hạn áp dụng — suy luận:** cần kiểm tra chất lượng mặt phẳng đất và cạnh chung giữa hai cảm biến; demo một frame không chứng minh khả năng hội tụ của Galibr. Chưa xác minh được kho code chính thức từ nguồn đã đọc.

**Tên đúng là Galibr** trong bài được dẫn, không tự đổi thành Kalibr.

### CalibRefine

- **Bài:** *CalibRefine: Deep Learning-Based Online Automatic Targetless LiDAR-Camera Calibration with Iterative and Attention-Driven Post-Refinement* — Lei Cheng, Lihao Guo, Tianya Zhang, Tam Bang, Austin Harris, Mustafa Hajij, Mina Sartipi, Siyang Cao. Preprint **2025**, [arXiv:2502.17648v6](https://arxiv.org/abs/2502.17648v6).
- **Trạng thái xuất bản:** **IEEE Transactions on Instrumentation and Measurement, vol. 75, pp. 1–18, 2026**, theo [trang công bố của Radar Lab](https://radar.ece.arizona.edu/publications/calibrefine-lidar-camera-calibration-2025/); [DOI: 10.1109/TIM.2025.3647989](https://doi.org/10.1109/TIM.2025.3647989).
- **Code:** [radar-lab/Lidar_Camera_Automatic_Calibration](https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration), được liên kết trực tiếp trong preprint.
- **Ý tưởng:** ghép vật thể qua Common Feature Discriminator, ước lượng homography thô, tinh chỉnh lặp qua nhiều frame, rồi dùng attention. [Phương pháp và giới hạn, mục III và IV-C5](https://arxiv.org/html/2502.17648v6).
- **Giới hạn tác giả nêu:** thí nghiệm tại giao lộ phẳng với cảm biến cố định trên cột; cần vật thể trong vùng nhìn chung. Cảnh đường trống không thuộc phạm vi đánh giá.
- **Liên hệ demo — suy luận của nhóm:** phù hợp để giải thích association; homography 2D của bài không thay thế trực tiếp phép chiếu ngoại tham 6-DoF KITTI. Muốn đối chiếu cần cùng dữ liệu, định nghĩa correspondence và metric.

### DF-Calib và phiên bản UniCalib

- **Đúng phiên bản theo tên task:** *DF-Calib: Targetless LiDAR-Camera Calibration via Depth Flow* — Shu Han, Xubo Zhu, Ji Wu, Ximeng Cai, Wen Yang, Huai Yu, Gui-Song Xia; preprint **2025**, [arXiv:2504.01416v1](https://arxiv.org/abs/2504.01416v1).
- **Ý tưởng v1:** chuyển ảnh camera và LiDAR sang depth, dùng encoder chung để ước lượng depth flow; reliability map và loss có trọng số giảm ảnh hưởng vùng không đáng tin. [Bản đầy đủ DF-Calib v1](https://arxiv.org/html/2504.01416v1).
- **Cập nhật tên:** cùng mã arXiv đã đổi tiêu đề thành *UniCalib: Targetless LiDAR-Camera Calibration via Probabilistic Flow on Unified Depth Representations* ở [v2, 2025-08-09](https://arxiv.org/abs/2504.01416v2). Bản này bổ sung probabilistic flow; không gộp số liệu v1/v2.
- **Bản xuất bản:** **WACV 2026, pp. 1906–1915**, [CVF Open Access](https://openaccess.thecvf.com/content/WACV2026/html/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.html). [Code chính thức UniCalib](https://github.com/han-15/UniCalib) được bài WACV dẫn; repo yêu cầu depth preprocessing và CUDA extension, không phải một hàm plotting thay thế ngay được.
- **Liên hệ demo — suy luận của nhóm:** mũi tên dịch chuyển pixel trực quan hóa sai lệch correspondence. Mũi tên từ phép chiếu baseline/corrupted trong repo là dịch chuyển hình học đã biết, **không phải depth flow do DF-Calib dự đoán**. Việc hiệu chỉnh bằng mạng cần thí nghiệm riêng.

## Cách trình bày kết quả của sprint

Các quy ước dưới đây là hướng dẫn diễn giải cho thí nghiệm của nhóm, không phải số đo trích từ ba bài báo:

1. **Overlay:** giữ nguyên ảnh, point cloud, calibration baseline và point IDs; chỉ thay ngoại tham. Ghi rõ góc, đơn vị, trục/hệ tọa độ và phép ghép perturbation do nhóm thống nhất. Dùng cùng màu/legend giữa các mức drift.
2. **Error:** nếu tính khoảng cách giữa pixel baseline và corrupted của cùng điểm, gọi là *reprojection displacement relative to baseline*. Nó không phải sai số so với correspondence 2D được gán nhãn độc lập. Báo median/P90 và số điểm còn hợp lệ để tránh che mất điểm ra ngoài ảnh.
3. **Association:** công bố rõ cách chọn điểm/vật thể, bbox GT, mẫu số và cách xử lý điểm mất khỏi ảnh. Association retention là metric/proxy theo giao thức nhóm; không dùng nó thay cho detection accuracy.
4. **mAP drop:** chỉ báo khi có detector, confidence, GT, tập đánh giá và evaluator giống nhau cho baseline/corrupted. Overlay hoặc association drop đơn lẻ chưa đo được mAP drop.
5. **Research pitch:** ba hướng hiệu chỉnh để thảo luận là hình học mặt đất/cạnh (Galibr), correspondence vật thể (CalibRefine), và depth flow (DF-Calib/UniCalib). Phần triển khai sprint đo độ nhạy với drift; việc tái lập thuật toán hiệu chỉnh là bước sau.
