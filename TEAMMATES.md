# Team G36

| Thành viên | MSSV | Công việc đã đóng góp |
| --- | --- | --- |
| Hà Mạnh Tuân | 2A202602982 | Trưởng nhóm; xây dựng và kiểm chứng baseline Camera–LiDAR cho frame `000008`, đưa mẫu demo vào repo; tích hợp benchmark yaw LiDAR cuối cùng, giao diện Streamlit CalibGuard và tài liệu chạy nhóm. |
| Trần Cao Quốc Định | 2A202602939 | Xử lý nhãn và ground truth của mẫu KITTI; kiểm tra phép chiếu với calibration gốc, xuất bảng object, ảnh bbox, dữ liệu point–box association và kiểm thử validation. |
| Lê Trọng Khánh | 2A202602941 | Xây dựng hàm tạo sai lệch extrinsic bằng yaw trong hệ LiDAR, quy ước trục/góc và thứ tự nhân ma trận; viết kiểm thử cho phép quay và tích hợp với phép chiếu KITTI. |
| Đào Quang Cảnh | 2A202602542 | Xây dựng engine benchmark calibration drift, các metric reprojection/association, bộ đọc dữ liệu và script chạy benchmark; xuất kết quả CSV theo mức yaw và từng object. |
| Đinh Quang Lâm | 2A202602875 | Làm visualization baseline/corrupted, biểu đồ và demo HTML offline; nghiên cứu, tổng hợp nguồn tham khảo và viết handoff visualization. Demo riêng này dùng camera-frame yaw. |

Phân công trên dựa vào các phần đã được tích hợp trong lịch sử commit. Xem [README](README.md) để chạy benchmark cuối cùng, [validation handoff](docs/DINH_VALIDATION.md), [visualization handoff](docs/VISUALIZATION_HANDOFF.md) và [nguồn tham khảo](references/SOURCES.md). Mẫu KITTI do OpenMMLab MMDetection3D phân phối, không phải dữ liệu do nhóm tạo ra.
