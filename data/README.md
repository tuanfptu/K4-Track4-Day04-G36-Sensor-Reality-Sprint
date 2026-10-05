# KITTI demo frame 000008

Dataset origin: KITTI. Demo distribution: [OpenMMLab MMDetection3D `demo/data/kitti`](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). All three source files share frame ID `000008`; the pickle also identifies its CAM2 image as `000008.png` and LiDAR as `000008.bin`.

From the repository root in PowerShell, download only the three small files:

```powershell
New-Item -ItemType Directory -Force data/sample | Out-Null
Invoke-WebRequest https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/000008.png -OutFile data/sample/image.png
Invoke-WebRequest https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/000008.bin -OutFile data/sample/points.bin
Invoke-WebRequest https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/demo/data/kitti/000008.pkl -OutFile data/sample/000008.pkl
python scripts/export_kitti_calib.py
```

This produces `data/sample/calib.txt`. The projector uses only `image.png`, `points.bin`, and `calib.txt`. The source pickle contains `P0`–`P3`, `R0_rect`, and the original `lidar_points.Tr_velo_to_cam`. Its `CAM2.lidar2cam` already includes rectification, so the exporter deliberately uses the original transform to match the baseline's `P2 @ R0_rect @ Tr_velo_to_cam` sequence. It checks this composition against the pickle's `CAM2.lidar2img` matrix.

The downloaded image, point cloud, pickle, and exported calibration remain ignored by Git. The KITTI dataset is licensed [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/); credit KITTI and follow those terms before redistributing data or derived images.
