# Sources and attribution

- **Dataset origin:** [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/), Geiger, Lenz, and Urtasun, CVPR 2012. KITTI publishes the data under [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/). The overlay is derived from a KITTI image and should be shared under compatible terms with attribution.
- **Demo sample distribution:** [OpenMMLab MMDetection3D, `demo/data/kitti/000008.png`, `000008.bin`, `000008.pkl`](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). These are synchronized frame `000008`. The repository's [license](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE) covers its software; the underlying KITTI data retain KITTI terms.
- **Calibration source:** `000008.pkl` from the same OpenMMLab demo. It stores `P0`–`P3`, `R0_rect`, `lidar_points.Tr_velo_to_cam`, and `CAM2.lidar2img`. `scripts/export_kitti_calib.py` exports the original unrectified transform and verifies the composition against `CAM2.lidar2img`.
- **Implementation:** Team G36 baseline projection. No external projection code was copied.

The raw sample files and exported calibration are kept out of Git; [data/README.md](../data/README.md) gives the three-file retrieval commands.
