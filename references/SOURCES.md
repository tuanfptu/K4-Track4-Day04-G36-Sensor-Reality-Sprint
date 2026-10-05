# Sources and attribution

- **Dataset origin:** [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/), Geiger, Lenz, and Urtasun, CVPR 2012. KITTI publishes the data under [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/). The overlay is derived from a KITTI image and should be shared under compatible terms with attribution.
- **Demo sample distribution:** [OpenMMLab MMDetection3D, `demo/data/kitti/000008.png`, `000008.bin`, `000008.pkl`](https://github.com/open-mmlab/mmdetection3d/tree/main/demo/data/kitti). These are synchronized frame `000008`. The repository's [license](https://github.com/open-mmlab/mmdetection3d/blob/main/LICENSE) covers its software; the underlying KITTI data retain KITTI terms.
- **Calibration source:** `000008.pkl` from the same OpenMMLab demo. It stores `P0`–`P3`, `R0_rect`, `lidar_points.Tr_velo_to_cam`, and `CAM2.lidar2img`. `scripts/export_kitti_calib.py` exports the original unrectified transform and verifies the composition against `CAM2.lidar2img`.
- **Implementation:** Team G36 baseline projection. No external projection code was copied.

The four small frame `000008` files are bundled for a reproducible classroom benchmark. They retain the KITTI dataset's CC BY-NC-SA 3.0 terms; see [data/README.md](../data/README.md) for their source and regeneration details.

For bbox/label extraction and point count validation, the team checked the
official MMDetection3D [KITTI info converter](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py),
[point count code](https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/kitti_converter.py),
[box operations](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/ops/box_np_ops.py),
and [camera box definition](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/cam_box3d.py).
