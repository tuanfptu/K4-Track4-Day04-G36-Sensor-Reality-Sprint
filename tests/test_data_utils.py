"""Geometry regressions and integration checks; standard-library test runner."""

import io
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_sample import run
from src.data_utils import (SampleUnpickler, association_diagnostics, bbox_iou,
                            camera_box_corners, extract_objects, load_sample_info,
                            points_in_bboxes, points_in_camera_boxes, prepare_association,
                            project_camera_box, project_with_indices)

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_projection_preserves_indices_and_filters_depth_and_bounds(self):
        points = np.array([[1, 1, 1], [2, 0, 1], [0, 0, -1], [0, 0, 0],
                           [0, 0, 1], [np.nan, 0, 1], [0, 2, 1]], dtype=float)
        p = np.eye(3, 4)
        result = project_with_indices(points, p, np.eye(3), p, 2, 2)
        np.testing.assert_array_equal(result["point_indices"][result["in_image"]], [0, 4])
        np.testing.assert_allclose(result["pixels"][[0, 4]], [[1, 1], [0, 0]])
        self.assertTrue(np.isnan(result["pixels"][[2, 3, 5]]).all())

    def test_projection_uses_p2_translation_and_denominator(self):
        p = np.array([[2, 0, 0, 2], [0, 3, 0, 1], [0, 0, 1, 1]], dtype=float)
        result = project_with_indices([[1, 1, 1]], p, np.eye(3), np.eye(3, 4), 10, 10)
        np.testing.assert_allclose(result["pixels"], [[2, 2]])
        self.assertEqual(result["depth"][0], 1)  # rectified Z, not q[2]

    def test_bottom_center_dimensions_and_yaw(self):
        box = [0, 2, 10, 4, 2, 2, 0]
        corners = camera_box_corners(box)
        np.testing.assert_allclose(corners.min(0), [-2, 0, 9])
        np.testing.assert_allclose(corners.max(0), [2, 2, 11])
        box[-1] = np.pi / 2
        # +local x rotates toward -camera z, distinguishing the yaw sign.
        np.testing.assert_allclose(camera_box_corners(box)[4], [-1, 0, 8])
        points = [[0, 1, 8.5], [1.5, 1, 10], [0, -0.1, 10], [0, 2.1, 10]]
        np.testing.assert_array_equal(points_in_camera_boxes(points, [box])[:, 0], [True, False, False, False])

    def test_pixel_membership_half_open_and_empty(self):
        pixels = [[0, 0], [1.999, 1.999], [2, 1], [1, 2], [np.nan, 0]]
        np.testing.assert_array_equal(points_in_bboxes(pixels, [[0, 0, 2, 2]])[:, 0],
                                      [True, True, False, False, False])
        self.assertEqual(points_in_bboxes(pixels, []).shape, (5, 0))

    def test_iou_known_values_and_degenerate_boxes(self):
        result = bbox_iou([[0, 0, 2, 2], [1, 1, 1, 1]], [[1, 1, 3, 3], [0, 0, 2, 2]])
        np.testing.assert_allclose(result, [[1 / 7, 1], [0, 0]])
        self.assertEqual(bbox_iou([], []).shape, (0, 0))

    def test_near_plane_and_invalid_3d_boxes(self):
        self.assertIsNone(project_camera_box([0, 0, 0, 2, 2, 2, 0], np.eye(3, 4), 10, 10))
        with self.assertRaises(ValueError):
            camera_box_corners([-1000, -1000, -1000, -1, -1, -1, -10])

    def test_ignore_overlap_ambiguity_and_background_policy(self):
        objs = [dict(object_id=i, valid_3d=True, ignore=False, bbox_label=2,
                     bbox=[0, 0, 2, 2], bbox_3d=[0, 1, 5, 2, 2, 2, 0]) for i in range(2)]
        objs.append(dict(object_id=2, valid_3d=False, ignore=True, bbox=[3, 3, 4, 4]))
        projection = dict(camera_xyz=np.array([[0, 0, 5], [9, 0, 5], [9, 0, 5], [9, 0, 5]]),
                          pixels=np.array([[1, 1], [1, 1], [3.5, 3.5], [4, 4]]),
                          in_image=np.array([True, True, True, False]))
        data = prepare_association(np.zeros((4, 4)), projection, objs)
        np.testing.assert_array_equal(data["metric_eligible"], [False, True, False, False])
        np.testing.assert_array_equal(data["reference_object_id"], [-2, -1, -1, -1])
        for row in association_diagnostics(data):
            self.assertEqual((row["tp"], row["fp"], row["fn"]), (0, 1, 0))
            self.assertIsNone(row["recall"])
        empty = prepare_association(np.zeros((4, 4)), projection, [])
        self.assertEqual(empty["membership_3d"].shape, (4, 0))
        self.assertEqual(association_diagnostics(empty), [])

    def test_pickle_rejects_unexpected_global(self):
        with self.assertRaises(pickle.UnpicklingError):
            SampleUnpickler(io.BytesIO(b"cbuiltins\neval\n.")).load()

    def test_source_ids_and_ignore_preservation(self):
        sample = load_sample_info(ROOT / "data/sample/000008.pkl")
        objs = extract_objects(sample)
        self.assertEqual([o["class_name"] for o in objs], ["Car"] * 6 + ["Ignore"] * 4)
        self.assertEqual([o["object_id"] for o in objs], list(range(10)))
        self.assertEqual(objs[6]["bbox_3d"], [-1000, -1000, -1000, -1, -1, -1, -10])
        with self.assertRaises(ValueError):
            load_sample_info(ROOT / "data/sample/000008.pkl", sample_id=9)


class SampleIntegrationTests(unittest.TestCase):
    def test_full_sample_exports_and_source_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = run(ROOT / "data/sample", output)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["total_points"], 17238)
            self.assertLess(report["source_pixel_max_error"], 0.001)
            self.assertEqual([o["recomputed_kitti_lidar_points"] for o in report["objects"]],
                             [1325, 1900, 881, 659, 55, 162])
            with np.load(output / "association_000008.npz", allow_pickle=False) as data:
                self.assertEqual(data["membership_3d"].shape, (17238, 6))
                self.assertEqual(data["metric_eligible"].sum(), 17204)
                np.testing.assert_array_equal(data["point_indices"], np.arange(17238))
                for key in data.files:
                    self.assertFalse(data[key].dtype.hasobject)
            self.assertGreater((output / "baseline_bbox.png").stat().st_size, 0)

    def test_wrong_rectification_is_detected(self):
        # Real-data regression: the original baseline still projects points with
        # this wrong transform; source-matrix verification must reject it.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("000008.pkl", "image.png", "points.bin"):
                shutil.copyfile(ROOT / "data/sample" / name, folder / name)
            lines = (ROOT / "data/sample/calib.txt").read_text().splitlines()
            lines = ["R0_rect: 1 0 0 0 1 0 0 0 1" if line.startswith("R0_rect:") else line for line in lines]
            (folder / "calib.txt").write_text("\n".join(lines))
            report = run(folder, folder / "results")
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["projection_matrix_matches_source"])


if __name__ == "__main__":
    unittest.main()
