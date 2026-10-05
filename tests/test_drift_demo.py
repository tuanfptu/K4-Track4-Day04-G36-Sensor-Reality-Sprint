"""Checks for point identity, camera convention, and explicitly limited proxies."""

import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.calibration import load_calibration
from src.drift_demo import camera_yaw_transform, load_sample_boxes, measure_drift, project_aligned
from src.kitti_loader import load_image, load_lidar
from src.projection import project


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = np.column_stack((np.eye(3), np.zeros(3)))
P = np.array([[10.0, 0, 5, 0], [0, 10.0, 5, 0], [0, 0, 1, 0]])


class DriftDemoTests(unittest.TestCase):
    def test_identity_and_row_alignment(self):
        points = np.array([[0, 0, 10], [0, 0, -2], [100, 0, 10], [1, 0, 10]])
        pixels, depth, inside = project_aligned(points, P, np.eye(3), IDENTITY, 10, 10)
        np.testing.assert_allclose(pixels[[0, 2, 3]], [[5, 5], [105, 5], [6, 5]])
        self.assertTrue(np.isnan(pixels[1]).all())
        np.testing.assert_equal(depth, [10, -2, 10, 10])
        np.testing.assert_equal(inside, [True, False, False, True])
        stats = measure_drift(pixels, pixels, inside, inside, [[4, 4, 7, 7]])
        self.assertEqual(stats["median_error_px"], 0)
        self.assertEqual(stats["p90_error_px"], 0)
        self.assertEqual(stats["association_retention"], 1)
        self.assertEqual(stats["association_drop_pp"], 0)
        self.assertEqual(stats["baseline_associations"], 2)

    def test_offscreen_dropout_counts_as_association_loss(self):
        before = np.array([[5, 5], [8, 5], [20, 5]], dtype=float)
        after = np.array([[6, 5], [11, 5], [6, 5]], dtype=float)
        stats = measure_drift(before, after, [1, 1, 0], [1, 0, 1], [[0, 0, 9, 9]])
        self.assertEqual(stats["median_error_px"], 1)
        self.assertEqual(stats["matched_visible"], 1)
        self.assertEqual(stats["association_retention"], 0.5)
        self.assertEqual(stats["association_drop_pp"], 50)

    def test_overlapping_boxes_retain_same_pairs(self):
        stats = measure_drift([[5, 5]], [[7, 5]], [1], [1], [[0, 0, 6, 6], [4, 4, 8, 8]])
        self.assertEqual(stats["baseline_associations"], 2)
        self.assertEqual(stats["retained_associations"], 1)
        self.assertEqual(stats["association_retention"], 0.5)

    def test_camera_yaw_rotates_translation_and_positive_z_toward_x(self):
        transform = IDENTITY.copy()
        transform[:, 3] = [1, 2, 3]
        rotated = camera_yaw_transform(transform, 90)
        np.testing.assert_allclose(rotated[:, 2], [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(rotated[:, 3], [3, 2, -1], atol=1e-14)
        positive, _, _ = project_aligned([[0, 0, 10]], P, np.eye(3), camera_yaw_transform(IDENTITY, 2), 10, 10)
        self.assertGreater(positive[0, 0], 5)
        np.testing.assert_array_equal(camera_yaw_transform(transform, 0), transform)
        with self.assertRaises(ValueError):
            camera_yaw_transform(transform, np.nan)

    def test_no_baseline_associations_and_no_matched_points(self):
        stats = measure_drift([[2, 2]], [[8, 8]], [1], [0], [[6, 6, 9, 9]])
        self.assertIsNone(stats["association_retention"])
        self.assertIsNone(stats["association_drop_pp"])
        self.assertIsNone(stats["median_error_px"])
        self.assertIsNone(stats["p90_error_px"])
        empty = measure_drift(np.empty((0, 2)), np.empty((0, 2)), [], [], [])
        self.assertEqual(empty["baseline_visible"], 0)

    def test_bundled_baseline_parity(self):
        sample = ROOT / "data/sample"
        points = load_lidar(sample / "points.bin")
        p, rect, transform = load_calibration(sample / "calib.txt")
        image = load_image(sample / "image.png")
        pixels, depth, inside = project_aligned(points, p, rect, transform, *image.size)
        legacy_pixels, legacy_depth, _ = project(points, p, rect, transform, *image.size)
        np.testing.assert_allclose(pixels[inside], legacy_pixels, rtol=0, atol=1e-12)
        np.testing.assert_allclose(depth[inside], legacy_depth, rtol=0, atol=1e-12)
        self.assertEqual(len(pixels), len(points))

    def test_restricted_box_extraction(self):
        boxes = load_sample_boxes(ROOT / "data/sample/000008.pkl")
        self.assertEqual(boxes.shape, (6, 4))
        np.testing.assert_allclose(boxes[0], [0, 192.37, 402.31, 374])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.pkl"
            # A GLOBAL opcode must be rejected before it can instantiate a class.
            path.write_bytes(pickle.dumps(Path("unused")))
            with self.assertRaises(pickle.UnpicklingError):
                load_sample_boxes(path)
            sample = {"sample_id": 8, "images": {"CAM2": {"img_path": "000009.png"}},
                      "lidar_points": {"lidar_path": "000008.bin"}, "instances": []}
            path.write_bytes(pickle.dumps({"data_list": [sample]}))
            with self.assertRaisesRegex(ValueError, "both match"):
                load_sample_boxes(path)
            sample["images"]["CAM2"]["img_path"] = "000008.png"
            sample["sample_id"] = 9
            path.write_bytes(pickle.dumps({"data_list": [sample]}))
            with self.assertRaisesRegex(ValueError, "sample_id"):
                load_sample_boxes(path)


if __name__ == "__main__":
    unittest.main()
