"""Matched-point and fixed-bbox regression checks for the final yaw benchmark."""

import unittest

import numpy as np

from src.metrics import measure_yaw_drift


class FinalMetricsTests(unittest.TestCase):
    def test_visibility_filter_keeps_original_point_identity(self):
        baseline = {"pixels": np.array([[1., 1.], [2., 2.], [np.nan, np.nan]]),
                    "in_image": np.array([True, True, False])}
        drifted = {"pixels": np.array([[np.nan, np.nan], [3., 2.], [1., 1.]]),
                   "in_image": np.array([False, True, True])}
        objects = [{"object_id": 7, "class_name": "Car", "bbox": [0, 0, 2.5, 2.5]}]
        summary, rows = measure_yaw_drift(baseline, drifted, objects)
        self.assertEqual(summary["matched_visible_points"], 1)
        self.assertEqual(summary["median_reprojection_px"], 1.0)
        self.assertEqual(summary["out_of_frame_pct"], 50.0)
        self.assertEqual(summary["baseline_associated_pairs"], 2)
        self.assertEqual(summary["association_retention_pct"], 0.0)
        self.assertEqual(rows[0]["baseline_associated_points"], 2)
        self.assertEqual(rows[0]["retained_points"], 0)

    def test_zero_drift_is_exact_identity(self):
        projection = {"pixels": np.array([[1., 1.], [2., 2.]]),
                      "in_image": np.array([True, True])}
        objects = [{"object_id": 0, "class_name": "Car", "bbox": [0, 0, 3, 3]}]
        summary, _ = measure_yaw_drift(projection, projection, objects)
        self.assertEqual(summary["p90_reprojection_px"], 0.0)
        self.assertEqual(summary["association_retention_pct"], 100.0)
        self.assertEqual(summary["out_of_frame_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
