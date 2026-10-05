"""Verify drift geometry and compatibility with the existing KITTI pipeline.

Run from the repository root: python -m unittest discover -s tests -v
"""

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.calibration import load_calibration
from src.kitti_loader import load_image, load_lidar
from src.perturbation import perturb_extrinsic
from src.projection import project


ROOT = Path(__file__).resolve().parents[1]


class PerturbExtrinsicTests(unittest.TestCase):
    def setUp(self):
        self.identity = np.eye(4)[:3, :]
        # Rx(90 degrees) plus translation: deliberately not an identity pose.
        self.extrinsic = np.array([
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, -1.0, 3.0],
            [0.0, 1.0, 0.0, 4.0],
        ])

    def test_zero_yaw_preserves_extrinsic(self):
        assert_array_equal(perturb_extrinsic(self.extrinsic), self.extrinsic)

    def test_positive_yaw_rotates_lidar_x_toward_y(self):
        result = perturb_extrinsic(self.identity, 90.0)
        assert_allclose(result @ [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0], atol=1e-12)
        assert_allclose(result @ [0.0, 1.0, 0.0, 1.0], [-1.0, 0.0, 0.0], atol=1e-12)
        assert_array_equal(result @ [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0])

    def test_negative_yaw_rotates_in_opposite_direction(self):
        result = perturb_extrinsic(self.identity, -90.0)
        assert_allclose(result @ [1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0], atol=1e-12)

    def test_small_angle_is_in_degrees(self):
        result = perturb_extrinsic(self.identity, 2.0)
        # Independently tabulated cos(2 degrees) and sin(2 degrees).
        assert_allclose(result @ [1.0, 0.0, 0.0, 1.0],
                        [0.9993908270190958, 0.03489949670250097, 0.0], atol=1e-12)

    def test_composition_uses_lidar_frame_and_lidar_origin(self):
        result = perturb_extrinsic(self.extrinsic, 90.0)
        # (1,2,3) -> yaw: (-2,1,3) -> Rx: (-2,-3,1) -> +t: (0,0,5).
        # Left multiplication or rotation about camera Z gives a different result.
        assert_allclose(result @ [1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 5.0], atol=1e-12)
        assert_array_equal(result @ [0.0, 0.0, 0.0, 1.0], [2.0, 3.0, 4.0])

    def test_benchmark_angles_preserve_rigid_rotation_and_translation(self):
        for angle in (0.0, 0.5, 1.0, 1.5, 2.0):
            with self.subTest(yaw_deg=angle):
                result = perturb_extrinsic(self.extrinsic, angle)
                self.assertEqual(result.shape, (3, 4))
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertTrue(np.isfinite(result).all())
                rotation = result[:, :3]
                assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
                assert_array_equal(result[:, 3], self.extrinsic[:, 3])

    def test_opposite_yaw_restores_original(self):
        result = perturb_extrinsic(perturb_extrinsic(self.extrinsic, 2.0), -2.0)
        assert_allclose(result, self.extrinsic, atol=1e-12)

    def test_input_is_not_modified_or_shared(self):
        for angle in (0.0, 2.0):
            with self.subTest(yaw_deg=angle):
                original = self.extrinsic.copy()
                result = perturb_extrinsic(original, angle)
                assert_array_equal(original, self.extrinsic)
                self.assertFalse(np.shares_memory(original, result))
                result[0, 3] = 100.0
                assert_array_equal(original, self.extrinsic)

    def test_read_only_float32_input_is_supported(self):
        original = self.extrinsic.astype(np.float32)
        original.flags.writeable = False
        result = perturb_extrinsic(original, 2.0)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        assert_array_equal(result[:, 3], original[:, 3])

    def test_wrong_transform_shapes_are_rejected(self):
        for shape in ((4, 4), (3, 3), (4, 3), (12,), (0,)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "3x4"):
                    perturb_extrinsic(np.zeros(shape), 1.0)

    def test_nonfinite_transform_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                original = self.extrinsic.copy()
                original[0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite real 3x4"):
                    perturb_extrinsic(original, 1.0)

    def test_nonnumeric_and_complex_transform_are_rejected(self):
        for original in ([['invalid'] * 4] * 3, self.extrinsic.astype(complex) + 1j):
            with self.subTest(transform=repr(original)):
                with self.assertRaisesRegex(ValueError, "finite real 3x4"):
                    perturb_extrinsic(original, 1.0)

    def test_invalid_yaw_is_rejected(self):
        for angle in (np.nan, np.inf, -np.inf, None, 'invalid', [1.0], 1j):
            with self.subTest(yaw_deg=angle):
                with self.assertRaisesRegex(ValueError, "finite real scalar"):
                    perturb_extrinsic(self.extrinsic, angle)


class KittiIntegrationTests(unittest.TestCase):
    def test_zero_drift_reproduces_bundled_baseline_projection(self):
        image = load_image(ROOT / 'data/sample/image.png')
        points = load_lidar(ROOT / 'data/sample/points.bin')
        p, rect, transform = load_calibration(ROOT / 'data/sample/calib.txt')
        baseline = project(points, p, rect, transform, *image.size)
        actual = project(points, p, rect, perturb_extrinsic(transform, 0.0), *image.size)
        self.assertGreater(len(baseline[0]), 0)
        assert_array_equal(actual[0], baseline[0])
        assert_array_equal(actual[1], baseline[1])
        self.assertEqual(actual[2], baseline[2])

    def test_yaw_changes_pixel_projection_with_fixed_intrinsics(self):
        p, rect, transform = load_calibration(ROOT / 'data/sample/calib.txt')
        original_p, original_rect, original_transform = p.copy(), rect.copy(), transform.copy()
        # A forward point stays visible at both angles: no correspondence loss.
        point = np.array([[10.0, 0.0, 0.0, 1.0]])
        baseline = project(point, p, rect, transform, 1242, 375)
        drifted = project(point, p, rect, perturb_extrinsic(transform, 2.0), 1242, 375)
        self.assertEqual(len(baseline[0]), 1)
        self.assertEqual(len(drifted[0]), 1)
        self.assertGreater(np.linalg.norm(drifted[0] - baseline[0]), 1.0)
        assert_array_equal(p, original_p)
        assert_array_equal(rect, original_rect)
        assert_array_equal(transform, original_transform)


if __name__ == '__main__':
    unittest.main()
