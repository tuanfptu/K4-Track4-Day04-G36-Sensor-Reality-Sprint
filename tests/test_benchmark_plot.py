import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmark_plot import REQUIRED_FIELDS, plot_benchmark, read_benchmark_csv


def sample(**changes):
    row = dict(condition="camera_yaw", drift_value=0, drift_unit="deg",
               median_error_px=0, p90_error_px=0, association_retention=1)
    row.update(changes)
    return row


class BenchmarkCsvTests(unittest.TestCase):
    def read_rows(self, rows, fields=None):
        fields = fields or list(dict.fromkeys(key for row in rows for key in row))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            return read_benchmark_csv(path)

    def test_signed_drift_undefined_values_and_extra_columns(self):
        result = self.read_rows([sample(drift_value=-2, median_error_px="", p90_error_px="",
                                       association_retention="", point_count="0"),
                                 sample(drift_value=2, association_retention=0.75, point_count="42")])
        self.assertEqual(result[0]["drift_value"], -2)
        for field in ("median_error_px", "p90_error_px", "association_retention", "association_drop_pp", "map"):
            self.assertIsNone(result[0][field])
        self.assertEqual(result[1]["association_drop_pp"], 25)
        self.assertEqual(result[1]["point_count"], "42")

    def test_empty_or_missing_schema_is_rejected(self):
        for content in ("", "condition,drift_value\na,0\n", ",".join(REQUIRED_FIELDS) + "\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_benchmark_csv(path)

    def test_invalid_measurements_rejected(self):
        changes = [dict(drift_value="NaN"), dict(median_error_px="inf"), dict(p90_error_px=-1),
                   dict(association_retention=1.01), dict(association_retention=-0.1),
                   dict(association_drop_pp=101), dict(association_drop_pp=10),
                   dict(condition=" "), dict(drift_unit=""), dict(map=1.1, map_protocol="test")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read_rows([sample(**change)])

    def test_duplicate_samples_rejected_but_conditions_and_units_are_separate(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read_rows([sample(), sample(drift_value="-0")])
        self.assertEqual(len(self.read_rows([sample(), sample(condition="camera_roll"), sample(drift_unit="rad")])), 3)

    def test_map_needs_protocol_and_series_baseline(self):
        cases = [[sample(map=0.8)], [sample(drift_value=1, map=0.7, map_protocol="AP40")],
                 [sample(map=0.8, map_protocol="AP40"), sample(drift_value=1, map=0.7, map_protocol="AP11")],
                 [sample(map=0.8, map_protocol="AP40"), sample(condition="camera_roll", drift_value=1, map=0.7, map_protocol="AP40")]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read_rows(rows)

    def test_map_gaps_are_preserved(self):
        rows = self.read_rows([sample(map=0.8, map_protocol="AP40 Car moderate"),
                               sample(drift_value=1, map="", map_protocol=""),
                               sample(drift_value=2, map=0.6, map_protocol="AP40 Car moderate")])
        self.assertIsNone(rows[1]["map"])
        self.assertEqual(rows[2]["map"], 0.6)


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is optional until plotting")
class BenchmarkPlotTests(unittest.TestCase):
    def test_mixed_units_generate_separate_safe_figures(self):
        rows = [sample(), sample(drift_value=2), sample(condition="camera_x", drift_unit="../m", drift_value=1),
                sample(condition="camera_y", drift_unit="m", drift_value=1)]
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_benchmark(rows, temporary)
            self.assertEqual({path.name for path in paths}, {"metrics_deg.png", "metrics_m.png", "metrics_m_2.png"})
            for path in paths:
                self.assertEqual(path.parent, Path(temporary))
                self.assertGreater(path.stat().st_size, 10000)

    def test_plot_preserves_gaps_and_separate_conditions(self):
        from matplotlib.axes import Axes
        calls = []
        original = Axes.plot

        def record(axis, x, y, *args, **kwargs):
            calls.append((list(x), list(y), kwargs.get("label")))
            return original(axis, x, y, *args, **kwargs)

        rows = [sample(drift_value=2, median_error_px=20),
                sample(drift_value=1, median_error_px=None), sample(),
                sample(condition="camera_roll", drift_value=-1, median_error_px=3)]
        with tempfile.TemporaryDirectory() as temporary, patch.object(Axes, "plot", record):
            paths = plot_benchmark(rows, temporary)
            self.assertEqual([path.name for path in paths], ["metrics.png"])
        yaw = next(call for call in calls if call[2] == "camera_yaw: median")
        self.assertEqual(yaw[0], [0, 1, 2])
        self.assertTrue(math.isnan(yaw[1][1]))
        roll = next(call for call in calls if call[2] == "camera_roll: median")
        self.assertEqual(roll[0], [-1])

    def test_map_is_optional_and_uses_true_measured_drop(self):
        from matplotlib.axes import Axes
        calls = []
        original = Axes.plot

        def record(axis, x, y, *args, **kwargs):
            calls.append((axis.get_ylabel(), list(y)))
            return original(axis, x, y, *args, **kwargs)

        rows = [sample(map=0.8, map_protocol="AP40"), sample(drift_value=1),
                sample(drift_value=2, map=0.6, map_protocol="AP40")]
        with tempfile.TemporaryDirectory() as temporary, patch.object(Axes, "plot", record):
            paths = plot_benchmark(rows, temporary)
            self.assertEqual([path.name for path in paths], ["metrics.png", "map.png"])
        drops = next(y for label, y in calls if label == "mAP drop (percentage points)")
        self.assertAlmostEqual(drops[0], 0)
        self.assertTrue(math.isnan(drops[1]))
        self.assertAlmostEqual(drops[2], 20)

    def test_all_undefined_metrics_still_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_benchmark([sample(median_error_px=None, p90_error_px=None, association_retention=None)], temporary)
            self.assertTrue(paths[0].is_file())

    def test_displacement_axis_contains_large_measurements(self):
        from matplotlib.figure import Figure
        upper_limits = []
        original = Figure.savefig

        def record(figure, *args, **kwargs):
            upper_limits.append(figure.axes[0].get_ylim()[1])
            return original(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, patch.object(Figure, "savefig", record):
            plot_benchmark([sample(), sample(drift_value=2, median_error_px=40, p90_error_px=80)], temporary)
        self.assertGreaterEqual(upper_limits[0], 80)


if __name__ == "__main__":
    unittest.main()
