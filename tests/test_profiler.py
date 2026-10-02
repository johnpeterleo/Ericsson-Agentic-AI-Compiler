"""Tests for structured JAX profiling."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.profiling.profiler import profile_function


class ProfileFunctionTests(unittest.TestCase):
    def test_records_latency_statistics_and_device_metadata(self):
        result = profile_function(
            lambda x: x + 1,
            (np.array([1.0, 2.0], dtype=np.float32),),
            repeats=3,
        )

        self.assertEqual(len(result.samples_ms), 3)
        self.assertGreaterEqual(result.min_ms, 0)
        self.assertLessEqual(result.min_ms, result.median_ms)
        self.assertLessEqual(result.median_ms, result.max_ms)
        self.assertGreaterEqual(result.stddev_ms, 0)
        self.assertGreaterEqual(result.compile_ms, 0)
        self.assertTrue(result.backend)
        self.assertTrue(result.device_platform)
        self.assertTrue(result.device_kind)
        self.assertIsNone(result.stablehlo_text)

    def test_can_capture_and_save_stablehlo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lowered" / "program.stablehlo"
            result = profile_function(
                lambda x: x * 2,
                (np.ones((2,), dtype=np.float32),),
                repeats=1,
                stablehlo_path=path,
            )

            self.assertIsNotNone(result.stablehlo_text)
            self.assertEqual(result.stablehlo_path, path)
            self.assertEqual(path.read_text(encoding="utf-8"), result.stablehlo_text)
            self.assertIn("module", result.stablehlo_text)

    def test_rejects_invalid_repeat_count(self):
        with self.assertRaises(ValueError):
            profile_function(lambda x: x, (np.array([1.0]),), repeats=0)


if __name__ == "__main__":
    unittest.main()
