"""Tests for the agent's bridge to structured JAX profiling."""

import unittest
from unittest.mock import patch

from src.agent.tools import JaxProfiler
from src.profiling.profiler import ProfileResult


class JaxProfilerTests(unittest.TestCase):
    def test_converts_structured_result_to_agent_profile_report(self):
        profile_result = ProfileResult(
            samples_ms=(1.0, 2.0, 3.0),
            median_ms=2.0,
            mean_ms=2.0,
            min_ms=1.0,
            max_ms=3.0,
            stddev_ms=0.8,
            compile_ms=12.0,
            backend="gpu",
            device_platform="gpu",
            device_kind="NVIDIA L4",
            device_id=0,
        )
        with patch("src.agent.tools.profile_function", return_value=profile_result) as profile:
            report = JaxProfiler(repeats=7).profile(lambda x: x, ("input",))

        profile.assert_called_once()
        self.assertEqual(profile.call_args.kwargs["repeats"], 7)
        self.assertEqual(report.backend, "gpu")
        self.assertEqual(report.total_ms, 2.0)
        self.assertEqual(report.raw["device_kind"], "NVIDIA L4")
        self.assertEqual(report.raw["compile_ms"], 12.0)
        self.assertEqual(report.raw["samples_ms"], (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
