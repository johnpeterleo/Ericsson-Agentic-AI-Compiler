"""Tests for the benchmark programs, input generation, and baseline runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.agent.tools import AgentToolkit, StubProfiler
from src.benchmarks.baseline import check_port, load_baseline, run_baselines
from src.benchmarks.registry import discover_programs, get_program, load_program_file
from src.optimization.evaluator import evaluate

EXPECTED_DEFAULT = {"matmul", "softmax", "fused_gemm_gelu_layernorm", "attention"}
EXPECTED_OPTIONAL = {"transformer_block", "gpt"}


class RegistryTests(unittest.TestCase):
    def test_discovers_the_four_programs_and_the_optional_fifth(self):
        self.assertEqual(set(discover_programs()), EXPECTED_DEFAULT)
        self.assertEqual(
            set(discover_programs(include_optional=True)), EXPECTED_DEFAULT | EXPECTED_OPTIONAL
        )

    def test_programs_are_ordered_by_level(self):
        levels = [p.level for p in discover_programs(include_optional=True).values()]
        self.assertEqual(levels, sorted(levels))

    def test_unknown_program_and_size_raise(self):
        with self.assertRaises(KeyError):
            get_program("does_not_exist")
        with self.assertRaises(ValueError):
            get_program("softmax").make_inputs(size="huge")

    def test_program_file_missing_attributes_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.py"
            path.write_text("NAME = 'broken'\n", encoding="utf-8")
            with self.assertRaises(AttributeError):
                load_program_file(path)


class ProgramTests(unittest.TestCase):
    def setUp(self):
        self.programs = discover_programs(include_optional=True).values()

    def test_inputs_are_deterministic_float32_arrays(self):
        for program in self.programs:
            with self.subTest(program=program.name):
                first = program.make_inputs(seed=0, size="small")
                again = program.make_inputs(seed=0, size="small")
                other = program.make_inputs(seed=1, size="small")
                for a, b in zip(first, again):
                    self.assertEqual(a.dtype, np.float32)
                    np.testing.assert_array_equal(a, b)
                self.assertFalse(all(np.array_equal(a, b) for a, b in zip(first, other)))

    def test_jax_port_matches_numpy_reference(self):
        for program in self.programs:
            with self.subTest(program=program.name):
                result = check_port(program, seeds=(0, 1))
                self.assertTrue(result.passed, result.message)

    def test_reference_passes_the_evaluator_against_itself(self):
        for program in self.programs:
            with self.subTest(program=program.name):
                inputs = program.make_inputs(seed=0, size="small")
                result = evaluate(program.reference, program.reference, inputs,
                                  repeats=1, rtol=program.rtol, atol=program.atol)
                self.assertTrue(result.correct, result.message)

    def test_source_is_available_for_the_agent(self):
        for program in self.programs:
            self.assertIn("def reference", program.reference_source())


class BaselineTests(unittest.TestCase):
    def test_run_baselines_writes_summary_files(self):
        program = get_program("softmax")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_baselines([program], size="small", repeats=2, out_dir=Path(tmp),
                                    port_seeds=(0,), verbose=False)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failures"], [])
            self.assertTrue((run_dir / "summary.csv").exists())
            self.assertTrue((run_dir / "softmax" / "stablehlo.mlir").exists())
            record = load_baseline(run_dir)["softmax"]
            self.assertGreater(record["median_ms"], 0)
            self.assertTrue(record["port_check"]["passed"])
            self.assertEqual(record["input_shapes"], [[64, 1024]])


class ToolkitToleranceTests(unittest.TestCase):
    def test_toolkit_forwards_program_tolerances(self):
        toolkit = AgentToolkit(profiler=StubProfiler(), rtol=1e-2, atol=1e-3)
        with patch("src.agent.tools.evaluate") as fake_evaluate:
            toolkit.measure_against_reference(abs, abs, (np.ones(2),), repeats=1)
        self.assertEqual(fake_evaluate.call_args.kwargs["rtol"], 1e-2)
        self.assertEqual(fake_evaluate.call_args.kwargs["atol"], 1e-3)


if __name__ == "__main__":
    unittest.main()
