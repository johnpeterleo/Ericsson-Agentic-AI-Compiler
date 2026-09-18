"""Correctness, timing, and failure reporting for JAX candidate evaluation."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import jax
import numpy as np

from src.optimization import demo_evaluator
from src.optimization.evaluator import EvaluationResult, evaluate, measure_latency, outputs_match


class CorrectnessTests(unittest.TestCase):
    def test_identical_values_match(self):
        values = np.array([0.0, -2.0, 5.0], dtype=np.float32)
        self.assertIs(outputs_match(values, values.copy()), True)

    def test_explicit_tolerance(self):
        reference = np.array([0.0, 1.0], dtype=np.float32)
        candidate = reference + np.float32(1e-5)
        self.assertTrue(outputs_match(reference, candidate, rtol=0, atol=2e-5))
        self.assertFalse(outputs_match(reference, candidate, rtol=0, atol=1e-7))

    def test_incorrect_values_are_rejected(self):
        self.assertFalse(outputs_match(np.array([1.0]), np.array([2.0])))

    def test_broadcastable_but_different_shapes_are_rejected(self):
        self.assertFalse(outputs_match(np.ones((2, 1)), np.ones((2, 2))))

    def test_dtype_changes_are_rejected(self):
        self.assertFalse(outputs_match(np.ones(2, np.float32), np.ones(2, np.float16)))

    def test_nonfinite_values_are_rejected_in_either_output(self):
        finite = np.array([1.0])
        for value in [np.nan, np.inf, -np.inf]:
            invalid = np.array([value])
            pairs = [(finite, invalid), (invalid, finite), (invalid, invalid)]
            for reference, candidate in pairs:
                with self.subTest(reference=reference, candidate=candidate):
                    self.assertFalse(outputs_match(reference, candidate))


class TimingTests(unittest.TestCase):
    def test_waits_for_work_and_excludes_warmup(self):
        # A simulated asynchronous device: submitting work takes no time;
        # waiting advances the clock. The first call is deliberately slow.
        now = [0.0]
        durations = iter([1.0, 0.002, 0.009, 0.004])

        class PendingResult:
            def block_until_ready(self):
                now[0] += next(durations)
                return self

        with patch("src.optimization.evaluator.perf_counter", side_effect=lambda: now[0]):
            latency = measure_latency(lambda: PendingResult(), (), repeats=3)
        self.assertAlmostEqual(latency, 4.0)  # Median of 2, 9, 4 milliseconds.

    def test_rejects_zero_measurements(self):
        with self.assertRaises(ValueError):
            measure_latency(lambda: None, (), repeats=0)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.inputs = (np.array([-1.0, 2.0], dtype=np.float32),)

    def assertFailure(self, result, stage, *message_parts):
        self.assertFalse(result.correct)
        self.assertEqual(result.failure_stage, stage)
        self.assertIsNone(result.reference_ms)
        self.assertIsNone(result.candidate_ms)
        self.assertIsNone(result.speedup)
        for part in message_parts:
            self.assertIn(part, result.message)

    def test_incorrect_candidate_is_not_benchmarked(self):
        with patch("src.optimization.evaluator.measure_latency") as timing:
            result = evaluate(lambda x: x + 1, lambda x: x + 2, self.inputs)
        self.assertFailure(result, "correctness", "Numerical mismatch")
        timing.assert_not_called()

    def test_correct_candidate_receives_a_ratio(self):
        with patch("src.optimization.evaluator.measure_latency", side_effect=[6.0, 3.0]):
            result = evaluate(lambda x: x + 1, lambda x: 1 + x, self.inputs)
        self.assertTrue(result.correct)
        self.assertEqual(result.speedup, 2.0)
        self.assertIsNone(result.failure_stage)
        self.assertIsNone(result.message)

    def test_success_with_real_jax_execution_and_timing(self):
        result = evaluate(lambda x: x + 1, lambda x: 1 + x, self.inputs, repeats=3)
        self.assertTrue(result.correct)
        self.assertGreater(result.reference_ms, 0)
        self.assertGreater(result.candidate_ms, 0)
        self.assertIsNone(result.failure_stage)
        self.assertIsNone(result.message)

    def test_correctness_diagnostics_report_first_mismatch(self):
        identity = lambda x: x
        cases = [
            # Shape takes precedence over dtype; dtype over nonfinite values.
            (identity, lambda x: x.astype(np.float16)[:, None],
             "Shape mismatch: expected (2,), got (2, 1)."),
            (identity, lambda x: (x * np.inf).astype(np.float16),
             "Dtype mismatch: expected float32, got float16."),
            (lambda x: x * np.nan, identity, "Nonfinite values in reference output(s)."),
            (identity, lambda x: x * np.inf, "Nonfinite values in candidate output(s)."),
            (lambda x: x * np.nan, lambda x: x * np.inf,
             "Nonfinite values in reference and candidate output(s)."),
            (identity, lambda x: x + 1,
             "Numerical mismatch: values differ beyond rtol=0.0001, atol=1e-07."),
        ]
        for reference, candidate, message in cases:
            with self.subTest(message=message):
                with patch("src.optimization.evaluator.measure_latency") as timing:
                    result = evaluate(
                        reference, candidate, self.inputs, rtol=1e-4, atol=1e-7
                    )
                self.assertFailure(result, "correctness", message)
                timing.assert_not_called()

    def test_candidate_compilation_error_is_reported(self):
        def candidate(x):
            raise ValueError("candidate cannot be lowered")

        result = evaluate(lambda x: x, candidate, self.inputs)
        self.assertFailure(result, "compile", "ValueError", "candidate cannot be lowered")

    def test_candidate_backend_compilation_error_is_reported(self):
        reference_jit = jax.jit(lambda x: x)
        candidate_jit = Mock()
        candidate_jit.lower.return_value.compile.side_effect = RuntimeError("compile failed")
        with patch("src.optimization.evaluator.jax.jit", side_effect=[reference_jit, candidate_jit]):
            result = evaluate(lambda x: x, lambda x: x, self.inputs)
        self.assertFailure(result, "compile", "RuntimeError", "compile failed")

    def test_candidate_execution_errors_include_synchronization(self):
        class PendingFailure:
            def block_until_ready(self):
                raise RuntimeError("synchronization failed")

        cases = [
            (Mock(side_effect=RuntimeError("execution failed")), "execution failed"),
            (Mock(return_value=PendingFailure()), "synchronization failed"),
        ]
        for executable, message in cases:
            with self.subTest(message=message):
                reference_jit = jax.jit(lambda x: x)
                candidate_jit = Mock()
                candidate_jit.lower.return_value.compile.return_value = executable
                with patch(
                    "src.optimization.evaluator.jax.jit",
                    side_effect=[reference_jit, candidate_jit],
                ):
                    with patch("src.optimization.evaluator.measure_latency") as timing:
                        result = evaluate(lambda x: x, lambda x: x, self.inputs)
                self.assertFailure(result, "execute", "RuntimeError", message)
                timing.assert_not_called()

    def test_candidate_benchmark_error_discards_all_timings(self):
        with patch(
            "src.optimization.evaluator.measure_latency",
            side_effect=[6.0, RuntimeError("timed execution failed")],
        ):
            result = evaluate(lambda x: x, lambda x: x, self.inputs)
        self.assertFailure(result, "benchmark", "RuntimeError", "timed execution failed")

    def test_invalid_arguments_raise_before_compilation(self):
        invalid_settings = [
            {"repeats": 0}, {"repeats": -1}, {"repeats": 1.5},
            {"rtol": -1}, {"atol": -1}, {"rtol": np.nan}, {"atol": np.inf},
            {"rtol": "invalid"},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with patch("src.optimization.evaluator.jax.jit") as jit:
                    with self.assertRaises((ValueError, TypeError)):
                        evaluate(lambda x: x, lambda x: x, self.inputs, **settings)
                jit.assert_not_called()

    def test_reference_compilation_error_propagates(self):
        def reference(x):
            raise ValueError("reference cannot be lowered")

        with self.assertRaisesRegex(ValueError, "reference cannot be lowered"):
            evaluate(reference, lambda x: x, self.inputs)

    def test_reference_execution_error_propagates(self):
        reference_jit = Mock()
        reference_jit.lower.return_value.compile.return_value.side_effect = RuntimeError(
            "reference execution failed"
        )
        with patch("src.optimization.evaluator.jax.jit", return_value=reference_jit):
            with self.assertRaisesRegex(RuntimeError, "reference execution failed"):
                evaluate(lambda x: x, lambda x: x, self.inputs)

    def test_reference_benchmark_error_propagates(self):
        with patch(
            "src.optimization.evaluator.measure_latency",
            side_effect=RuntimeError("reference benchmark failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "reference benchmark failed"):
                evaluate(lambda x: x, lambda x: x, self.inputs)

    def test_keyboard_interrupt_is_not_converted_to_feedback(self):
        def candidate(x):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            evaluate(lambda x: x, candidate, self.inputs)


class DemoTests(unittest.TestCase):
    def test_failure_details_are_displayed_without_a_speedup(self):
        result = EvaluationResult(
            correct=False, failure_stage="compile", message="ValueError: invalid shape"
        )
        output = io.StringIO()
        with patch("src.optimization.demo_evaluator.evaluate", return_value=result):
            with redirect_stdout(output):
                demo_evaluator.main()
        self.assertIn("Evaluation failed at compile: ValueError: invalid shape", output.getvalue())
        self.assertNotIn("Observed speedup", output.getvalue())

    def test_success_displays_a_speedup(self):
        result = EvaluationResult(correct=True, reference_ms=6.0, candidate_ms=3.0)
        output = io.StringIO()
        with patch("src.optimization.demo_evaluator.evaluate", return_value=result):
            with redirect_stdout(output):
                demo_evaluator.main()
        self.assertIn("Observed speedup: 2.000x", output.getvalue())


if __name__ == "__main__":
    unittest.main()
