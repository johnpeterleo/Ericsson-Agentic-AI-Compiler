"""Tests for the compile time limit. A real check starts a child process (about 0.5 s)."""

import unittest
from unittest.mock import patch

import numpy as np

from src.compiler.compile_time_limit import CompileTimeoutError, check_compile_time
from src.optimization.evaluator import evaluate

INPUTS = (np.ones(4, dtype=np.float32),)


def reference(x):
    return x * 2


def candidate(x):
    return x + x


def too_slow_to_compile(slow_program):
    """A stand-in for `check_compile_time` that times out only on `slow_program`."""

    def check(program, inputs, limit_seconds):
        if program is slow_program:
            raise CompileTimeoutError(f"Compilation took longer than {limit_seconds} s.")

    return check


class CheckCompileTimeTests(unittest.TestCase):
    def test_program_that_compiles_in_time_passes(self):
        check_compile_time(reference, INPUTS, limit_seconds=60)

    def test_program_over_the_limit_raises(self):
        # Starting the child process alone takes longer than this.
        with self.assertRaises(CompileTimeoutError):
            check_compile_time(reference, INPUTS, limit_seconds=0.01)


class EvaluateWithTimeLimitTests(unittest.TestCase):
    def test_slow_candidate_is_rejected_at_compile(self):
        with patch(
            "src.optimization.evaluator.check_compile_time", too_slow_to_compile(candidate)
        ):
            result = evaluate(reference, candidate, INPUTS, repeats=1, compile_time_limit=5)

        self.assertEqual(result.failure_stage, "compile")
        self.assertIn("longer than 5 s", result.message)

    def test_slow_reference_is_raised(self):
        with patch(
            "src.optimization.evaluator.check_compile_time", too_slow_to_compile(reference)
        ), self.assertRaises(CompileTimeoutError):
            evaluate(reference, candidate, INPUTS, repeats=1, compile_time_limit=5)

    def test_no_limit_means_no_check(self):
        with patch("src.optimization.evaluator.check_compile_time") as check:
            evaluate(reference, candidate, INPUTS, repeats=1)
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
