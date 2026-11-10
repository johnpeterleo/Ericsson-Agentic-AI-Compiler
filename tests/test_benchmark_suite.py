"""Tests for the benchmark programs and for how the suite decides "solved"."""

import contextlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from experiments import run_suite
from experiments.run_suite import SOLVED_FRACTION, first_solving_step
from src.agent.llm_policy import LLMQuotaExhaustedError
from src.compiler.compile_time_limit import CompileTimeoutError
from src.agent.types import OptimizationStep, RewriteProposal
from src.benchmarks.programs import PROGRAMS, PROGRAMS_BY_NAME
from src.optimization.evaluator import EvaluationResult, evaluate


def small_inputs(program, seed):
    return program.make_inputs(np.random.default_rng(seed), program.test_size)


class BenchmarkProgramTests(unittest.TestCase):
    def test_names_are_unique(self):
        names = [program.name for program in PROGRAMS]
        self.assertEqual(len(names), len(set(names)))

    def test_expert_matches_reference_on_two_input_cases(self):
        for program in PROGRAMS:
            with self.subTest(program.name):
                result = evaluate(
                    program.reference,
                    program.expert,
                    small_inputs(program, seed=1),
                    repeats=1,
                    holdout_inputs=small_inputs(program, seed=2),
                )
                self.assertTrue(result.correct, result.message)

    def test_reference_source_runs_without_the_rest_of_the_module(self):
        """The agent is shown only this source, with `jax` and `jnp` available."""
        for program in PROGRAMS:
            with self.subTest(program.name):
                namespace = {"jax": jax, "jnp": jnp}
                exec(inspect.getsource(program.reference), namespace)
                standalone = namespace[program.name]
                inputs = small_inputs(program, seed=1)
                np.testing.assert_array_equal(standalone(*inputs), program.reference(*inputs))

    def test_reference_source_does_not_name_the_expert(self):
        for program in PROGRAMS:
            with self.subTest(program.name):
                self.assertNotIn("expert", inspect.getsource(program.reference))


def attempt(index, speedup):
    """A correct attempt with the given speedup, or a rejected one if it is None."""
    if speedup is None:
        evaluation = EvaluationResult(correct=False, failure_stage="correctness")
    else:
        evaluation = EvaluationResult(correct=True, reference_ms=speedup, candidate_ms=1.0)
    return OptimizationStep(
        index=index, proposal=RewriteProposal(rationale=""), evaluation=evaluation
    )


class SolvedStepTests(unittest.TestCase):
    EXPERT_SPEEDUP = 10.0

    def test_first_attempt_that_reaches_the_target_counts(self):
        target = SOLVED_FRACTION * self.EXPERT_SPEEDUP
        history = [attempt(0, 2.0), attempt(1, None), attempt(2, target), attempt(3, 12.0)]
        self.assertEqual(first_solving_step(history, self.EXPERT_SPEEDUP), 2)

    def test_slow_or_rejected_attempts_do_not_count(self):
        history = [attempt(0, 2.0), attempt(1, None)]
        self.assertIsNone(first_solving_step(history, self.EXPERT_SPEEDUP))

    def test_no_attempts_means_not_solved(self):
        self.assertIsNone(first_solving_step([], self.EXPERT_SPEEDUP))


class ScriptedPolicy:
    """Stands in for Gemini: returns the queued proposals, then reports a used-up quota."""

    def __init__(self, proposals):
        self.proposals = list(proposals)

    def propose(self, context):
        if not self.proposals:
            raise LLMQuotaExhaustedError("no requests left today")
        return self.proposals.pop(0)


class SuiteRunnerTests(unittest.TestCase):
    """Runs the real runner on real programs. Only Gemini is replaced."""

    def run_suite_with(self, policy, programs):
        arguments = ["--agent", "--programs", programs, "--repeats", "1", "--iterations", "2"]
        with tempfile.TemporaryDirectory() as folder, patch.object(
            run_suite, "RESULTS_DIR", Path(folder)
        ), patch.object(run_suite, "GeminiPolicy", return_value=policy):
            with contextlib.redirect_stdout(io.StringIO()):
                run_suite.main(arguments)
            (results_file,) = Path(folder).iterdir()
            return [json.loads(line) for line in results_file.read_text().splitlines()]

    def test_every_attempt_is_saved_next_to_the_expert_result(self):
        unusable = RewriteProposal(rationale="first idea", load_error="SyntaxError: bad")
        expert_rewrite = RewriteProposal(
            rationale="second idea", candidate_fn=PROGRAMS_BY_NAME["prefix_sums"].expert
        )
        records = self.run_suite_with(ScriptedPolicy([unusable, expert_rewrite]), "prefix_sums")

        expert, first_attempt, second_attempt = records
        self.assertEqual((expert["kind"], expert["correct"]), ("expert", True))
        self.assertEqual(first_attempt["failure_stage"], "load")
        self.assertTrue(second_attempt["correct"], second_attempt["message"])
        self.assertEqual({record["program"] for record in records}, {"prefix_sums"})

    def test_used_up_quota_stops_the_run_and_keeps_what_was_measured(self):
        records = self.run_suite_with(ScriptedPolicy([]), "prefix_sums,matrix_power")

        self.assertEqual([record["kind"] for record in records], ["expert", "agent"])
        self.assertEqual(records[1]["status"], "quota")
        self.assertEqual({record["program"] for record in records}, {"prefix_sums"})

    def test_reference_that_compiles_too_slowly_is_skipped_and_the_run_goes_on(self):
        slow_reference = PROGRAMS_BY_NAME["prefix_sums"].reference

        def check(program, inputs, limit_seconds):
            if program is slow_reference:
                raise CompileTimeoutError("Compilation took longer than 60 s.")

        with patch("src.optimization.evaluator.check_compile_time", check):
            records = self.run_suite_with(ScriptedPolicy([]), "prefix_sums,matrix_power")

        self.assertEqual(records[0]["status"], "compile_timeout")
        self.assertEqual(records[1]["program"], "matrix_power")


if __name__ == "__main__":
    unittest.main()
