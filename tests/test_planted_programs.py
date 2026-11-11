"""The planted programs: each expert is right, and each reference stands on its own."""

import inspect
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.agent.tasks import SUITE_ATOL, SUITE_RTOL, load_task, task_names
from src.benchmarks.planted import PLANTED_PROGRAMS
from src.optimization.evaluator import evaluate


def small_inputs(program, seed):
    return program.make_inputs(np.random.default_rng(seed), program.test_size)


class PlantedProgramTests(unittest.TestCase):
    def test_expert_matches_reference_on_two_input_cases(self):
        for program in PLANTED_PROGRAMS:
            with self.subTest(program.name):
                result = evaluate(
                    program.reference, program.expert, small_inputs(program, seed=1),
                    repeats=1, rtol=SUITE_RTOL, atol=SUITE_ATOL,
                    holdout_inputs=small_inputs(program, seed=2),
                )
                self.assertTrue(result.correct, result.message)

    def test_reference_source_runs_without_the_rest_of_the_module(self):
        """The agent is shown only this source, with `jax` and `jnp` available."""
        for program in PLANTED_PROGRAMS:
            with self.subTest(program.name):
                namespace = {"jax": jax, "jnp": jnp}
                exec(inspect.getsource(program.reference), namespace)
                inputs = small_inputs(program, seed=1)
                np.testing.assert_array_equal(namespace[program.name](*inputs), program.reference(*inputs))

    def test_reference_source_does_not_name_the_expert(self):
        for program in PLANTED_PROGRAMS:
            with self.subTest(program.name):
                self.assertNotIn("expert", inspect.getsource(program.reference))

    def test_planted_programs_load_as_tasks_with_their_expert(self):
        task = load_task("planted:local_standardize", "small")

        self.assertIn("planted:local_standardize", task_names())
        self.assertTrue(task.source_text.startswith("def local_standardize("))
        self.assertIsNotNone(task.expert)


if __name__ == "__main__":
    unittest.main()
