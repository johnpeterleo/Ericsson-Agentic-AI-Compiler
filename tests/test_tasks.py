"""Both program sets load into the same Task format."""

import unittest

import numpy as np

from src.agent.tasks import load_task, task_names, visible_source

PROGRAM_TEXT = '''import numpy as np

def reference(x):
    return x


def numpy_reference(x):
    return np.asarray(x)


def make_inputs(seed=0, size="default"):
    return (np.ones(3),)
'''


class TaskLoaderTests(unittest.TestCase):
    def test_suite_task_shows_only_the_reference_and_keeps_the_expert(self):
        task = load_task("suite:prefix_sums", "small")

        self.assertTrue(task.source_text.startswith("def prefix_sums("))
        self.assertIsNotNone(task.expert)
        self.assertEqual(task.inputs[0].shape, (6, 6))

    def test_registry_task_shows_helpers_and_uses_the_program_tolerances(self):
        task = load_task("transformer_block", "small")

        self.assertIn("def _layer_norm(", task.source_text)
        self.assertIn("HEAD_DIM = 64", task.source_text)
        self.assertNotIn("def make_inputs", task.source_text)
        self.assertNotIn("def numpy_reference", task.source_text)
        self.assertEqual((task.rtol, task.atol), (1e-2, 1e-2))
        self.assertIsNone(task.expert)

    def test_the_three_input_cases_differ(self):
        task = load_task("softmax", "small")
        first, holdout, remeasure = task.inputs[0], task.holdout_inputs[0], task.remeasure_inputs[0]

        self.assertFalse(np.array_equal(first, holdout))
        self.assertFalse(np.array_equal(holdout, remeasure))

    def test_visible_source_removes_the_hidden_functions(self):
        self.assertEqual(visible_source(PROGRAM_TEXT), "import numpy as np\n\ndef reference(x):\n    return x\n")

    def test_names_keep_the_two_attention_programs_apart(self):
        names = task_names()

        self.assertIn("attention", names)
        self.assertIn("suite:attention", names)

    def test_unknown_size_is_rejected(self):
        with self.assertRaises(ValueError):
            load_task("softmax", "huge")


if __name__ == "__main__":
    unittest.main()
