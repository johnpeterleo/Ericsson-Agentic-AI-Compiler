"""The tool agent's loop: budgets, stopping, the leaderboard and the final re-measurement.

Timings are faked so that every speedup is exact: each candidate's code sets
the speedup it should get in the search and in the re-measurement.
"""

import unittest
from unittest.mock import patch

from src.agent.tool_agent import ToolAgent
from src.optimization.evaluator import EvaluationResult
from tests.tool_agent_fakes import (
    WRONG,
    ScriptedModel,
    action,
    evaluate_action,
    fast_config,
    tiny_task,
)

REFERENCE_MS = 1.0


def code_with_speedup(search, remeasured=None):
    return (
        "def candidate(x):\n    return x + x\n"
        f"candidate.search_speedup = {search}\n"
        f"candidate.remeasured_speedup = {remeasured if remeasured is not None else search}\n"
    )


def fake_evaluate(task):
    """Correct candidates get the speedup their code asks for; others fail as usual."""

    def evaluate(reference, function, inputs, **_):
        speedup_name = "remeasured_speedup" if inputs is task.remeasure_inputs else "search_speedup"
        if not hasattr(function, speedup_name):
            return EvaluationResult(correct=False, failure_stage="correctness", message="Numerical mismatch: faked")
        return EvaluationResult(
            correct=True, reference_ms=REFERENCE_MS, candidate_ms=REFERENCE_MS / getattr(function, speedup_name)
        )

    return evaluate


class ToolAgentTests(unittest.TestCase):
    def setUp(self):
        self.task = tiny_task()
        patcher = patch("src.agent.search.evaluate", side_effect=fake_evaluate(self.task))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_agent(self, replies, **config):
        model = ScriptedModel(replies)
        run = ToolAgent(model, fast_config(**config)).run(self.task)
        return run, model

    def test_finish_ends_the_run(self):
        run, model = self.run_agent([action("finish", summary="nothing to gain")])

        self.assertEqual(run.stop_reason, "finished")
        self.assertEqual(len(model.prompts), 1)

    def test_turn_budget_ends_the_run(self):
        run, model = self.run_agent([action("inspect", id="reference")] * 3, max_turns=3)

        self.assertEqual(run.stop_reason, "max_turns")
        self.assertEqual(len(run.turns), 3)

    def test_evaluation_budget_ends_the_run_and_cuts_the_last_turn(self):
        reply = evaluate_action(code_with_speedup(1.5), code_with_speedup(2.0), code_with_speedup(3.0))
        run, _ = self.run_agent([reply], max_evaluations=2)

        self.assertEqual(run.stop_reason, "max_evaluations")
        self.assertEqual(len(run.candidates), 2)
        self.assertIn("Only the first 2 were evaluated", run.turns[0].observation)

    def test_patience_ends_the_run_after_evaluate_turns_without_improvement(self):
        no_gain = evaluate_action(code_with_speedup(1.0))
        run, _ = self.run_agent([no_gain, action("inspect", id="reference"), no_gain], patience=2)

        self.assertEqual(run.stop_reason, "patience")
        self.assertEqual(len(run.turns), 3)

    def test_only_a_gain_beyond_the_margin_resets_patience(self):
        replies = [
            evaluate_action(code_with_speedup(1.5)),   # improvement
            evaluate_action(code_with_speedup(1.52)),  # within the 5% margin: stale 1
            evaluate_action(code_with_speedup(1.7)),   # improvement: stale 0
            evaluate_action(code_with_speedup(1.0)),   # stale 1
            evaluate_action(code_with_speedup(1.0)),   # stale 2: stop
        ]
        run, _ = self.run_agent(replies, patience=2)

        self.assertEqual(run.stop_reason, "patience")
        self.assertEqual(len(run.turns), 5)
        self.assertIn("speedup 1.52x: not better than the best (1.50x)", run.turns[1].observation)

    def test_an_invalid_reply_uses_a_turn_and_is_explained_next_turn(self):
        run, model = self.run_agent(["I think we should fuse things.", action("finish", summary="done")])

        self.assertEqual(len(run.turns), 2)
        self.assertIsNone(run.turns[0].action)
        self.assertIn("Your reply could not be used: The reply was not one JSON object", model.prompts[1])

    def test_every_prompt_holds_the_whole_transcript(self):
        replies = [action("inspect", id="reference"), action("time", code="def probe(x):\n    return x"),
                   action("finish", summary="done")]
        _, model = self.run_agent(replies)

        self.assertTrue(model.prompts[2].startswith(model.prompts[0].removesuffix("\n")))
        self.assertIn("## Turn 2: observation", model.prompts[2])
        self.assertIn("probe:", model.prompts[2])

    def test_observations_end_with_the_leaderboard_and_the_budget(self):
        reply = evaluate_action(code_with_speedup(1.2), code_with_speedup(1.5), WRONG)
        run, _ = self.run_agent([reply, action("finish", summary="done")])
        observation = run.turns[0].observation

        self.assertIn("Leaderboard (correct candidates, fastest first):\n  c2  1.50x  idea 1\n  c1  1.20x", observation)
        self.assertIn("Source of the best candidate, c2:\n" + code_with_speedup(1.5).strip(), observation)
        self.assertIn("Budget left: 11 turns, 13 evaluations.", observation)

    def test_the_result_is_chosen_by_the_remeasured_speedup(self):
        reply = evaluate_action(
            code_with_speedup(2.0, remeasured=1.1),
            code_with_speedup(1.8, remeasured=1.6),
            code_with_speedup(1.5, remeasured=1.4),
            code_with_speedup(1.2, remeasured=3.0),  # fourth in the search: not re-measured
        )
        run, _ = self.run_agent([reply, action("finish", summary="done")])
        record = run.record()

        self.assertEqual([r.candidate.id for r in run.remeasurements], ["c2", "c3", "c1"])
        self.assertEqual(record["best_id"], "c2")
        self.assertAlmostEqual(record["remeasured_speedup"], 1.6)
        self.assertAlmostEqual(record["search_speedup"], 1.8)

    def test_record_holds_calls_evaluations_and_the_transcript(self):
        run, _ = self.run_agent([evaluate_action(code_with_speedup(1.5)), action("finish", summary="done")])
        record = run.record()

        self.assertEqual((record["llm_calls"], record["evaluations"]), (2, 1))
        self.assertEqual(record["best_float_dtypes"], ["f32"])
        self.assertEqual(len(record["transcript"]["turns"]), 2)
        self.assertIn("Task: double", record["transcript"]["opening"])


if __name__ == "__main__":
    unittest.main()
