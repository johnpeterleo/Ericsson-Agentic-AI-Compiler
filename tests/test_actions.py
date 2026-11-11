"""Parsing replies into actions, and what each action's observation says."""

import json
import unittest

from src.agent.actions import (
    ActionError,
    Workspace,
    parse_action,
    run_evaluate,
    run_inspect,
    run_time,
)
from src.agent.feedback import summarize_compiled
from src.agent.search import EvaluationSettings, SearchState
from tests.tool_agent_fakes import DOUBLING, WRONG, action, tiny_task

IMPROVEMENT_MARGIN = 1.05


def make_workspace():
    task = tiny_task()
    return Workspace(
        task=task,
        settings=EvaluationSettings(repeats=1, compile_time_limit=None),
        reference_summary=summarize_compiled(task.reference, task.inputs),
        search=SearchState(IMPROVEMENT_MARGIN),
    )


def spec(code, idea="an idea"):
    return {"idea": idea, "code": code}


class ParseTests(unittest.TestCase):
    def test_reads_the_action_and_its_fields_from_a_fenced_reply(self):
        reply = "Here you go:\n```json\n" + action("inspect", id="reference") + "\n```"
        parsed = parse_action(reply)

        self.assertEqual((parsed.name, parsed.fields["id"], parsed.thought), ("inspect", "reference", "scripted"))

    def test_invalid_replies_say_what_was_wrong(self):
        cases = [
            ("no json here", "not one JSON object"),
            (action("profile"), "Unknown action 'profile'"),
            (action("time"), "`code` must be a non-empty string"),
            (action("inspect", id="c1", code="x"), "exactly one of `code` or `id`"),
            (action("evaluate", candidates=[]), "1 to 4"),
            (action("evaluate", candidates=[spec(DOUBLING)] * 5), "1 to 4"),
            (action("evaluate", candidates=[{"code": DOUBLING}]), "`idea`"),
            (action("finish"), "`summary`"),
        ]
        for reply, message in cases:
            with self.subTest(reply=reply):
                with self.assertRaisesRegex(ActionError, message):
                    parse_action(reply)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.workspace = make_workspace()

    def test_inspect_code_compares_with_the_reference(self):
        observation = run_inspect(parse_action(action("inspect", code=DOUBLING)), self.workspace)

        self.assertIn("Compiled form of your code:", observation)
        self.assertIn("launched operations:", observation)
        self.assertIn("Compared with the reference:", observation)

    def test_inspect_unknown_id_is_an_action_error(self):
        with self.assertRaisesRegex(ActionError, "'c9'"):
            run_inspect(parse_action(action("inspect", id="c9")), self.workspace)

    def test_inspect_reports_a_compile_error(self):
        code = "def candidate(x):\n    return x @ x[None]"
        observation = run_inspect(parse_action(action("inspect", code=code)), self.workspace)

        self.assertIn("Could not compile it", observation)

    def test_time_reports_the_probe_and_the_reference(self):
        code = "def probe(x):\n    return x + 1"
        observation = run_time(parse_action(action("time", code=code)), self.workspace)

        self.assertRegex(observation, r"probe: [\d.]+ ms; the whole reference, timed alongside: [\d.]+ ms")

    def test_time_reports_a_probe_that_returns_several_arrays(self):
        code = "def probe(x):\n    return x, x"
        observation = run_time(parse_action(action("time", code=code)), self.workspace)

        self.assertIn("Could not time the probe", observation)

    def test_evaluate_gives_each_candidate_an_id_and_an_outcome(self):
        observation = run_evaluate([spec(DOUBLING), spec(WRONG), spec("def nothing(): pass")], self.workspace)
        lines = observation.splitlines()

        self.assertRegex(lines[0], r"^c1 \(an idea\): correct\. reference [\d.]+ ms, candidate [\d.]+ ms, speedup")
        self.assertIn("Compiled operations against the reference:", lines[0])
        self.assertIn("c2 (an idea): rejected at correctness. On the first input: Numerical mismatch", lines[1])
        self.assertIn("times the allowed error", lines[1])
        self.assertIn("c3 (an idea): rejected at load. ValueError: The code must define", lines[2])
        self.assertEqual([c.id for c in self.workspace.search.candidates], ["c1", "c2", "c3"])

    def test_evaluate_names_a_held_out_failure(self):
        # Right on the first input only: the candidate hard-codes its values.
        values = json.dumps((self.workspace.task.inputs[0] * 2).tolist())
        code = f"def candidate(x):\n    return jnp.array({values}, dtype=jnp.float32)"
        observation = run_evaluate([spec(code)], self.workspace)

        self.assertIn("rejected at correctness. On the held-out input: Numerical mismatch", observation)


if __name__ == "__main__":
    unittest.main()
