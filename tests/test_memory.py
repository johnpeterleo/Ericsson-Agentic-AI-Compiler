"""Lessons are stored, shown newest first to other tasks only, and filtered by evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.memory import (
    MAX_LESSONS_SHOWN,
    Lesson,
    append_lessons,
    learn_from_run,
    lessons_for_task,
    lessons_text,
    load_lessons,
    parse_lessons,
)
from src.agent.tool_agent import ToolAgent
from tests.tool_agent_fakes import DOUBLING, ScriptedModel, action, evaluate_action, fast_config, tiny_task


def lesson(task, rule="Multiply right to left.", kind="worked"):
    return Lesson(rule, "a chain of matrix products", kind, {"task": task, "device": "cpu", "speedup": 1.5})


def lessons_reply(*kinds):
    return json.dumps({"lessons": [{"rule": f"rule {kind}", "when": "always", "kind": kind} for kind in kinds]})


class MemoryFileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.jsonl"

    def test_lessons_survive_a_round_trip(self):
        append_lessons([lesson("a"), lesson("b")], self.path)
        self.assertEqual(load_lessons(self.path), [lesson("a"), lesson("b")])

    def test_a_missing_file_has_no_lessons(self):
        self.assertEqual(load_lessons(self.path), [])

    def test_off_shows_nothing_and_read_shows_stored_rules(self):
        append_lessons([lesson("other")], self.path)

        self.assertEqual(lessons_text("off", "double", self.path), "")
        self.assertIn("Multiply right to left.", lessons_text("read", "double", self.path))
        self.assertIn("Multiply right to left.", lessons_text("readwrite", "double", self.path))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            lessons_text("sometimes", "double", self.path)


class SelectionTests(unittest.TestCase):
    def test_a_task_never_sees_its_own_lessons(self):
        shown = lessons_for_task([lesson("double"), lesson("other")], "double")
        self.assertEqual([item.evidence["task"] for item in shown], ["other"])

    def test_newest_first_and_capped(self):
        stored = [lesson("other", rule=f"rule {n}") for n in range(MAX_LESSONS_SHOWN + 5)]
        shown = lessons_for_task(stored, "double")

        self.assertEqual(len(shown), MAX_LESSONS_SHOWN)
        self.assertEqual(shown[0].rule, f"rule {MAX_LESSONS_SHOWN + 4}")

    def test_worked_lessons_need_a_large_enough_speedup(self):
        evidence = {"task": "double", "speedup": 1.1}
        kinds = [item.kind for item in parse_lessons(lessons_reply("worked", "did_not_help"), evidence)]
        self.assertEqual(kinds, ["did_not_help"])

    def test_malformed_lessons_are_dropped(self):
        replies = [
            json.dumps({"lessons": [{"rule": "no when"}, {"rule": "r", "when": "w", "kind": "maybe"}]}),
            json.dumps({"lessons": [{"rule": "r", "when": "w", "kind": []}]}),
            json.dumps({"lessons": None}),
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                self.assertEqual(parse_lessons(reply, {"speedup": 2.0}), [])


class LearningTests(unittest.TestCase):
    def run_agent(self, replies, lessons=""):
        model = ScriptedModel(replies)
        return ToolAgent(model, fast_config()).run(tiny_task(), lessons), model

    def test_opening_message_shows_the_lessons(self):
        _, model = self.run_agent([action("finish", summary="done")], "- [worked] Fuse it.")
        self.assertIn("Lessons from earlier runs on other programs:\n- [worked] Fuse it.", model.prompts[0])

    def test_no_lesson_call_without_a_correct_candidate(self):
        run, _ = self.run_agent([action("finish", summary="done")])
        model = ScriptedModel([])

        self.assertEqual(learn_from_run(model, run, "source"), [])
        self.assertEqual(model.prompts, [])

    def test_lessons_carry_the_evidence_of_the_run(self):
        run, _ = self.run_agent([evaluate_action(DOUBLING), action("finish", summary="done")])
        learned = learn_from_run(ScriptedModel([lessons_reply("did_not_help")]), run, "source")

        self.assertEqual(learned[0].evidence["task"], "double")
        self.assertEqual(learned[0].evidence["speedup"], run.best.speedup)


if __name__ == "__main__":
    unittest.main()
