"""The baseline arms: the existing agent at a given number of LLM calls."""

import json
import unittest


from src.agent.baselines import (
    ArmSettings,
    BaselineRunner,
    TaskSourcePolicy,
    run_best_of_n,
    run_chain,
    run_single,
)
from src.agent.llm_policy import build_user_text
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tool_prompts import FAIR_BASELINE_PROMPT
from src.agent.search import EvaluationSettings, RemeasureSettings
from src.agent.tasks import load_task
from src.agent.types import AgentContext, HLOSnapshot, ProfileReport
from tests.tool_agent_fakes import DOUBLING, WRONG, ScriptedModel, tiny_task

SETTINGS = ArmSettings(
    EvaluationSettings(repeats=1, compile_time_limit=None), RemeasureSettings(repeats=1), FAIR_BASELINE_PROMPT
)


def old_reply(source):
    return json.dumps({"stop": False, "rationale": "scripted", "candidate_source": source})


def runner_replying(replies):
    return BaselineRunner(ScriptedModel(replies), tiny_task(), SETTINGS)


class BaselineTests(unittest.TestCase):
    def test_suite_prompt_is_the_one_the_existing_policy_sends(self):
        task = load_task("suite:prefix_sums", "small")
        context = AgentContext(
            reference_fn=task.reference,
            inputs=task.inputs,
            backend="cpu",
            baseline_hlo=HLOSnapshot(text="stablehlo", backend="cpu"),
            baseline_profile=ProfileReport(backend="cpu", total_ms=1.0),
            baseline_evaluation=None,
        )
        model = ScriptedModel([old_reply(DOUBLING)])
        TaskSourcePolicy(model, task.source_text, SYSTEM_PROMPT).propose(context)

        self.assertEqual(model.prompts, [build_user_text(context)])

    def test_the_fair_arms_send_the_fair_system_prompt(self):
        model = ScriptedModel([old_reply(DOUBLING)])
        run_single(BaselineRunner(model, tiny_task(), SETTINGS))

        self.assertEqual(model.system_prompts, [FAIR_BASELINE_PROMPT])
        self.assertNotIn("inspect_compilation", FAIR_BASELINE_PROMPT)

    def test_single_makes_one_call_and_remeasures_its_candidate(self):
        record = run_single(runner_replying([old_reply(DOUBLING)])).record()

        self.assertEqual((record["llm_calls"], record["evaluations"]), (1, 1))
        self.assertEqual(record["best_id"], "a0")
        self.assertEqual(record["best_float_dtypes"], ["f32"])
        self.assertEqual(len(record["transcript"]), 1)

    def test_best_of_n_makes_independent_calls(self):
        runner = runner_replying([old_reply(WRONG), old_reply(DOUBLING), old_reply(WRONG)])
        record = run_best_of_n(runner, 3).record()

        self.assertEqual(record["llm_calls"], 3)
        self.assertEqual([c["id"] for c in record["candidates"]], ["s0-a0", "s1-a0", "s2-a0"])
        self.assertEqual(record["best_id"], "s1-a0")
        # Independent: no call shows another call's attempt.
        self.assertTrue(all("previous_attempts" not in e["prompt"] for e in record["transcript"]))

    def test_chain_shows_earlier_attempts(self):
        record = run_chain(runner_replying([old_reply(WRONG), old_reply(DOUBLING)]), 2).record()

        self.assertEqual([c["id"] for c in record["candidates"]], ["a0", "a1"])
        self.assertIn("previous_attempts", record["transcript"][1]["prompt"])

    def test_registry_tolerances_reach_the_existing_agent(self):
        task = load_task("softmax", "small")
        within_tolerance = "def candidate(x):\n    e = jnp.exp(x - jnp.max(x, -1, keepdims=True))\n    return e / jnp.sum(e, -1, keepdims=True) * (1 + 5e-5)"
        runner = BaselineRunner(ScriptedModel([old_reply(within_tolerance)]), task, SETTINGS)

        self.assertTrue(run_single(runner).candidates[0].result.correct)


if __name__ == "__main__":
    unittest.main()
