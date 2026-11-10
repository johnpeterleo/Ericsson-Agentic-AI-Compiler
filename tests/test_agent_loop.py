"""Offline tests for the feedback loop: what the agent records and what Gemini is shown."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.agent.agent import AgentConfig, DemoPolicy, OptimizationAgent
from src.agent.llm_policy import GeminiPolicy, format_context_for_llm
from src.agent.tools import AgentToolkit, StubProfiler
from src.agent.types import (
    AgentContext,
    HLOSnapshot,
    OptimizationStep,
    ProfileReport,
    RewriteProposal,
)
from src.optimization.evaluator import EvaluationResult

DOUBLING_SOURCE = "def candidate(x):\n    return x + x"
BROKEN_SOURCE = "def candidate(x) return x"


def reference(x):
    return x * 2


def make_agent(policy, max_iterations, compile_time_limit=None):
    """The compile time limit is off unless a test needs it: each check starts a process."""
    return OptimizationAgent(
        toolkit=AgentToolkit(profiler=StubProfiler(repeats=1), profile_repeats=1),
        policy=policy,
        config=AgentConfig(
            max_iterations=max_iterations,
            benchmark_repeats=1,
            compile_time_limit_seconds=compile_time_limit,
        ),
    )


def reply(**fields):
    return SimpleNamespace(text=json.dumps(fields))


class FakeGemini:
    """Returns the queued replies in order and keeps every prompt it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate_content(self, *, contents, **_):
        self.prompts.append(contents)
        return self.replies.pop(0)


def policy_replying(*replies):
    policy = GeminiPolicy(api_key="not-a-real-key")
    gemini = FakeGemini(replies)
    policy._client = SimpleNamespace(models=gemini)
    return policy, gemini


def context_with_history(history):
    return AgentContext(
        reference_fn=reference,
        inputs=(np.ones(3, dtype=np.float32),),
        backend="cpu",
        baseline_hlo=HLOSnapshot(text="baseline stablehlo", backend="cpu"),
        baseline_profile=ProfileReport(backend="cpu", total_ms=2.0),
        baseline_evaluation=None,
        history=history,
    )


def step(index, source, evaluation, optimized_hlo=None):
    hlo = HLOSnapshot(text="", backend="cpu", optimized_hlo=optimized_hlo)
    return OptimizationStep(
        index=index,
        proposal=RewriteProposal(
            rationale=f"idea {index}", metadata={"candidate_source": source}
        ),
        evaluation=evaluation,
        hlo=hlo if optimized_hlo else None,
    )


class UnusableReplyTests(unittest.TestCase):
    """A reply that cannot be used is a failed attempt, not the end of the run."""

    def setUp(self):
        self.context = context_with_history([])

    def test_source_that_does_not_parse_keeps_the_source_and_the_error(self):
        policy, _ = policy_replying(
            reply(stop=False, rationale="idea", candidate_source=BROKEN_SOURCE)
        )
        proposal = policy.propose(self.context)
        self.assertFalse(proposal.stop)
        self.assertIsNone(proposal.candidate_fn)
        self.assertIn("SyntaxError", proposal.load_error)
        self.assertEqual(proposal.metadata["candidate_source"], BROKEN_SOURCE)

    def test_source_without_a_candidate_function_is_a_load_error(self):
        policy, _ = policy_replying(
            reply(stop=False, rationale="idea", candidate_source="def other(x):\n    return x")
        )
        self.assertIn("candidate", policy.propose(self.context).load_error)

    def test_reply_without_json_is_a_load_error(self):
        policy, _ = policy_replying(SimpleNamespace(text="I think you should use vmap."))
        proposal = policy.propose(self.context)
        self.assertFalse(proposal.stop)
        self.assertIn("JSON", proposal.load_error)

    def test_missing_source_is_a_load_error(self):
        policy, _ = policy_replying(reply(stop=False, rationale="idea", candidate_source=None))
        self.assertIn("candidate_source", policy.propose(self.context).load_error)

    def test_only_an_explicit_stop_stops(self):
        policy, _ = policy_replying(reply(stop=True, rationale="nothing left to try"))
        proposal = policy.propose(self.context)
        self.assertTrue(proposal.stop)
        self.assertIsNone(proposal.load_error)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.inputs = (np.array([1.0, -2.0, 3.0], dtype=np.float32),)

    def test_agent_recovers_after_an_unusable_reply(self):
        policy, gemini = policy_replying(
            reply(stop=False, rationale="first idea", candidate_source=BROKEN_SOURCE),
            reply(stop=False, rationale="fixed syntax", candidate_source=DOUBLING_SOURCE),
        )
        result = make_agent(policy, max_iterations=2).run(reference, self.inputs)

        failed, fixed = result.context.history
        self.assertEqual(failed.evaluation.failure_stage, "load")
        self.assertIn("SyntaxError", failed.evaluation.message)
        self.assertTrue(fixed.evaluation.correct, fixed.evaluation.message)
        self.assertEqual(result.status, "max_iterations")

        first_prompt, second_prompt = gemini.prompts
        self.assertNotIn("previous_attempts", first_prompt)
        self.assertIn("attempt 0: rejected at load: SyntaxError", second_prompt)
        self.assertIn(BROKEN_SOURCE, second_prompt)

    def test_wrong_candidate_is_shown_back_with_its_code(self):
        wrong_source = "def candidate(x):\n    return x + 2"
        policy, gemini = policy_replying(
            reply(stop=False, rationale="add instead", candidate_source=wrong_source),
            reply(stop=True, rationale="giving up"),
        )
        result = make_agent(policy, max_iterations=2).run(reference, self.inputs)

        self.assertEqual(result.status, "stopped_by_policy")
        second_prompt = gemini.prompts[1]
        self.assertIn("attempt 0: rejected at correctness: Numerical mismatch", second_prompt)
        self.assertIn(wrong_source, second_prompt)
        self.assertIn("No attempt has been correct yet", second_prompt)

    def test_holdout_input_reaches_the_evaluator(self):
        def right_only_when_x_is_two(x):
            return x * x

        twos = (np.full(3, 2.0, dtype=np.float32),)
        threes = (np.full(3, 3.0, dtype=np.float32),)
        agent = make_agent(DemoPolicy(candidates=[right_only_when_x_is_two]), max_iterations=1)

        result = agent.run(reference, twos, holdout_inputs=threes)

        evaluation = result.context.history[0].evaluation
        self.assertEqual(evaluation.failure_stage, "correctness")
        self.assertIn("held-out", evaluation.message)

    def test_compile_time_limit_from_the_config_reaches_the_evaluator(self):
        agent = make_agent(DemoPolicy(candidates=[reference]), 1, compile_time_limit=5)

        with patch("src.optimization.evaluator.check_compile_time") as check:
            agent.run(reference, (np.ones(3, dtype=np.float32),))

        self.assertEqual([call.args[2] for call in check.call_args_list], [5, 5])


class PromptTests(unittest.TestCase):
    def test_first_prompt_has_no_attempts_section(self):
        self.assertNotIn("previous_attempts", format_context_for_llm(context_with_history([])))

    def test_best_correct_attempt_is_named_with_its_compiled_form(self):
        history = [
            step(0, "def candidate(x):\n    return x + x  # slow",
                 EvaluationResult(correct=True, reference_ms=2.0, candidate_ms=1.6)),
            step(1, "def candidate(x):\n    return 2 * x  # fast",
                 EvaluationResult(correct=True, reference_ms=2.0, candidate_ms=0.5),
                 optimized_hlo="HLO of the fast candidate"),
            step(2, "def candidate(x):\n    return x  # wrong",
                 EvaluationResult(correct=False, failure_stage="correctness", message="mismatch")),
        ]
        prompt = format_context_for_llm(context_with_history(history))

        for source_marker in ("# slow", "# fast", "# wrong"):
            self.assertIn(source_marker, prompt)
        self.assertIn("attempt 0: correct, speedup 1.250x", prompt)
        self.assertIn("attempt 2: rejected at correctness: mismatch", prompt)
        self.assertIn("best_so_far: attempt 1 (4.000x)", prompt)
        self.assertIn("HLO of the fast candidate", prompt)

    def test_best_step_ignores_rejected_attempts(self):
        rejected = step(
            0, "", EvaluationResult(correct=False, failure_stage="compile", message="bad")
        )
        context = context_with_history([rejected])
        self.assertIsNone(context.best_step())
        self.assertIsNone(context.best_speedup())


if __name__ == "__main__":
    unittest.main()
