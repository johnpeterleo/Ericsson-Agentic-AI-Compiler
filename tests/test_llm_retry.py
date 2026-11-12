"""Offline tests for Gemini retries and saving agent results. No API calls."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from google.genai.errors import ClientError, ServerError

from src.agent.llm_policy import (
    GeminiPolicy,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
    RETRY_WAIT_SECONDS,
)
from src.agent.types import OptimizationStep, RewriteProposal
from src.optimization.evaluator import EvaluationResult
from tests.benchmark_neural_network_agent import (
    _append_record,
    _step_record,
    make_agent_inputs,
    mlp_dataset_reference,
    mlp_vectorized_reference,
)


def overloaded():
    return ServerError(503, {"error": {"message": "overloaded"}})


def daily_quota_used_up():
    return ClientError(
        429, {"error": {"message": "Generate content requests per day limit exceeded"}}
    )


def rate_limited():
    return ClientError(429, {"error": {"message": "Too many requests per minute"}})


def bad_key():
    return ClientError(403, {"error": {"message": "forbidden"}})


class FakeModels:
    """Raises each queued error in turn, then returns a fixed reply.

    The queued errors are per model *name*, so a test can make one model busy and
    check that the next one is asked instead of the same one again.
    """

    def __init__(self, errors, *, per_model=None):
        self.errors = list(errors)
        self.per_model = per_model or {}
        self.calls = 0
        self.asked: list[str] = []

    def generate_content(self, **kwargs):
        self.calls += 1
        name = kwargs.get("model")
        self.asked.append(name)
        queue = self.per_model.get(name)
        if queue:
            raise queue.pop(0)
        if self.errors:
            raise self.errors.pop(0)
        return f"reply from {name}"


def policy_with_errors(errors, *, fallback_models=(), **kwargs):
    policy = GeminiPolicy(
        api_key="not-a-real-key", model="primary", fallback_models=fallback_models
    )
    models = FakeModels(errors, **kwargs)
    policy._client = SimpleNamespace(models=models)
    return policy, models


@patch("src.agent.llm_policy.time.sleep")
class RetryTests(unittest.TestCase):
    """A per-minute limit or a network error is retried; a busy model is swapped."""

    def test_per_minute_rate_limit_is_retried_on_the_same_model(self, sleep):
        policy, models = policy_with_errors([rate_limited()])
        self.assertTrue(policy._generate_with_retries("prompt").startswith("reply"))
        self.assertEqual(models.calls, 2)
        self.assertEqual(set(models.asked), {"primary"})
        sleep.assert_called_once()

    def test_network_timeout_is_retried_on_the_same_model(self, sleep):
        policy, models = policy_with_errors([httpx.ReadTimeout("slow")])
        self.assertTrue(policy._generate_with_retries("prompt").startswith("reply"))
        self.assertEqual(models.calls, 2)
        sleep.assert_called_once()

    def test_succeeds_after_every_retry_within_the_same_model(self, _):
        policy, models = policy_with_errors([rate_limited()] * len(RETRY_WAIT_SECONDS))
        self.assertTrue(policy._generate_with_retries("prompt").startswith("reply"))
        self.assertEqual(models.calls, len(RETRY_WAIT_SECONDS) + 1)

    def test_gives_up_after_every_retry_when_there_is_no_other_model(self, _):
        attempts = len(RETRY_WAIT_SECONDS) + 1
        policy, models = policy_with_errors([rate_limited()] * attempts)
        with self.assertRaises(LLMUnavailableError):
            policy._generate_with_retries("prompt")
        self.assertEqual(models.calls, attempts)

    def test_used_up_daily_quota_is_not_retried_and_is_named(self, sleep):
        policy, models = policy_with_errors([daily_quota_used_up()])
        with self.assertRaises(LLMQuotaExhaustedError):
            policy._generate_with_retries("prompt")
        self.assertEqual(models.calls, 1)
        sleep.assert_not_called()

    def test_used_up_daily_quota_counts_as_unavailable_for_existing_callers(self, _):
        self.assertTrue(issubclass(LLMQuotaExhaustedError, LLMUnavailableError))

    def test_permanent_error_is_not_retried(self, sleep):
        policy, models = policy_with_errors([bad_key()])
        with self.assertRaises(ClientError):
            policy._generate_with_retries("prompt")
        self.assertEqual(models.calls, 1)
        sleep.assert_not_called()


@patch("src.agent.llm_policy.time.sleep")
class ModelRotationTests(unittest.TestCase):
    """The bug behind "status 504; retrying in 30 s" every day for three days."""

    def test_busy_model_is_swapped_without_waiting(self, sleep):
        policy, models = policy_with_errors([], fallback_models=("backup",))
        models.per_model = {"primary": [ServerError(504, {"error": {"message": "timeout"}})],
                            "backup": []}
        reply = policy._generate_with_retries("prompt")
        self.assertEqual(reply, "reply from backup")
        self.assertEqual(models.asked, ["primary", "backup"])
        sleep.assert_not_called()

    def test_a_model_that_failed_is_not_asked_again_in_the_same_run(self, _):
        policy, models = policy_with_errors([], fallback_models=("backup",))
        models.per_model = {"primary": [overloaded(), overloaded()]}
        self.assertEqual(policy._generate_with_retries("p"), "reply from backup")
        self.assertEqual(policy._generate_with_retries("p"), "reply from backup")
        self.assertEqual(models.asked, ["primary", "backup", "backup"])

    def test_quota_is_per_model_so_the_next_model_is_tried(self, _):
        policy, models = policy_with_errors([], fallback_models=("backup",))
        models.per_model = {"primary": [daily_quota_used_up()]}
        self.assertEqual(policy._generate_with_retries("p"), "reply from backup")
        self.assertEqual(policy.models, ("backup",))

    def test_every_model_out_of_quota_is_a_quota_error(self, _):
        policy, models = policy_with_errors([], fallback_models=("backup",))
        models.per_model = {
            "primary": [daily_quota_used_up()],
            "backup": [daily_quota_used_up()],
        }
        with self.assertRaises(LLMQuotaExhaustedError):
            policy._generate_with_retries("p")

    def test_every_model_busy_is_an_unavailable_error(self, _):
        policy, models = policy_with_errors([], fallback_models=("backup",))
        models.per_model = {"primary": [overloaded()], "backup": [overloaded()]}
        with self.assertRaises(LLMUnavailableError):
            policy._generate_with_retries("p")

    def test_duplicate_model_names_are_asked_once(self, _):
        policy = GeminiPolicy(
            api_key="k", model="same", fallback_models=("same", "other")
        )
        self.assertEqual(policy.models, ("same", "other"))


class SavedResultTests(unittest.TestCase):
    def test_saved_line_keeps_source_and_timings(self):
        step = OptimizationStep(
            index=0,
            proposal=RewriteProposal(
                rationale="remove scan",
                metadata={"candidate_source": "def candidate(x): return x"},
            ),
            evaluation=EvaluationResult(correct=True, reference_ms=2.0, candidate_ms=1.0),
        )
        with tempfile.TemporaryDirectory() as folder:
            results_path = Path(folder) / "new-folder" / "run.jsonl"
            _append_record(results_path, _step_record(step))
            saved = json.loads(results_path.read_text())
        self.assertEqual(saved["candidate_source"], "def candidate(x): return x")
        self.assertEqual(saved["speedup"], 2.0)


class ReferenceTests(unittest.TestCase):
    def test_vectorized_reference_matches_scan_reference(self):
        inputs = make_agent_inputs(
            depth=3, width=8, samples=20, batch_size=4, rng=np.random.default_rng(1)
        )
        np.testing.assert_allclose(
            mlp_vectorized_reference(*inputs),
            mlp_dataset_reference(*inputs),
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
