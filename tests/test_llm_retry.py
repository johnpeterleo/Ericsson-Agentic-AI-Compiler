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
    """Raises each queued error in turn, then returns a fixed reply."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def generate_content(self, **_):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "reply"


def policy_with_errors(errors):
    policy = GeminiPolicy(api_key="not-a-real-key")
    models = FakeModels(errors)
    policy._client = SimpleNamespace(models=models)
    return policy, models


@patch("src.agent.llm_policy.time.sleep")
class RetryTests(unittest.TestCase):
    def test_temporary_error_is_retried_until_success(self, sleep):
        policy, models = policy_with_errors([overloaded(), overloaded()])
        self.assertEqual(policy._generate_with_retries("prompt"), "reply")
        self.assertEqual(models.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_succeeds_on_the_final_attempt(self, _):
        policy, _models = policy_with_errors([overloaded()] * len(RETRY_WAIT_SECONDS))
        self.assertEqual(policy._generate_with_retries("prompt"), "reply")

    def test_gives_up_after_every_retry(self, _):
        attempts = len(RETRY_WAIT_SECONDS) + 1
        policy, models = policy_with_errors([overloaded()] * attempts)
        with self.assertRaises(LLMUnavailableError):
            policy._generate_with_retries("prompt")
        self.assertEqual(models.calls, attempts)

    def test_timeout_is_retried(self, sleep):
        policy, models = policy_with_errors([httpx.ReadTimeout("slow")])
        self.assertEqual(policy._generate_with_retries("prompt"), "reply")
        self.assertEqual(models.calls, 2)
        sleep.assert_called_once()

    def test_per_minute_rate_limit_is_retried(self, sleep):
        policy, models = policy_with_errors([rate_limited()])
        self.assertEqual(policy._generate_with_retries("prompt"), "reply")
        self.assertEqual(models.calls, 2)

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
