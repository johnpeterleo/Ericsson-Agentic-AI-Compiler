"""Offline tests for the Claude CLI policy. The CLI is faked, so no request is sent."""

import json
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from src.agent.agent import AgentConfig, OptimizationAgent
from src.agent.llm_policy import (
    ClaudeCLIPolicy,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
)
from src.agent.tools import AgentToolkit, StubProfiler
from tests.benchmark_neural_network_agent import make_agent_inputs, mlp_dataset_reference

CANDIDATE_SOURCE = (
    "def candidate(hidden_weights, hidden_biases, output_weight, output_bias, input_batches):\n"
    "    activations = input_batches\n"
    "    for layer in range(hidden_weights.shape[0]):\n"
    "        activations = jax.nn.relu(activations @ hidden_weights[layer] + hidden_biases[layer])\n"
    "    outputs = activations @ output_weight + output_bias\n"
    "    return jnp.reshape(outputs, (-1, output_weight.shape[-1]))\n"
)


def cli_reply(result, *, is_error=False, returncode=0):
    stdout = json.dumps({"type": "result", "is_error": is_error, "result": result})
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def rewrite_reply():
    return cli_reply(json.dumps({
        "stop": False, "rationale": "Batch all inputs at once.", "candidate_source": CANDIDATE_SOURCE,
    }))


def small_mlp_inputs():
    return make_agent_inputs(depth=2, width=4, samples=17, batch_size=8, rng=np.random.default_rng(7))


class ClaudeCLIPolicyTests(unittest.TestCase):
    def test_command_turns_off_tools_and_uses_project_system_prompt(self):
        command = ClaudeCLIPolicy(model="claude-sonnet-5-5")._command()

        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertEqual(command[command.index("--model") + 1], "claude-sonnet-5-5")
        self.assertIn("--system-prompt", command)

    @patch("src.agent.llm_policy.subprocess.run", return_value=rewrite_reply())
    def test_agent_evaluates_the_candidate_claude_returns(self, _run):
        agent = OptimizationAgent(
            toolkit=AgentToolkit(profiler=StubProfiler(repeats=1), profile_repeats=1),
            policy=ClaudeCLIPolicy(),
            config=AgentConfig(max_iterations=1, benchmark_repeats=1),
        )

        result = agent.run(mlp_dataset_reference, small_mlp_inputs())

        evaluation = result.context.history[0].evaluation
        self.assertTrue(evaluation.correct, evaluation.message)

    @patch("src.agent.llm_policy.subprocess.run", return_value=cli_reply("Here is my idea, no JSON."))
    def test_reply_without_json_becomes_a_load_error(self, _run):
        agent = OptimizationAgent(
            toolkit=AgentToolkit(profiler=StubProfiler(repeats=1), profile_repeats=1),
            policy=ClaudeCLIPolicy(),
            config=AgentConfig(max_iterations=1, benchmark_repeats=1),
        )

        proposal = agent.policy.propose(agent.build_context(mlp_dataset_reference, small_mlp_inputs()))

        self.assertIn("JSON", proposal.load_error)

    @patch(
        "src.agent.llm_policy.subprocess.run",
        return_value=cli_reply("Claude usage limit reached", is_error=True, returncode=1),
    )
    def test_usage_limit_stops_the_run(self, _run):
        with self.assertRaises(LLMQuotaExhaustedError):
            ClaudeCLIPolicy()._ask("prompt")

    @patch(
        "src.agent.llm_policy.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=120),
    )
    def test_timeout_is_reported_as_unavailable(self, _run):
        with self.assertRaises(LLMUnavailableError):
            ClaudeCLIPolicy()._ask("prompt")


if __name__ == "__main__":
    unittest.main()
