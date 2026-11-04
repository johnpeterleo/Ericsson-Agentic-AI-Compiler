"""Offline smoke test for MLP execution through the optimization agent."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.agent.agent import AgentConfig, DemoPolicy, OptimizationAgent
from src.agent.tools import AgentToolkit, StubProfiler
from tests.benchmark_neural_network_agent import make_agent_inputs, mlp_dataset_reference


def scripted_candidate(hidden_weights, hidden_biases, output_weight, output_bias, input_batches):
    """Equivalent vectorized rewrite for testing without an API request."""
    activations = input_batches
    for layer in range(hidden_weights.shape[0]):
        activations = jax.nn.relu(
            activations @ hidden_weights[layer] + hidden_biases[layer]
        )
    outputs = activations @ output_weight + output_bias
    return jnp.reshape(outputs, (-1, output_weight.shape[-1]))


class NeuralNetworkAgentTests(unittest.TestCase):
    def test_agent_evaluates_mlp_candidate_without_llm_network_call(self):
        inputs = make_agent_inputs(
            depth=2,
            width=4,
            samples=17,
            batch_size=8,
            rng=np.random.default_rng(7),
        )
        agent = OptimizationAgent(
            toolkit=AgentToolkit(profiler=StubProfiler(repeats=1), profile_repeats=1),
            policy=DemoPolicy(candidates=[scripted_candidate]),
            config=AgentConfig(max_iterations=1, benchmark_repeats=1),
        )

        result = agent.run(mlp_dataset_reference, inputs)

        self.assertEqual(len(result.context.history), 1)
        evaluation = result.context.history[0].evaluation
        self.assertIsNotNone(evaluation)
        self.assertTrue(evaluation.correct, evaluation.message)
        self.assertIsNotNone(evaluation.speedup)


if __name__ == "__main__":
    unittest.main()
