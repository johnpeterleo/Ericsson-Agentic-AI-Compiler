"""compare_agents end to end with a scripted model: every arm runs and every record names the model."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import compare_agents
from src.agent.prompts import SYSTEM_PROMPT
from tests.tool_agent_fakes import ScriptedModel, action

CUMSUM_SOURCE = "def candidate(sequences):\n    return jnp.cumsum(sequences, axis=-1)"


def old_reply():
    return json.dumps({"stop": False, "rationale": "cumsum", "candidate_source": CUMSUM_SOURCE})


class CompareAgentsTests(unittest.TestCase):
    def test_every_arm_runs_and_records_the_model(self):
        # One call per arm: single, best_of_n, chain and chain_original each send one old-format reply.
        model = ScriptedModel([old_reply()] * 4 + [action("finish", summary="nothing to do")])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compare_agents, "RESULTS_DIR", Path(directory)), \
                 patch.object(compare_agents, "create_text_model", return_value=model):
                compare_agents.main([
                    "--tasks", "suite:prefix_sums", "--calls", "1", "--repeats", "1",
                    "--remeasure-repeats", "1", "--compile-time-limit", "0",
                ])
            [results] = Path(directory).glob("*.jsonl")
            rows = [json.loads(line) for line in results.read_text().splitlines()]

        self.assertEqual([row["arm"] for row in rows], list(compare_agents.ARMS))
        self.assertTrue(all(row["model_name"] == "scripted-model" for row in rows))
        self.assertTrue(all(row["status"] == "done" for row in rows))
        self.assertEqual(model.system_prompts[3], SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
