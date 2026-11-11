"""Both policies can be asked with another system prompt; `propose` still sends the old one."""

import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.llm_policy import ClaudeCLIPolicy, GeminiPolicy
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.text_model import RecordingModel, create_text_model
from tests.tool_agent_fakes import ScriptedModel

OTHER_PROMPT = "You are a different agent."


class FakeModels:
    def __init__(self):
        self.configs = []

    def generate_content(self, *, config, **_):
        self.configs.append(config)
        return SimpleNamespace(text="the reply")


def cli_result(text):
    stdout = json.dumps({"type": "result", "is_error": False, "result": text})
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TextModelTests(unittest.TestCase):
    def test_gemini_ask_sends_the_given_system_prompt(self):
        policy = GeminiPolicy(api_key="not-a-real-key")
        models = FakeModels()
        policy._client = SimpleNamespace(models=models)

        self.assertEqual(policy.ask(OTHER_PROMPT, "hello"), "the reply")
        self.assertEqual(models.configs, [{"system_instruction": OTHER_PROMPT}])

    def test_gemini_default_is_still_the_baseline_prompt(self):
        policy = GeminiPolicy(api_key="not-a-real-key")
        models = FakeModels()
        policy._client = SimpleNamespace(models=models)

        policy._generate_with_retries("hello")
        self.assertEqual(models.configs, [{"system_instruction": SYSTEM_PROMPT}])

    def test_claude_ask_sends_the_given_system_prompt(self):
        with patch("src.agent.llm_policy.subprocess.run", return_value=cli_result("the reply")) as run:
            reply = ClaudeCLIPolicy().ask(OTHER_PROMPT, "hello")
        command = run.call_args.args[0]

        self.assertEqual(reply, "the reply")
        self.assertEqual(command[command.index("--system-prompt") + 1], OTHER_PROMPT)
        self.assertEqual(run.call_args.kwargs["input"], "hello")

    def test_claude_command_defaults_to_the_baseline_prompt(self):
        command = ClaudeCLIPolicy()._command()
        self.assertEqual(command[command.index("--system-prompt") + 1], SYSTEM_PROMPT)

    def test_recording_model_counts_calls_and_keeps_exchanges(self):
        model = RecordingModel(ScriptedModel(["first", "second"]))
        model.ask(OTHER_PROMPT, "a")
        model.ask(OTHER_PROMPT, "b")

        self.assertEqual(model.calls, 2)
        self.assertEqual(model.exchanges[1], {"prompt": "b", "reply": "second"})

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            create_text_model("gpt")


if __name__ == "__main__":
    unittest.main()
