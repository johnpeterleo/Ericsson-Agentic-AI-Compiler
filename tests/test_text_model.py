"""The text models: any system prompt can be sent, and `propose` still sends the old one."""

import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.llm_policy import (
    ClaudeCLIPolicy,
    GeminiPolicy,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
)
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.text_model import CodexCLIModel, RecordingModel, create_text_model, model_details
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


def fake_codex(reply, returncode=0, stderr=""):
    """Stands in for `subprocess.run`: writes `reply` where Codex writes its last message."""

    def run(command, **_):
        if reply is not None:
            Path(command[command.index("--output-last-message") + 1]).write_text(reply)
        return subprocess.CompletedProcess(args=command, returncode=returncode, stdout="", stderr=stderr)

    return run


class CodexTests(unittest.TestCase):
    def test_ask_returns_the_last_message_and_sends_both_prompts(self):
        with patch("src.agent.text_model.subprocess.run", side_effect=fake_codex("the reply")) as run:
            reply = CodexCLIModel().ask(OTHER_PROMPT, "hello")
        message = run.call_args.kwargs["input"]

        self.assertEqual(reply, "the reply")
        self.assertLess(message.index(OTHER_PROMPT), message.index("hello"))

    def test_command_is_read_only_and_ignores_the_user_config(self):
        command = CodexCLIModel(model="gpt-6-luna")._command("/tmp/empty", Path("/tmp/empty/reply.txt"))

        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertEqual(command[command.index("--model") + 1], "gpt-6-luna")
        self.assertIn("--ignore-user-config", command)

    def test_failure_and_usage_limit_raise_llm_errors(self):
        cases = [
            (fake_codex(None, returncode=1, stderr="model_not_found"), LLMUnavailableError),
            (fake_codex(None, returncode=1, stderr="You've hit your usage limit"), LLMQuotaExhaustedError),
            (fake_codex("   "), LLMUnavailableError),
        ]
        for fake, error in cases:
            with self.subTest(error=error.__name__):
                with patch("src.agent.text_model.subprocess.run", side_effect=fake):
                    with self.assertRaises(error):
                        CodexCLIModel().ask(OTHER_PROMPT, "hello")


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

    def test_model_details_name_the_model_and_any_reasoning_effort(self):
        self.assertEqual(model_details(CodexCLIModel(model="gpt-6-luna")),
                         {"model_name": "gpt-6-luna", "reasoning_effort": "medium"})
        self.assertEqual(model_details(ClaudeCLIPolicy(model="claude-sonnet-5-5")),
                         {"model_name": "claude-sonnet-5-5", "reasoning_effort": None})

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            create_text_model("gpt")


if __name__ == "__main__":
    unittest.main()
