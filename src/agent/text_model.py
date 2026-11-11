"""The text interface the tool agent talks to, and the models behind it.

`GeminiPolicy` and `ClaudeCLIPolicy` both have an `ask` method, so they are
text models already. `CodexCLIModel` adds OpenAI models through the Codex CLI.
The interface is stateless: the caller sends the whole conversation each time.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from src.agent.llm_policy import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    ClaudeCLIPolicy,
    GeminiPolicy,
    LLMQuotaExhaustedError,
    LLMUnavailableError,
    load_project_env,
)

PROVIDERS = ("claude", "codex", "gemini")
DEFAULT_CODEX_MODEL = "gpt-6-luna"
# The user's Codex default is "xhigh", which would make every turn slow and costly.
CODEX_REASONING_EFFORT = "medium"
CODEX_TIMEOUT_SECONDS = 300
MAX_ERROR_CHARS = 300
# Codex has no system prompt option and is an agent with a shell, so the
# system prompt goes first in the message and it is told to answer in text only.
CODEX_PREAMBLE = (
    "Answer with text only. Do not run commands, read files or use any tool; "
    "everything you need is in this message."
)


class TextModel(Protocol):
    def ask(self, system_prompt: str, user_text: str) -> str:
        ...


class RecordingModel:
    """Counts the calls to a text model and keeps every prompt and reply."""

    def __init__(self, model: TextModel):
        self.model = model
        self.exchanges = []

    @property
    def calls(self) -> int:
        return len(self.exchanges)

    def ask(self, system_prompt: str, user_text: str) -> str:
        reply = self.model.ask(system_prompt, user_text)
        self.exchanges.append({"prompt": user_text, "reply": reply})
        return reply


class CodexCLIModel:
    """Asks an OpenAI model through the logged-in Codex CLI (``codex exec``).

    It runs read-only in an empty temporary folder, without the user's Codex
    config or rules, so it cannot see or change this repository.
    """

    def __init__(self, model: str = DEFAULT_CODEX_MODEL, executable: str = "codex"):
        self.model = model
        self.executable = executable

    def _command(self, folder: str, reply_path: Path) -> list[str]:
        return [
            self.executable, "exec",
            "--model", self.model,
            "--config", f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
            "--sandbox", "read-only",
            "--cd", folder,
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-last-message", str(reply_path),
            "-",
        ]

    def ask(self, system_prompt: str, user_text: str) -> str:
        message = f"{CODEX_PREAMBLE}\n\n{system_prompt}\n\n{user_text}"
        with tempfile.TemporaryDirectory() as folder:
            reply_path = Path(folder) / "reply.txt"
            try:
                completed = subprocess.run(
                    self._command(folder, reply_path), input=message, capture_output=True,
                    text=True, timeout=CODEX_TIMEOUT_SECONDS, cwd=folder,
                )
            except subprocess.TimeoutExpired as error:
                raise LLMUnavailableError(f"Codex did not answer within {error.timeout:.0f} s") from error
            except FileNotFoundError as error:
                raise LLMUnavailableError(f"{self.executable!r} is not installed") from error
            reply = reply_path.read_text(encoding="utf-8") if reply_path.exists() else ""
        if completed.returncode != 0 or not reply.strip():
            details = (completed.stderr or completed.stdout)[-MAX_ERROR_CHARS:]
            if "limit" in details.lower():
                raise LLMQuotaExhaustedError(f"Codex usage limit reached: {details}")
            raise LLMUnavailableError(f"Codex CLI failed: {details}")
        return reply


def create_text_model(provider: str, model_name: str | None = None) -> TextModel:
    """Claude and Codex go through their logged-in CLIs; Gemini needs GEMINI_API_KEY."""
    if provider == "claude":
        return ClaudeCLIPolicy(model=model_name or DEFAULT_CLAUDE_MODEL)
    if provider == "codex":
        return CodexCLIModel(model=model_name or DEFAULT_CODEX_MODEL)
    if provider == "gemini":
        load_project_env()
        return GeminiPolicy(model=model_name)
    raise ValueError(f"Unknown provider {provider!r}; choose from {PROVIDERS}.")
