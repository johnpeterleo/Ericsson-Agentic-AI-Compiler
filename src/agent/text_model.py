"""The text interface the tool agent talks to, and the models behind it.

`GeminiPolicy` and `ClaudeCLIPolicy` both have an `ask` method, so they are
text models already; this module only names the interface and picks one.
The interface is stateless: the caller sends the whole conversation each time.
"""

from typing import Protocol

from src.agent.llm_policy import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEMINI_MODEL,
    ClaudeCLIPolicy,
    GeminiPolicy,
    load_project_env,
)

PROVIDERS = ("claude", "gemini")
DEFAULT_MODELS = {"claude": DEFAULT_CLAUDE_MODEL, "gemini": DEFAULT_GEMINI_MODEL}


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


def create_text_model(provider: str, model_name: str | None = None) -> TextModel:
    """Claude goes through the logged-in Claude Code CLI; Gemini needs GEMINI_API_KEY."""
    if provider == "claude":
        return ClaudeCLIPolicy(model=model_name or DEFAULT_CLAUDE_MODEL)
    if provider == "gemini":
        load_project_env()
        return GeminiPolicy(model=model_name)
    raise ValueError(f"Unknown provider {provider!r}; choose from {PROVIDERS}.")
