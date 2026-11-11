"""LLM optimization policies: Gemini via an API key, or Claude via the Claude Code CLI."""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from src.agent.agent import DemoPolicy, OptimizationPolicy
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.types import AgentContext, JaxFn, OptimizationStep, RewriteProposal

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
MAX_HLO_CHARS = 4000
MAX_FAILURE_MESSAGE_CHARS = 1000
REQUEST_TIMEOUT_MS = 120_000
RETRY_WAIT_SECONDS = (30, 60, 120, 240)
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_START = 500


class LLMUnavailableError(RuntimeError):
    """Gemini was still unavailable, slow, or rate-limited after every retry."""


class LLMQuotaExhaustedError(LLMUnavailableError):
    """The daily request quota is used up, so no request can succeed until it resets."""


def _is_daily_quota_error(error) -> bool:
    """A used-up daily quota only resets the next day, so waiting minutes cannot help."""
    message = (getattr(error, "message", None) or "").lower()
    return "per day" in message or "perday" in message


def _is_temporary_error(error) -> bool:
    """Network problems and timeouts have no status code and are always temporary."""
    code = getattr(error, "code", None)
    if code is None:
        return True
    if code == HTTP_TOO_MANY_REQUESTS:
        return not _is_daily_quota_error(error)
    return code >= HTTP_SERVER_ERROR_START


def _describe_error(error) -> str:
    code = getattr(error, "code", None)
    return f"status {code}" if code is not None else type(error).__name__


def load_project_env() -> None:
    """Load ``GEMINI_API_KEY`` from repo-root ``.env`` (file is gitignored)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def _load_candidate_from_source(source: str) -> JaxFn:
    """Compile LLM-produced source into ``candidate(x, ...)``."""
    namespace: dict[str, Any] = {"jax": jax, "jnp": jnp}
    exec(source, namespace)  # noqa: S102 — project experiment; trust team prompts only
    candidate = namespace.get("candidate")
    if not callable(candidate):
        raise ValueError("candidate_source must define a callable named `candidate`.")
    return candidate


def _attempt_outcome(step: OptimizationStep) -> str:
    evaluation = step.evaluation
    if evaluation is None:
        return "not evaluated"
    if not evaluation.correct:
        message = (evaluation.message or "")[:MAX_FAILURE_MESSAGE_CHARS]
        return f"rejected at {evaluation.failure_stage}: {message}"
    return (
        f"correct, speedup {evaluation.speedup:.3f}x "
        f"(reference {evaluation.reference_ms:.3f} ms, candidate {evaluation.candidate_ms:.3f} ms)"
    )


def _format_attempt(step: OptimizationStep) -> list[str]:
    lines = [
        f"--- attempt {step.index}: {_attempt_outcome(step)}",
        f"rationale: {step.proposal.rationale}",
    ]
    source = step.proposal.metadata.get("candidate_source")
    if source:
        lines.extend(["candidate_source:", source.strip()])
    return lines


def _format_previous_attempts(context: AgentContext) -> list[str]:
    """Show the model what it already tried: the code, the result, and the best so far."""
    lines = ["previous_attempts (each was compiled, checked against the reference, and timed):"]
    for step in context.history:
        lines.extend(_format_attempt(step))

    best = context.best_step()
    if best is None:
        lines.append(
            "No attempt has been correct yet. Fix the reported problem or try a different rewrite."
        )
        return lines
    lines.append(f"best_so_far: attempt {best.index} ({best.evaluation.speedup:.3f}x)")
    if best.hlo and best.hlo.optimized_hlo:
        lines.extend(["best_so_far_optimized_hlo:", best.hlo.optimized_hlo[:MAX_HLO_CHARS]])
    lines.append(
        "Improve on the best attempt or try a different idea. Do not repeat a rejected attempt."
    )
    return lines


def format_context_for_llm(context: AgentContext, *, reference_source: str | None = None) -> str:
    lines = [
        f"backend={context.backend}",
        context.hardware_note,
        f"input_ranks={[getattr(a, 'shape', None) for a in context.inputs]}",
        f"baseline_profile={context.baseline_profile.summary()}",
        "baseline_stablehlo:",
        context.baseline_hlo.text[:MAX_HLO_CHARS],
    ]
    if context.baseline_hlo.optimized_hlo:
        lines.extend(["optimized_hlo:", context.baseline_hlo.optimized_hlo[:MAX_HLO_CHARS]])
    if reference_source:
        lines.extend(["reference_jax_source:", reference_source.strip()])
    if context.history:
        lines.extend(_format_previous_attempts(context))
    lines.append(
        'Respond with JSON only: {"stop": bool, "rationale": str, '
        '"candidate_source": str | null}. '
        "If stop is false, candidate_source must be Python defining "
        "`def candidate(...):` using jax.numpy as jnp, pure JAX only."
    )
    return "\n".join(lines)


class GeminiPolicy:
    """Calls Gemini (free tier with AI Studio key) to propose the next rewrite."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)."
            )
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise ImportError(
                    "Install the Gemini SDK: pip install google-genai"
                ) from error
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": REQUEST_TIMEOUT_MS},
            )
        return self._client

    def _generate_with_retries(self, user_text: str):
        """Call Gemini, waiting and retrying when the error is temporary.

        A request that takes longer than ``REQUEST_TIMEOUT_MS`` counts as temporary.
        Permanent errors, such as a bad key, are raised immediately. A used-up
        daily quota is raised as ``LLMQuotaExhaustedError`` so callers can stop cleanly.
        """
        from google.genai.errors import APIError
        from httpx import TransportError

        client = self._client_lazy()
        final_attempt = None
        for wait_seconds in (*RETRY_WAIT_SECONDS, final_attempt):
            try:
                return self._generate(client, user_text)
            except (APIError, TransportError) as error:
                if _is_daily_quota_error(error):
                    raise LLMQuotaExhaustedError(
                        f"Gemini daily quota is used up for {self.model}"
                    ) from error
                if not _is_temporary_error(error):
                    raise
                if wait_seconds is final_attempt:
                    raise LLMUnavailableError(
                        f"Gemini failed every retry ({_describe_error(error)})"
                    ) from error
                print(f"Gemini {_describe_error(error)}; retrying in {wait_seconds} s")
                time.sleep(wait_seconds)

    def _generate(self, client, user_text: str):
        return client.models.generate_content(
            model=self.model,
            contents=user_text,
            config={"system_instruction": SYSTEM_PROMPT},
        )

    def propose(self, context: AgentContext) -> RewriteProposal:
        """Ask Gemini for the next rewrite.

        A reply that cannot be used comes back with ``load_error`` set. The agent
        shows that error to Gemini on the next attempt instead of ending the run.
        """
        response = self._generate_with_retries(build_user_text(context))
        return proposal_from_reply(getattr(response, "text", None) or str(response))


def build_user_text(context: AgentContext) -> str:
    """The prompt every LLM policy sends, so all models see the same information."""
    try:
        reference_source = inspect.getsource(context.reference_fn)
    except (OSError, TypeError):
        reference_source = "# source unavailable (built-in or lambda)"
    return format_context_for_llm(context, reference_source=reference_source)


def proposal_from_reply(raw_text: str) -> RewriteProposal:
    """Turn a model's JSON reply into a proposal, or a ``load_error`` the agent can show it."""
    try:
        payload = _extract_json_object(raw_text)
    except ValueError as error:
        return RewriteProposal(
            rationale="", load_error=f"The reply was not the requested JSON object: {error}"
        )
    rationale = str(payload.get("rationale", ""))
    if payload.get("stop"):
        return RewriteProposal(rationale=rationale or "LLM requested stop.", stop=True)
    source = str(payload.get("candidate_source") or "")
    if not source.strip():
        return RewriteProposal(
            rationale=rationale, load_error="The reply had no candidate_source."
        )
    try:
        candidate_fn = _load_candidate_from_source(source)
    except Exception as error:
        return RewriteProposal(
            rationale=rationale,
            metadata={"candidate_source": source},
            load_error=f"{type(error).__name__}: {error}",
        )
    return RewriteProposal(
        rationale=rationale,
        candidate_fn=candidate_fn,
        metadata={"candidate_source": source},
    )


class ClaudeCLIPolicy:
    """Asks Claude for the next rewrite through the Claude Code CLI (``claude -p``).

    This uses the logged-in Claude subscription instead of an API key. Claude's
    tools are turned off, so it only answers with text: every candidate is still
    compiled, checked, and timed by this project's evaluator, exactly like Gemini's.
    """

    def __init__(self, *, model: str = DEFAULT_CLAUDE_MODEL, executable: str = "claude"):
        self.model = model
        self.executable = executable

    def _command(self) -> list[str]:
        return [
            self.executable,
            "--print",
            "--model", self.model,
            "--tools", "",
            "--system-prompt", SYSTEM_PROMPT,
            "--output-format", "json",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources", "",
        ]

    def _ask(self, user_text: str) -> str:
        """Send one prompt and return Claude's reply text."""
        try:
            # Run outside the repository so the CLI does not read its project files.
            completed = subprocess.run(
                self._command(),
                input=user_text,
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT_MS / 1000,
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as error:
            raise LLMUnavailableError(f"Claude did not answer within {error.timeout:.0f} s") from error
        except FileNotFoundError as error:
            raise LLMUnavailableError(f"{self.executable!r} is not installed") from error

        try:
            reply = json.loads(completed.stdout)
        except json.JSONDecodeError:
            reply = {"is_error": True, "result": completed.stderr or completed.stdout}
        message = str(reply.get("result") or "")
        if reply.get("is_error") or completed.returncode != 0:
            if "limit" in message.lower():
                raise LLMQuotaExhaustedError(f"Claude usage limit reached: {message[:300]}")
            raise LLMUnavailableError(f"Claude CLI failed: {message[:300]}")
        return message

    def propose(self, context: AgentContext) -> RewriteProposal:
        return proposal_from_reply(self._ask(build_user_text(context)))


def create_default_policy(
    *,
    demo_candidates: Sequence[JaxFn] | None = None,
    prefer_llm: bool = True,
) -> OptimizationPolicy:
    """Use Gemini when an API key is set; otherwise fall back to ``DemoPolicy``."""
    load_project_env()
    if prefer_llm and (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        return GeminiPolicy()
    return DemoPolicy(candidates=demo_candidates)
