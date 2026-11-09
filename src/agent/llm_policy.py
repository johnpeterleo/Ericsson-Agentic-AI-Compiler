"""LLM optimization policy (Google Gemini free tier via AI Studio API key)."""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp

from src.agent.agent import DemoPolicy, OptimizationPolicy, format_step_summary
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.types import AgentContext, JaxFn, RewriteProposal

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
MAX_HLO_CHARS = 4000
REQUEST_TIMEOUT_MS = 120_000
RETRY_WAIT_SECONDS = (30, 60, 120, 240)
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_START = 500


class LLMUnavailableError(RuntimeError):
    """Gemini was still unavailable, slow, or rate-limited after every retry."""


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
        lines.append("previous_steps:")
        for step in context.history:
            lines.append(format_step_summary(step))
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
        model: str = DEFAULT_GEMINI_MODEL,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)."
            )
        self.model = model
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
        Permanent errors, such as a bad key or no credit, are raised immediately.
        """
        from google.genai.errors import APIError
        from httpx import TransportError

        client = self._client_lazy()
        final_attempt = None
        for wait_seconds in (*RETRY_WAIT_SECONDS, final_attempt):
            try:
                return self._generate(client, user_text)
            except (APIError, TransportError) as error:
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
        reference_source = None
        try:
            reference_source = inspect.getsource(context.reference_fn)
        except (OSError, TypeError):
            reference_source = "# source unavailable (built-in or lambda)"
        user_text = format_context_for_llm(context, reference_source=reference_source)
        response = self._generate_with_retries(user_text)
        raw_text = getattr(response, "text", None) or str(response)
        payload = _extract_json_object(raw_text)
        rationale = str(payload.get("rationale", ""))
        if payload.get("stop"):
            return RewriteProposal(rationale=rationale or "LLM requested stop.", stop=True)
        source = payload.get("candidate_source")
        if not source or not str(source).strip():
            return RewriteProposal(
                rationale=rationale or "LLM did not supply candidate_source.",
                stop=True,
            )
        try:
            candidate_fn = _load_candidate_from_source(str(source))
        except Exception as error:
            return RewriteProposal(
                rationale=f"Invalid candidate from LLM: {type(error).__name__}: {error}",
                stop=True,
            )
        return RewriteProposal(
            rationale=rationale,
            candidate_fn=candidate_fn,
            metadata={"candidate_source": str(source)},
        )


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
