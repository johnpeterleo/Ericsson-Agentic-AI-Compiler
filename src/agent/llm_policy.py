"""LLM optimization policies: Gemini, any OpenAI-compatible endpoint, or the Claude Code CLI.

All three send the same prompt (``build_user_text``) and are judged by the same
evaluator, so the choice only changes which model answers.

The Gemini free tier has two limits that matter here: the daily request quota is
counted *per model*, and an overloaded model answers 503/504 (which still count
against that quota). ``GeminiPolicy`` therefore walks a chain of models instead
of spending every retry, and every request, on one busy model.
"""

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

import httpx
import jax
import jax.numpy as jnp

from src.agent.agent import DemoPolicy, OptimizationPolicy
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.types import AgentContext, JaxFn, OptimizationStep, RewriteProposal

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
# The daily quota is per model, so the second model is a second quota; it is also
# a different serving pool, which is what a 503/504 is complaining about.
FALLBACK_GEMINI_MODELS = ("gemini-3.6-flash",)
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"

# Free OpenAI-compatible endpoints. None of these need a paid plan; each is one
# environment variable away, and each has its own quota that Gemini's does not share.
# The model id has no safe default (providers rename them often), so pass --model
# or list what the endpoint offers with: python -m src.agent.llm_policy groq
PROVIDER_PRESETS: dict[str, tuple[str, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
    "ollama": ("http://localhost:11434/v1", None),
}
MAX_HLO_CHARS = 4000
MAX_FAILURE_MESSAGE_CHARS = 1000
REQUEST_TIMEOUT_MS = 120_000
RETRY_WAIT_SECONDS = (30, 60, 120, 240)
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_START = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504


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


def _is_overloaded(error) -> bool:
    """503/504 come from Google's gateway, not from the prompt: try another model."""
    return getattr(error, "code", None) in (HTTP_SERVICE_UNAVAILABLE, HTTP_GATEWAY_TIMEOUT)


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
    """Calls Gemini (free tier with AI Studio key) to propose the next rewrite.

    ``model`` is tried first. When it says it is overloaded (503/504) or its daily
    quota is gone, the policy moves on to the next name in ``fallback_models``
    instead of retrying the same model: on the free tier the quota is per model,
    so each name is a fresh 20 requests per day.

    A model that has failed once in this run is not asked again, which keeps a
    busy endpoint from eating the requests that a working model could answer.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        fallback_models: Sequence[str] | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)."
            )
        first = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        fallbacks = (
            FALLBACK_GEMINI_MODELS if fallback_models is None else tuple(fallback_models)
        )
        # Names are deduplicated so a repeated model cannot be asked twice.
        self.models = tuple(dict.fromkeys(name for name in (first, *fallbacks) if name))
        self.model = self.models[0]
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
        """Ask each model in turn until one answers.

        A request that takes longer than ``REQUEST_TIMEOUT_MS`` counts as temporary,
        but a retry of the same model only pays off for 429 (a per-minute limit that
        clears) and for network errors. A 503/504 means that model's serving pool is
        overloaded, so the next model is tried instead of waiting on the same one.

        A model whose daily quota is gone, or which keeps failing, is removed from
        ``self.models`` for the rest of the run. ``LLMQuotaExhaustedError`` is only
        raised once every model is out of quota, because that is when no request can
        succeed until the quota resets.
        """
        from google.genai.errors import APIError
        from httpx import TransportError

        client = self._client_lazy()
        exhausted: list[str] = []
        last_error = None
        for name in list(self.models):
            reply, error = self._ask_one_model(client, name, user_text, APIError, TransportError)
            if error is None:
                self.model = name
                return reply
            last_error = error
            if _is_daily_quota_error(error):
                exhausted.append(name)
                print(f"Gemini daily quota is used up for {name}; trying the next model")
            elif not _is_temporary_error(error):
                raise error
            else:
                print(f"Gemini {_describe_error(error)} from {name}; trying the next model")
            self.models = tuple(other for other in self.models if other != name)

        tried = len(exhausted) + len(self.models)
        if exhausted and not self.models:
            raise LLMQuotaExhaustedError(
                f"Gemini daily quota is used up for every model tried ({', '.join(exhausted)})"
            ) from last_error
        raise LLMUnavailableError(
            f"Gemini was unavailable for every model tried ({tried}): "
            f"{_describe_error(last_error)}"
        ) from last_error

    def _ask_one_model(self, client, name: str, user_text: str, api_error, transport_error):
        """Ask one model, retrying only the errors that a retry can fix.

        Returns ``(reply, None)`` on success, or ``(None, error)`` with the last
        error so the caller can decide whether to try a different model.
        """
        for wait_seconds in (*RETRY_WAIT_SECONDS, None):
            try:
                return self._generate(client, name, user_text), None
            except (api_error, transport_error) as error:
                if _is_daily_quota_error(error) or not _is_temporary_error(error):
                    return None, error
                if _is_overloaded(error) or wait_seconds is None:
                    return None, error
                print(f"Gemini {_describe_error(error)}; retrying {name} in {wait_seconds} s")
                time.sleep(wait_seconds)

    def _generate(self, client, model: str, user_text: str):
        return client.models.generate_content(
            model=model,
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


class OpenAICompatiblePolicy:
    """Asks any OpenAI-compatible chat endpoint, using only ``httpx``.

    This exists because a single provider's free tier can be busy or used up for
    days. All of these offer a free tier and speak the same API, so one policy
    covers them:

    ============  ===============================  ===============================
    provider      base URL                         key in
    ============  ===============================  ===============================
    groq          https://api.groq.com/openai/v1   ``GROQ_API_KEY``
    openrouter    https://openrouter.ai/api/v1    ``OPENROUTER_API_KEY``
    github        https://models.github.ai/inference ``GITHUB_TOKEN``
    ollama        http://localhost:11434/v1       key not needed
    ============  ===============================  ===============================

    Pass ``provider="groq"`` to use a preset, or ``base_url`` for anything else.
    The model id is required because providers rename models often; list what an
    endpoint offers with ``python -m src.agent.llm_policy groq``.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = REQUEST_TIMEOUT_MS / 1000,
    ):
        if provider and provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"unknown provider {provider!r}; choose from: {', '.join(PROVIDER_PRESETS)}"
            )
        preset_url, preset_key = PROVIDER_PRESETS.get(provider, (None, None))
        self.base_url = (base_url or preset_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError(
                "Pass provider= (groq, openrouter, github, ollama) or an explicit base_url."
            )
        self.api_key = api_key or (os.environ.get(preset_key) if preset_key else None)
        if not self.api_key and provider != "ollama" and not self.base_url.startswith("http://localhost"):
            raise ValueError(
                f"{provider or self.base_url} needs an API key; set {preset_key} or pass api_key=."
            )
        self.model = model
        self.provider = provider or self.base_url
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "openrouter":
            # OpenRouter attributes free-tier usage with these; they are optional.
            headers["HTTP-Referer"] = "https://github.com/Ericsson-Agentic-AI-Compiler"
            headers["X-Title"] = "Ericsson Agentic AI Compiler"
        return headers

    def _post(self, user_text: str):
        """The raw HTTP response for one chat request."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
        }
        try:
            return httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TransportError as error:
            raise LLMUnavailableError(f"{self.provider} request failed: {error}") from error

    def _ask(self, user_text: str) -> str:
        """One chat request, retrying a per-minute limit like the Gemini policy does.

        A used-up daily quota is raised as ``LLMQuotaExhaustedError`` so the suite
        can stop cleanly; anything else is ``LLMUnavailableError`` and the suite
        moves on to the next program.
        """
        for wait_seconds in (*RETRY_WAIT_SECONDS, None):
            response = self._post(user_text)
            if response.status_code != HTTP_TOO_MANY_REQUESTS:
                break
            body = _response_message(response)
            if _is_daily_quota_error(SimpleApiError(response.status_code, body)):
                raise LLMQuotaExhaustedError(f"{self.provider} daily quota used up: {body[:300]}")
            if wait_seconds is None:
                raise LLMUnavailableError(f"{self.provider} rate limited: {body[:300]}")
            print(f"{self.provider} rate limited; retrying in {wait_seconds} s")
            time.sleep(wait_seconds)

        if response.status_code >= HTTP_SERVER_ERROR_START:
            raise LLMUnavailableError(
                f"{self.provider} status {response.status_code}: "
                f"{_response_message(response)[:300]}"
            )
        if response.status_code >= 400:
            raise LLMUnavailableError(
                f"{self.provider} rejected the request (status {response.status_code}): "
                f"{_response_message(response)[:300]}"
            )
        try:
            reply = response.json()
            return reply["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise LLMUnavailableError(
                f"{self.provider} reply was not a chat completion: {response.text[:300]}"
            ) from error

    def propose(self, context: AgentContext) -> RewriteProposal:
        return proposal_from_reply(self._ask(build_user_text(context)))

    def list_models(self) -> list[str]:
        """The model ids this endpoint offers, to fill in ``--model``."""
        response = httpx.get(
            f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout_seconds
        )
        response.raise_for_status()
        return sorted(item["id"] for item in response.json().get("data", []))


class SimpleApiError:
    """Just enough of an API error for ``_is_daily_quota_error`` to read its message."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message


def _response_message(response) -> str:
    """The readable part of an error body, whichever shape the endpoint uses."""
    try:
        body = response.json()
    except (json.JSONDecodeError, TypeError):
        return response.text
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        return str(body.get("message") or body)
    return str(body)


def create_default_policy(
    *,
    demo_candidates: Sequence[JaxFn] | None = None,
    prefer_llm: bool = True,
) -> OptimizationPolicy:
    """Use whichever free LLM is configured, or fall back to ``DemoPolicy``.

    Order: ``LLM_PROVIDER``+``LLM_MODEL`` (any OpenAI-compatible endpoint), then
    Gemini (which rotates models on its own), then the scripted ``DemoPolicy``.
    """
    load_project_env()
    if prefer_llm:
        provider_policy = _policy_from_environment()
        if provider_policy is not None:
            return provider_policy
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return GeminiPolicy()
    return DemoPolicy(candidates=demo_candidates)


def _policy_from_environment() -> OptimizationPolicy | None:
    """An ``OpenAICompatiblePolicy`` from ``LLM_PROVIDER``/``LLM_MODEL``, if both are set."""
    provider = os.environ.get("LLM_PROVIDER")
    model = os.environ.get("LLM_MODEL")
    if not (provider and model):
        return None
    return OpenAICompatiblePolicy(
        model=model,
        provider=provider if provider in PROVIDER_PRESETS else None,
        base_url=None if provider in PROVIDER_PRESETS else provider,
    )


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


def _command_line(argv: Sequence[str] | None = None) -> int:
    """Check a free provider without running a benchmark.

        python -m src.agent.llm_policy groq            # list that provider's models
        python -m src.agent.llm_policy groq MODEL      # send one real request
        python -m src.agent.llm_policy gemini          # show the model fallback chain
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", help="gemini, claude, or a key from PROVIDER_PRESETS")
    parser.add_argument("model", nargs="?", help="model id; omitted means list the models")
    arguments = parser.parse_args(argv)

    load_project_env()
    try:
        if arguments.provider == "gemini":
            policy = GeminiPolicy(model=arguments.model)
            print(f"Gemini models are tried in this order: {list(policy.models)}")
            if arguments.model is None:
                return 0
            reply = policy._generate_with_retries("Reply with the single word: ready")
        elif arguments.provider == "claude":
            policy = ClaudeCLIPolicy(model=arguments.model or DEFAULT_CLAUDE_MODEL)
            if arguments.model is None:
                print(f"Claude via the CLI; default model is {DEFAULT_CLAUDE_MODEL}")
                return 0
            reply = policy._ask("Reply with the single word: ready")
        elif arguments.model is None:
            policy = OpenAICompatiblePolicy(provider=arguments.provider, model="unused")
            for name in policy.list_models():
                print(name)
            return 0
        else:
            policy = OpenAICompatiblePolicy(provider=arguments.provider, model=arguments.model)
            reply = policy._ask("Reply with the single word: ready")
    except (LLMUnavailableError, ValueError) as error:
        print(f"failed: {error}")
        return 1
    print(f"ok from {policy.model}: {str(reply).strip()[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_command_line())


