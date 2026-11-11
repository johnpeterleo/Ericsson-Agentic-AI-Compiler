"""Shared helpers for the tool agent tests: a scripted model and a tiny task."""

import json

import numpy as np

from src.agent.search import EvaluationSettings, RemeasureSettings
from src.agent.tasks import Task
from src.agent.tool_agent import ToolAgentConfig

DOUBLING = "def candidate(x):\n    return x + x"
WRONG = "def candidate(x):\n    return x * 3"


def reference(x):
    return x * 2


def tiny_task():
    def case(seed):
        return (np.random.default_rng(seed).standard_normal(8).astype(np.float32),)

    return Task(
        name="double",
        reference=reference,
        source_text="def reference(x):\n    return x * 2\n",
        inputs=case(0),
        holdout_inputs=case(1),
        remeasure_inputs=case(2),
        rtol=1e-5,
        atol=1e-6,
    )


def fast_config(**overrides):
    """One timing repeat and no compile time limit: each limit check starts a process."""
    settings = {
        "evaluation": EvaluationSettings(repeats=1, compile_time_limit=None),
        "remeasure": RemeasureSettings(repeats=1),
        **overrides,
    }
    return ToolAgentConfig(**settings)


def action(name, **fields):
    return json.dumps({"thought": "scripted", "action": name, **fields})


def evaluate_action(*codes):
    return action("evaluate", candidates=[{"idea": f"idea {n}", "code": c} for n, c in enumerate(codes)])


class ScriptedModel:
    """Returns the queued replies in order and keeps every prompt it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def ask(self, system_prompt, user_text):
        self.prompts.append(user_text)
        return self.replies.pop(0)
