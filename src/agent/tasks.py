"""One task format for both program sets, so the agent does not care where a program came from.

Suite programs (`src.benchmarks.programs`) are named `suite:<name>`, programs
with planted inefficiencies (`src.benchmarks.planted`) `planted:<name>`, and
programs in `data/programs/` (`src.benchmarks.registry`) use their plain name.
Both the suite and `data/programs/` have an `attention`, so the prefixes keep
the names apart.
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.benchmarks import planted
from src.benchmarks import programs as suite
from src.benchmarks import registry

SUITE_PREFIX = "suite:"
PLANTED_PREFIX = "planted:"
SIZES = ("small", "default")
INPUT_SEED = 2026
HOLDOUT_SEED = 2027
# The final re-measurement uses a third input case that the search never saw.
REMEASURE_SEED = 2028
# Suite and planted programs are checked with these.
SUITE_RTOL = 1e-5
SUITE_ATOL = 1e-6
# Only used to build inputs and check the port; the model must not see them.
HIDDEN_FUNCTIONS = ("numpy_reference", "make_inputs")


@dataclass(frozen=True)
class Task:
    """`inputs` are what the model's tools run on; the other cases are for the evaluator."""

    name: str
    reference: Callable
    source_text: str
    inputs: tuple
    holdout_inputs: tuple
    remeasure_inputs: tuple
    rtol: float
    atol: float
    expert: Callable | None = None

    def input_summary(self) -> str:
        return ", ".join(f"{array.dtype}{list(array.shape)}" for array in self.inputs)


def _suite_size(program: suite.BenchmarkProgram, size: str) -> int:
    return program.test_size if size == "small" else program.benchmark_size


def task_from_suite(program: suite.BenchmarkProgram, prefix: str, size: str) -> Task:
    """A program with an expert, from the suite or the planted set; `prefix` names the set."""

    def inputs_for(seed):
        return program.make_inputs(np.random.default_rng(seed), _suite_size(program, size))

    return Task(
        name=prefix + program.name,
        reference=program.reference,
        source_text=inspect.getsource(program.reference),
        inputs=inputs_for(INPUT_SEED),
        holdout_inputs=inputs_for(HOLDOUT_SEED),
        remeasure_inputs=inputs_for(REMEASURE_SEED),
        rtol=SUITE_RTOL,
        atol=SUITE_ATOL,
        expert=program.expert,
    )


def visible_source(program_text: str) -> str:
    """The program file without the functions that build inputs or check the port."""
    lines = program_text.splitlines(keepends=True)
    hidden = [
        node for node in ast.parse(program_text).body
        if isinstance(node, ast.FunctionDef) and node.name in HIDDEN_FUNCTIONS
    ]
    for node in sorted(hidden, key=lambda node: node.lineno, reverse=True):
        first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
        del lines[first_line - 1 : node.end_lineno]
    return "".join(lines).rstrip() + "\n"


def task_from_registry(program: registry.BenchmarkProgram, size: str) -> Task:
    def inputs_for(seed):
        return program.make_inputs(seed=seed, size=size)

    return Task(
        name=program.name,
        reference=program.reference,
        source_text=visible_source(program.reference_source()),
        inputs=inputs_for(INPUT_SEED),
        holdout_inputs=inputs_for(HOLDOUT_SEED),
        remeasure_inputs=inputs_for(REMEASURE_SEED),
        rtol=program.rtol,
        atol=program.atol,
    )


_PROGRAM_SETS = ((SUITE_PREFIX, suite.PROGRAMS_BY_NAME), (PLANTED_PREFIX, planted.PLANTED_BY_NAME))


def load_task(name: str, size: str) -> Task:
    if size not in SIZES:
        raise ValueError(f"Unknown size {size!r}; choose from {SIZES}.")
    for prefix, programs in _PROGRAM_SETS:
        if name.startswith(prefix):
            return task_from_suite(programs[name.removeprefix(prefix)], prefix, size)
    return task_from_registry(registry.get_program(name), size)


def task_names() -> list[str]:
    names = list(registry.discover_programs(include_optional=True))
    for prefix, programs in _PROGRAM_SETS:
        names.extend(prefix + name for name in programs)
    return names
