"""Discover and load the benchmark programs in ``data/programs/``.

Every ``*.py`` file in ``data/programs/`` (except files starting with ``_``)
is one benchmark program. A program file must define:

``NAME``             short identifier, e.g. ``"softmax"``
``LEVEL``            KernelBench-style level: 1 single op, 2 fused ops, 3 model
``CHARACTERISTIC``   e.g. ``"compute-bound"`` or ``"memory-bound"``
``DESCRIPTION``      one sentence describing the computation
``SIZES``            mapping size name -> shape parameters; must contain
                     ``"small"`` (CPU/tests) and ``"default"`` (GPU runs)
``RTOL``, ``ATOL``   tolerances used when comparing candidates to ``reference``
``reference(*inputs)``        pure JAX function, the ground truth for the agent
``numpy_reference(*inputs)``  independent float64 NumPy version, used to check
                              that the JAX port computes the intended thing
``make_inputs(seed=0, size="default")``  deterministic tuple of NumPy arrays

Optional: ``OPTIONAL = True`` marks a program that is skipped unless asked for,
and ``SOURCE`` is free text naming the original (e.g. a KernelBench problem).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAMS_DIR = REPO_ROOT / "data" / "programs"

_REQUIRED = (
    "NAME", "LEVEL", "CHARACTERISTIC", "DESCRIPTION", "SIZES", "RTOL", "ATOL",
    "reference", "numpy_reference", "make_inputs",
)


@dataclass(frozen=True)
class BenchmarkProgram:
    name: str
    level: int
    characteristic: str
    description: str
    sizes: Mapping[str, Mapping[str, Any]]
    rtol: float
    atol: float
    reference: Callable[..., Any]
    numpy_reference: Callable[..., np.ndarray]
    make_inputs_fn: Callable[..., tuple]
    optional: bool = False
    source: str = ""
    path: Path | None = None
    module: ModuleType | None = field(default=None, repr=False, compare=False)

    def make_inputs(self, seed: int = 0, size: str = "default") -> tuple[np.ndarray, ...]:
        if size not in self.sizes:
            raise ValueError(
                f"Unknown size {size!r} for {self.name}; choose from {sorted(self.sizes)}."
            )
        inputs = tuple(self.make_inputs_fn(seed=seed, size=size))
        if not inputs or not all(isinstance(a, np.ndarray) for a in inputs):
            raise TypeError(f"{self.name}.make_inputs must return a non-empty tuple of NumPy arrays.")
        return inputs

    def reference_source(self) -> str:
        """Source of the program file, useful as context for the LLM."""
        return self.path.read_text(encoding="utf-8") if self.path else ""


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"benchmark_programs.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import benchmark program from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_program_file(path: str | Path) -> BenchmarkProgram:
    path = Path(path)
    module = _load_module(path)
    missing = [attr for attr in _REQUIRED if not hasattr(module, attr)]
    if missing:
        raise AttributeError(f"{path.name} is missing required attributes: {', '.join(missing)}")
    for size in ("small", "default"):
        if size not in module.SIZES:
            raise ValueError(f"{path.name}: SIZES must contain {size!r}.")
    return BenchmarkProgram(
        name=module.NAME,
        level=int(module.LEVEL),
        characteristic=module.CHARACTERISTIC,
        description=module.DESCRIPTION,
        sizes=module.SIZES,
        rtol=float(module.RTOL),
        atol=float(module.ATOL),
        reference=module.reference,
        numpy_reference=module.numpy_reference,
        make_inputs_fn=module.make_inputs,
        optional=bool(getattr(module, "OPTIONAL", False)),
        source=str(getattr(module, "SOURCE", "")),
        path=path,
        module=module,
    )


def discover_programs(
    directory: str | Path = PROGRAMS_DIR, *, include_optional: bool = False
) -> dict[str, BenchmarkProgram]:
    """Return programs keyed by name, ordered by level and then name."""
    programs = [
        load_program_file(path)
        for path in sorted(Path(directory).glob("*.py"))
        if not path.name.startswith("_")
    ]
    names = [p.name for p in programs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate program names: {sorted(duplicates)}")
    programs.sort(key=lambda p: (p.level, p.name))
    return {p.name: p for p in programs if include_optional or not p.optional}


def get_program(name: str, directory: str | Path = PROGRAMS_DIR) -> BenchmarkProgram:
    programs = discover_programs(directory, include_optional=True)
    if name not in programs:
        raise KeyError(f"Unknown program {name!r}; available: {', '.join(programs)}")
    return programs[name]
