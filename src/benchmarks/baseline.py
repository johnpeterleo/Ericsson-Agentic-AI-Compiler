"""Collect vanilla-XLA baseline measurements for the benchmark programs.

Run from the repository root:

    python -m src.benchmarks.baseline                      # all programs, default size
    python -m src.benchmarks.baseline --size small         # quick run on CPU
    python -m src.benchmarks.baseline --programs softmax attention --repeats 100
    python -m src.benchmarks.baseline --include-optional   # also transformer_block

For each program this
1. checks the JAX port against its float64 NumPy reference on the "small"
   size for several seeds (catches porting mistakes before the agent sees them),
2. compiles the reference once and saves its StableHLO and optimized HLO,
3. profiles the compiled reference (median, spread, compile time) and records
   XLA's cost and memory analysis (FLOPs, bytes accessed, buffer sizes).

Results go to ``artifacts/baselines/<backend>_<device>_<size>_<timestamp>/``:
``summary.json``, ``summary.csv`` and one folder per program with HLO text.
Baselines are hardware specific: only compare agent results to a baseline
measured on the same device.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

from src.benchmarks.registry import REPO_ROOT, BenchmarkProgram, discover_programs, get_program
from src.compiler.compile import compile_program
from src.profiling.profiler import profile_function

DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "baselines"


@dataclass
class PortCheck:
    passed: bool
    seeds: list[int]
    max_abs_error: float
    message: str = ""


@dataclass
class BaselineRecord:
    program: str
    level: int
    characteristic: str
    size: str
    size_params: dict[str, Any]
    seed: int
    input_shapes: list[list[int]]
    input_dtypes: list[str]
    output_shape: list[int]
    output_dtype: str
    repeats: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    cv_percent: float
    compile_ms: float
    flops: float | None
    bytes_accessed: float | None
    gflops_per_s: float | None
    gbytes_per_s: float | None
    arithmetic_intensity: float | None
    argument_bytes: int | None
    output_bytes: int | None
    temp_bytes: int | None
    port_check: dict[str, Any]
    stablehlo_file: str | None = None
    optimized_hlo_file: str | None = None
    notes: list[str] = field(default_factory=list)


def check_port(program: BenchmarkProgram, seeds=(0, 1, 2), size: str = "small") -> PortCheck:
    """Compare the JAX reference with the float64 NumPy reference."""
    worst = 0.0
    for seed in seeds:
        inputs = program.make_inputs(seed=seed, size=size)
        actual = np.asarray(jax.jit(program.reference)(*inputs))
        expected = program.numpy_reference(*inputs)
        if actual.shape != expected.shape:
            return PortCheck(False, list(seeds), float("nan"),
                             f"seed {seed}: shape {actual.shape} != numpy {expected.shape}")
        if not np.all(np.isfinite(actual)):
            return PortCheck(False, list(seeds), float("nan"), f"seed {seed}: non-finite JAX output")
        error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        worst = max(worst, error)
        if not np.allclose(actual, expected, rtol=program.rtol, atol=program.atol):
            return PortCheck(False, list(seeds), worst,
                             f"seed {seed}: JAX and NumPy differ beyond rtol={program.rtol}, atol={program.atol}")
    return PortCheck(True, list(seeds), worst)


def _cost_analysis(executable) -> dict[str, float]:
    try:
        cost = executable.cost_analysis()
    except Exception:
        return {}
    if isinstance(cost, (list, tuple)):  # older JAX returns one dict per device
        cost = cost[0] if cost else {}
    return dict(cost or {})


def _memory_analysis(executable) -> dict[str, int | None]:
    try:
        stats = executable.memory_analysis()
    except Exception:
        stats = None
    fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    return {name: (int(getattr(stats, name)) if stats is not None and hasattr(stats, name) else None)
            for name in fields}


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return float(numerator) / float(denominator)


def measure_baseline(
    program: BenchmarkProgram,
    *,
    size: str = "default",
    seed: int = 0,
    repeats: int = 50,
    port_check: PortCheck | None = None,
    hlo_dir: Path | None = None,
) -> BaselineRecord:
    inputs = program.make_inputs(seed=seed, size=size)
    device_inputs = jax.block_until_ready(jax.device_put(inputs))

    # Clear JAX's in-memory caches so compile time reflects a cold compile.
    jax.clear_caches()
    compile_start = time.perf_counter()
    compilation = compile_program(program.reference, device_inputs)
    compile_ms = (time.perf_counter() - compile_start) * 1000
    output = jax.block_until_ready(compilation.executable(*device_inputs))
    notes = []
    if not bool(np.all(np.isfinite(np.asarray(output)))):
        notes.append("reference output contains non-finite values")

    cost = _cost_analysis(compilation.executable)
    memory = _memory_analysis(compilation.executable)
    profile = profile_function(program.reference, device_inputs, repeats=repeats)

    flops = cost.get("flops")
    bytes_accessed = cost.get("bytes accessed")
    seconds = profile.median_ms / 1000.0

    stablehlo_file = optimized_hlo_file = None
    if hlo_dir is not None:
        hlo_dir.mkdir(parents=True, exist_ok=True)
        stablehlo_path = hlo_dir / "stablehlo.mlir"
        stablehlo_path.write_text(compilation.stablehlo, encoding="utf-8")
        stablehlo_file = str(stablehlo_path)
        if compilation.optimized_hlo:
            optimized_path = hlo_dir / "optimized_hlo.txt"
            optimized_path.write_text(compilation.optimized_hlo, encoding="utf-8")
            optimized_hlo_file = str(optimized_path)

    return BaselineRecord(
        program=program.name,
        level=program.level,
        characteristic=program.characteristic,
        size=size,
        size_params=dict(program.sizes[size]),
        seed=seed,
        input_shapes=[list(a.shape) for a in inputs],
        input_dtypes=[str(a.dtype) for a in inputs],
        output_shape=list(output.shape),
        output_dtype=str(output.dtype),
        repeats=repeats,
        median_ms=profile.median_ms,
        mean_ms=profile.mean_ms,
        min_ms=profile.min_ms,
        max_ms=profile.max_ms,
        stddev_ms=profile.stddev_ms,
        cv_percent=100.0 * profile.stddev_ms / profile.mean_ms if profile.mean_ms else 0.0,
        compile_ms=compile_ms,
        flops=flops,
        bytes_accessed=bytes_accessed,
        gflops_per_s=_ratio(flops, seconds * 1e9) if flops is not None else None,
        gbytes_per_s=_ratio(bytes_accessed, seconds * 1e9) if bytes_accessed is not None else None,
        arithmetic_intensity=_ratio(flops, bytes_accessed),
        argument_bytes=memory["argument_size_in_bytes"],
        output_bytes=memory["output_size_in_bytes"],
        temp_bytes=memory["temp_size_in_bytes"],
        port_check=asdict(port_check) if port_check else {},
        stablehlo_file=stablehlo_file,
        optimized_hlo_file=optimized_hlo_file,
        notes=notes,
    )


def environment_info() -> dict[str, Any]:
    device = jax.devices()[0]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "backend": jax.default_backend(),
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "device_count": jax.device_count(),
        "git_commit": commit,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


_CSV_COLUMNS = (
    "program", "level", "size", "median_ms", "stddev_ms", "cv_percent", "min_ms", "max_ms",
    "compile_ms", "gflops_per_s", "gbytes_per_s", "arithmetic_intensity", "temp_bytes",
    "port_check_passed",
)


def run_baselines(
    programs: list[BenchmarkProgram],
    *,
    size: str = "default",
    seed: int = 0,
    port_seeds=(0, 1, 2),
    repeats: int = 50,
    out_dir: Path = DEFAULT_OUT_DIR,
    skip_port_check: bool = False,
    verbose: bool = True,
) -> Path:
    env = environment_info()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_dir) / f"{env['backend']}_{_slug(env['device_kind'])}_{size}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records, failures = [], []
    for program in programs:
        if verbose:
            print(f"[{program.name}] ", end="", flush=True)
        port = None if skip_port_check else check_port(program, seeds=port_seeds)
        if port is not None and not port.passed:
            failures.append({"program": program.name, "stage": "port_check", "message": port.message})
            if verbose:
                print(f"PORT CHECK FAILED: {port.message}")
            continue
        try:
            record = measure_baseline(
                program, size=size, seed=seed, repeats=repeats,
                port_check=port, hlo_dir=run_dir / program.name,
            )
        except Exception as error:  # e.g. out of memory at "large"
            failures.append({"program": program.name, "stage": "measure",
                             "message": f"{type(error).__name__}: {error}"})
            if verbose:
                print(f"FAILED: {type(error).__name__}: {error}")
            continue
        records.append(record)
        if verbose:
            print(f"median {record.median_ms:.3f} ms (cv {record.cv_percent:.1f}%)")

    summary = {
        "environment": env,
        "settings": {"size": size, "seed": seed, "port_seeds": list(port_seeds), "repeats": repeats},
        "results": [asdict(r) for r in records],
        "failures": failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {key: getattr(record, key, None) for key in _CSV_COLUMNS}
            row["port_check_passed"] = record.port_check.get("passed") if record.port_check else None
            writer.writerow(row)
    if verbose:
        print(f"\nWrote {run_dir}")
    return run_dir


def load_baseline(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a ``summary.json`` (or its folder) and index results by program name."""
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {result["program"]: result for result in summary["results"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--programs", nargs="*", help="program names (default: all non-optional)")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--size", default="default", help="small | default | large")
    parser.add_argument("--seed", type=int, default=0, help="seed for the timed inputs")
    parser.add_argument("--port-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-port-check", action="store_true")
    args = parser.parse_args(argv)

    if args.programs:
        programs = [get_program(name) for name in args.programs]
    else:
        programs = list(discover_programs(include_optional=args.include_optional).values())
    print(f"JAX {jax.__version__} on {jax.default_backend()} ({jax.devices()[0].device_kind}), size={args.size}")
    run_baselines(
        programs, size=args.size, seed=args.seed, port_seeds=tuple(args.port_seeds),
        repeats=args.repeats, out_dir=args.out, skip_port_check=args.skip_port_check,
    )


if __name__ == "__main__":
    main()
