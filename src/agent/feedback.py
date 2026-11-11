"""Short descriptions of a compiled program that fit in a prompt.

The optimized HLO of a transformer block is tens of thousands of characters,
most of it the bodies of fused computations and source locations. The model
gets the operations XLA actually launches instead: the entry computation and
any loop or branch bodies it calls, counted by operation and compared with the
reference, plus XLA's own FLOP and byte estimates.
"""

import re
from collections import Counter
from dataclasses import dataclass

import jax

from src.benchmarks.baseline import _cost_analysis
from src.compiler.compile import compile_program

MAX_HLO_CHARS = 3000
# These cost nothing at run time and would only crowd the counts.
IGNORED_OPCODES = frozenset({"parameter", "constant", "get-tuple-element", "tuple", "bitcast"})
# Long attributes that say nothing about what is computed.
DROPPED_ATTRIBUTES = ("metadata", "backend_config", "frontend_attributes")

_COMPUTATION_HEADER = re.compile(r"^(ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$")
_OPCODE = re.compile(r"=\s.*?\b([a-z][a-z0-9\-]*)\(")
_CUSTOM_CALL_TARGET = re.compile(r'custom_call_target="([^"]+)"')
_CALLED_BODY = re.compile(r"\b(?:body|condition|true_computation|false_computation)=%?([\w.\-]+)")
_CALLED_FUNCTION = re.compile(r"\bcall\(.*\bto_apply=%?([\w.\-]+)")
_FUSION_KIND = re.compile(r"\bkind=(k\w+)")
_CALLED_BRANCHES = re.compile(r"\bbranch_computations=\{([^}]*)\}")
_FLOAT_DTYPE = re.compile(r"\b(bf16|f8\w*|f16|f32|f64)\[")


@dataclass(frozen=True)
class CompiledSummary:
    op_counts: dict[str, int]
    float_dtypes: tuple[str, ...]
    flops: float | None
    bytes_accessed: float | None
    excerpt: str


def _split_computations(hlo_text):
    """Map each computation name to its instruction lines, and name the entry computation."""
    computations, entry, current = {}, None, None
    for line in hlo_text.splitlines():
        header = _COMPUTATION_HEADER.match(line)
        if header:
            current = header.group(2)
            computations[current] = []
            if header.group(1):
                entry = current
        elif line.startswith("}"):
            current = None
        elif current is not None and line.strip():
            computations[current].append(line)
    return computations, entry


def _called_computations(lines):
    called = []
    for line in lines:
        called.extend(_CALLED_BODY.findall(line))
        called.extend(_CALLED_FUNCTION.findall(line))
        for branches in _CALLED_BRANCHES.findall(line):
            called.extend(name.strip().lstrip("%") for name in branches.split(","))
    return called


def _launched_computations(computations, entry):
    """The entry computation and the loop and branch bodies reachable from it, in order."""
    order, pending = [], [entry]
    while pending:
        name = pending.pop(0)
        if name in order or name not in computations:
            continue
        order.append(name)
        pending.extend(_called_computations(computations[name]))
    return order


def _opcode(line):
    match = _OPCODE.search(line)
    if match is None:
        return None
    opcode = match.group(1)
    detail = {"custom-call": _CUSTOM_CALL_TARGET, "fusion": _FUSION_KIND}.get(opcode)
    found = detail.search(line) if detail else None
    return f"{opcode}({found.group(1)})" if found else opcode


def _count_operations(lines):
    opcodes = (_opcode(line) for line in lines)
    return Counter(opcode for opcode in opcodes if opcode and opcode not in IGNORED_OPCODES)


def _drop_attribute(line, name):
    """Remove `, name={...}` from one instruction, including any nested braces."""
    start = line.find(f", {name}={{")
    if start == -1:
        return line
    depth, position = 0, line.index("{", start)
    for position in range(position, len(line)):
        depth += {"{": 1, "}": -1}.get(line[position], 0)
        if depth == 0:
            break
    return line[:start] + line[position + 1 :]


def _clean_line(line):
    for name in DROPPED_ATTRIBUTES:
        line = _drop_attribute(line, name)
    return line


def summarize_hlo(hlo_text: str, cost: dict) -> CompiledSummary:
    computations, entry = _split_computations(hlo_text)
    launched = _launched_computations(computations, entry)
    launched_lines = [line for name in launched for line in computations[name]]
    all_lines = [line for lines in computations.values() for line in lines]
    excerpt = "\n".join(
        text for name in launched
        for text in (f"{name}:", *(_clean_line(line) for line in computations[name]))
    )
    return CompiledSummary(
        op_counts=dict(_count_operations(launched_lines).most_common()),
        float_dtypes=tuple(sorted({d for line in all_lines for d in _FLOAT_DTYPE.findall(line)})),
        flops=cost.get("flops"),
        bytes_accessed=cost.get("bytes accessed"),
        excerpt=excerpt[:MAX_HLO_CHARS],
    )


def summarize_compiled(function, inputs) -> CompiledSummary:
    """Compile `function` for `inputs` and summarize XLA's optimized program."""
    device_inputs = jax.block_until_ready(jax.device_put(inputs))
    compilation = compile_program(function, device_inputs)
    return summarize_hlo(compilation.optimized_hlo or "", _cost_analysis(compilation.executable))


def format_op_counts(op_counts: dict[str, int]) -> str:
    return ", ".join(f"{opcode}: {count}" for opcode, count in op_counts.items()) or "none"


def format_op_diff(reference: CompiledSummary, candidate: CompiledSummary) -> str:
    """Operations whose count changed, e.g. `dot 3 -> 1, while 1 -> 0`."""
    opcodes = list(reference.op_counts) + [
        opcode for opcode in candidate.op_counts if opcode not in reference.op_counts
    ]
    changes = [
        f"{opcode} {reference.op_counts.get(opcode, 0)} -> {candidate.op_counts.get(opcode, 0)}"
        for opcode in opcodes
        if reference.op_counts.get(opcode, 0) != candidate.op_counts.get(opcode, 0)
    ]
    return ", ".join(changes) or "same operation counts as the reference"


def _format_cost(summary: CompiledSummary) -> str:
    if summary.flops is None and summary.bytes_accessed is None:
        return "XLA cost analysis: unavailable"
    parts = []
    if summary.flops is not None:
        parts.append(f"{summary.flops:.3g} FLOPs")
    if summary.bytes_accessed is not None:
        parts.append(f"{summary.bytes_accessed:.3g} bytes accessed")
    return "XLA cost analysis: " + ", ".join(parts)


def format_summary(summary: CompiledSummary) -> str:
    return "\n".join([
        f"launched operations: {format_op_counts(summary.op_counts)}",
        _format_cost(summary),
        f"floating-point types in the compiled program: {', '.join(summary.float_dtypes) or 'none'}",
        f"optimized HLO (launched computations, first {MAX_HLO_CHARS} characters):",
        summary.excerpt,
    ])
