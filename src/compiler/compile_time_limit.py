"""Give up on a compilation that takes too long.

XLA cannot be stopped once it starts compiling. On an NVIDIA L4 its autotuner
took over 15 minutes for one short program (pairwise distances of 1024 points),
so a single slow candidate could hang a whole agent run.

`check_compile_time` therefore compiles the program first in a child process,
which can be killed. The caller compiles it again in its own process only if
the child finished in time, so every checked program is compiled twice.
"""

import multiprocessing
import os

import jax
from jax import export

COMPILE_TIME_LIMIT_SECONDS = 60


class CompileTimeoutError(RuntimeError):
    """Compiling the program took longer than the time limit."""


def _compile_in_child(serialized_program):
    # Share the GPU with the parent process instead of reserving most of its memory.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        exported = export.deserialize(serialized_program)
        shapes = [jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in exported.in_avals]
        jax.jit(exported.call).lower(*shapes).compile()
    except Exception:
        # Only the time matters here. The caller's own compile reports the error.
        raise SystemExit(1)


def check_compile_time(program, inputs, limit_seconds=COMPILE_TIME_LIMIT_SECONDS):
    """Raise CompileTimeoutError if compiling `program` for `inputs` takes too long.

    Errors while tracing `program` are raised as they are. Errors while compiling
    it are not reported here; the caller's own compile raises them.
    """
    serialized_program = export.export(jax.jit(program))(*inputs).serialize()
    child = multiprocessing.get_context("spawn").Process(
        target=_compile_in_child, args=(serialized_program,)
    )
    child.start()
    child.join(limit_seconds)
    if child.is_alive():
        child.kill()
        child.join()
        raise CompileTimeoutError(f"Compilation took longer than {limit_seconds} s.")
