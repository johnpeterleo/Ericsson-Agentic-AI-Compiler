# Ericsson-Agentic-AI-Compiler
## Project Description
Running large AI models is one of the dominant costs in the industry, so every percent of inference or training speedup translates directly into financial gain and larger models. This is why frontier labs invest heavily in model lowering: turning a high-level model definition into fast machine code for a specific accelerator. Today this pipeline mixes manual and automatic stages. At the top, human experts redesign algorithms with mathematical and hardware-aware tricks, such as FlashAttention. Below that, engineers hand-write and tune device kernels (in CUDA, Triton, or Pallas), which is a slow and expertise-heavy task. Compilers then lower the model’s computation graph automatically, performing operator fusion, layout assignment, and scheduling. Compiler is constrained by fixed rewrite rules: it can fuse and schedule operations, yet cannot restructure an algorithm, so on its own it never rediscovers e.g. the streaming-softmax reformulation behind FlashAttention. The kernel-writing stage is an active research area: agentic systems like Sakana AI CUDA Engineer and DeepMind’s AlphaEvolve discover kernels and low-level optimizations that beat expert-tuned baselines. The common thread is that a reasoning agent can read profiles, form hypotheses, rewrite code, and measure the result – the way a human performance engineer does – can explore optimizations that a rule-based compiler cannot.


In this project, an agentic system was built that supplements a compiler to make AI models faster. Given a model or algorithm written in JAX, the array-computation and program-transformation language used by leading AI labs, which runs the same code on different hardware. The agent works in an iterative loop around the XLA compiler: it inspects the lowered program and its runtime profile, proposes high-level rewrites of the JAX code, suggests and implements custom kernels, recompiles, and analyzes the resulting performance to decide the next step. Language models do not always know much about a given accelerator or about compiler behavior, so part of the task was to give the agent the right context and feedback signals: profiling data, compiler intermediate representations (HLO/StableHLO), and correctness checks.

### Project Goal
The goal was to build an agentic optimization loop around JAX/XLA that iteratively rewrites code, proposes kernels, and profiles the result; demonstrate on a set of models/algorithms that the agent produces a measurably faster program than vanilla XLA compilation while preserving correctness; gradually grow from simple optimizations to full model rewrites.



## Project Structure

```text
Ericsson-Agentic-AI-Compiler/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── agent/
│   ├── compiler/
│   ├── profiling/
│   └── optimization/
│
└── data/
    └── programs/
```

## Project Setup
### Setup environment
Using Python 3.14.4, setup the venv, activate and verify it
```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
which python
```
The Python executable should point to the .venv directory inside the project.

Then optionally, but recommended to automatically activate the environment when entering the repository, setup direnv using this command for macOS google for other OS
```bash
brew install direnv
```

Create the .envrc 
```bash
touch .envrc
```

Open .envrc and copy-paste this line into the file
```bash
source .venv/bin/activate
```

Then paste this command into terminal
```bash
direnv allow
```

### Install requirements
```bash
pip install -r requirements.txt
```

## Pipeline

### Compiler interface

`compile_program(program, inputs: tuple) -> CompilationResult` is the shared
compilation interface used by the evaluator and compiler demo. It accepts a
pure JAX function and a tuple of positional array inputs, lowers and compiles
once per call, and returns:

| Field | Contents |
| --- | --- |
| `executable` | A compiled callable specialized to the input shapes and dtypes |
| `stablehlo` | The lowered StableHLO representation as text |
| `optimized_hlo` | The compiled HLO representation as text, or `None` when unavailable |

The wrapper uses JAX's normal specialization and device behavior. Callers own
input placement, execution, synchronization, and benchmarking. Compilation and
unexpected inspection errors propagate unchanged. Compiler text is intended
for inspection, not executable serialization; there is no custom cache.

```python
import jax
import jax.numpy as jnp

from src.compiler.compile import compile_program, simple_program

inputs = jax.block_until_ready((
    jnp.ones((32, 32), dtype=jnp.float32),
    jnp.ones((32, 32), dtype=jnp.float32),
))
compilation = compile_program(simple_program, inputs)
result = jax.block_until_ready(compilation.executable(*inputs))
```

Run the compiler demo from the repository root:

```bash
python src/compiler/compile.py
```

It prints StableHLO, optimized HLO (or an availability message), the result
shape, and execution time after warm-up. Representation text and timings vary
with the backend and software versions.

### Evaluator

The evaluator compiles both reference and candidate through `compile_program`,
then compares them on one tuple of array inputs. It owns input placement,
correctness checks, execution timing, and failure classification; compiler
representations remain in `CompilationResult`, separate from evaluation metrics.
Both functions must be pure and return a single finite
floating-point array with the same shape and dtype. Values are compared using
`atol + rtol * abs(reference)`, with defaults of `rtol=1e-5` and `atol=1e-6`.
Tolerances must be finite, nonnegative scalars; `repeats` must be a positive integer.

Successful evaluations report median execution times in milliseconds and an
observed speedup. Compilation, input placement, and warm-up are excluded from
timing. Candidate failures report a `failure_stage` (`compile`, `execute`,
`correctness`, or `benchmark`) and a readable `message`, without timings or a
speedup. Invalid arguments and reference-function errors raise exceptions.

Run the demo and tests from the repository root with the environment activated:

```bash
python -m src.optimization.demo_evaluator
python -m unittest discover -s tests -v
```

The demo uses minimal inputs. A single evaluation does not establish general
correctness or a repeatable speedup, and peak memory is not measured. Use the
target GPU for GPU performance comparisons.

## Contact
John Christensen - johnchristensen@outlook.com


Joel Maharena - joel.maharena@gmail.com 


Lidya Nasser -        


Sara Strandberg -   

Ericsson Superviser: Alexander Kravberg - alexander.kravberg@ericsson.com
