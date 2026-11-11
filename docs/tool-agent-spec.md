# Spec: a tool-using optimization agent

Written 2026-11-11 for the implementer (Claude Opus 5.5). Owner: Joel Maharena.
Read this whole file before writing code.

## 1. Why this work exists

This is a KTH course project (DD2430, with Ericsson): an LLM agent that rewrites
JAX programs so they run faster on an NVIDIA L4 GPU than what the XLA compiler
produces by itself. A draft report is due 2026-11-20.

Today's agent (`src/agent/agent.py`, `OptimizationAgent`) is a fixed pipeline.
Each attempt sends one prompt (reference source, compiled form, timing, earlier
attempts) and gets back one candidate function, which the evaluator checks and
times. The script decides everything else: what the model sees, what is
measured, and when to stop.

Results so far (`results/2026-11-10-gpu/`, `results/2026-11-11-gpu/`): on the
11 suite programs it reached, gemini-3.6-flash found the expert's rewrite on
its **first** attempt every time. No later attempt was clearly better. So the
loop adds nothing on these programs; one LLM call is enough.

Joel's goal for this work: **a strong agent, not an LLM call**. The model
should decide what to inspect and measure, explore several ideas, learn from
precise feedback, remember what worked, and know when to stop.

The agent can only prove itself on programs that one call does not solve. The
suite in `src/benchmarks/programs.py` is too easy for that. The target
programs are the larger ones in `data/programs/` (`transformer_block`, `gpt`,
`fused_gemm_gelu_layernorm`, `attention`, `softmax`), where XLA is already
strong and no single known trick exists.

It is possible that the new agent does not beat a single call on those
programs either. That is a valid result for the report. Do not tune the agent
or the experiment until the numbers look good; report what was measured.

## 2. What to build

Five capabilities, built in this order. Each phase must leave all tests
passing.

1. **Tool loop.** Each turn the model picks one action; the harness runs it
   and returns an observation.
2. **Search.** The model can submit several candidates per turn; the harness
   keeps a leaderboard and re-measures the winners at the end.
3. **Richer feedback.** Failed candidates report how wrong they were; compiled
   forms are summarised and compared against the reference.
4. **Memory.** Lessons from finished runs are stored and shown on later runs.
5. **Experiment runner.** Compares the new agent against the old one at an
   equal number of LLM calls.

### 2.1 Do not change

- `OptimizationAgent`, `GeminiPolicy.propose`, `format_context_for_llm` and
  `SYSTEM_PROMPT` keep their behaviour. The old agent is the baseline, and its
  first prompt must stay byte-identical so the November results stay
  comparable. Build the new agent in new files.
- `evaluate()` keeps its signature and its meaning. You may add detail to
  failure messages (section 5.3), but keep their current opening words
  (`Shape mismatch:`, `Dtype mismatch:`, `Nonfinite values`,
  `Numerical mismatch:`, `On the held-out input:`), because tests and saved
  results rely on them.
- `src/agent/agent.py` has CRLF line endings. Avoid editing it. If you must,
  keep CRLF.

## 3. Existing code to reuse

| Need | Use |
|---|---|
| Correctness + paired timing + held-out input + compile limit | `src/optimization/evaluator.py`: `evaluate`, `measure_latency_pair` |
| Compile and get StableHLO / optimized HLO | `src/compiler/compile.py`: `compile_program` |
| Kill a compile that takes too long | `src/compiler/compile_time_limit.py`: `check_compile_time` |
| Parse a JSON reply, load `candidate` from source | `src/agent/llm_policy.py`: `_extract_json_object`, `_load_candidate_from_source` |
| LLM errors (`LLMUnavailableError`, `LLMQuotaExhaustedError`) and retries | `src/agent/llm_policy.py` |
| FLOPs / bytes from XLA's cost analysis | `src/benchmarks/baseline.py`: `_cost_analysis`, `_memory_analysis` |
| Programs | `src/benchmarks/programs.py` (suite, has experts) and `src/benchmarks/registry.py` (`data/programs/`, no experts) |

Note that both `programs.py` and `registry.py` define a class named
`BenchmarkProgram`. They are different classes.

## 4. Design

### 4.1 Task

Add one small dataclass that both program sets convert into, so the agent does
not care where a program came from:

```python
@dataclass(frozen=True)
class Task:
    name: str
    reference: Callable
    source_text: str          # what the model is shown
    inputs: tuple             # the model's tools use these
    holdout_inputs: tuple     # only the evaluator uses these
    rtol: float
    atol: float
    expert: Callable | None = None   # only for reporting, never shown
```

- Suite programs: `source_text` is `inspect.getsource(reference)`; inputs come
  from `make_inputs(rng, size)` with two different seeds.
- Registry programs: `reference` uses module-level helpers and constants
  (`_layer_norm`, `EPS`, `HEAD_DIM`), so `source_text` must be the program
  file without `numpy_reference` and `make_inputs`. Inputs come from
  `make_inputs(seed, size)` with two seeds. Use the program's own `RTOL` and
  `ATOL`.

Candidates are loaded into a namespace that holds only `jax` and `jnp`, so a
candidate must define every helper it uses. Say this in the system prompt.

### 4.2 Text model

The new agent needs a lower-level interface than `propose(context)`:

```python
class TextModel(Protocol):
    def ask(self, system_prompt: str, user_text: str) -> str: ...
```

Provide one adapter for Gemini and one for the Claude CLI. Both existing
policies hard-code `SYSTEM_PROMPT`; add a way to pass a different system
prompt without changing what the existing `propose` methods send. Keep the
existing retry and quota handling.

The interface is stateless. The agent keeps the transcript and sends all of it
on every turn. This is simple and works with both models. Cap the size of each
observation (section 5.4) so the transcript stays manageable.

### 4.3 Actions

The model replies with one JSON object per turn:

```json
{"thought": "...", "action": "<name>", ...action fields...}
```

| Action | Fields | What the harness does | Counts as |
|---|---|---|---|
| `inspect` | `code` or `id` | Compiles it and returns the compiled-form summary (5.4) plus a diff against the reference | 1 turn |
| `time` | `code` defining `probe(*inputs)` | Times any function of the task's inputs, with no correctness check. Lets the model measure a part of the program, to find where the time goes | 1 turn |
| `evaluate` | `candidates`: 1 to 4 of `{"idea", "code"}` | Runs the full evaluator on each (both inputs, paired timing, compile limit). Each gets an id (`c1`, `c2`, ...) | 1 turn, N evaluations |
| `finish` | `summary` | Ends the run | 1 turn |

Rules:
- `id` refers to an earlier candidate, or `reference`.
- A reply that is not valid JSON, names an unknown action, or lacks a field
  gets an observation that says what was wrong. It still uses up a turn.
- `time` exists because we have no per-operation GPU profile: `kernel_stats`
  in `ProfileReport` is always empty. Timing sub-computations is the honest
  substitute. Do not pretend to offer a per-kernel profiler.
- Keep the action set this small. Add an action only if a test run shows the
  model needs it.

### 4.4 Loop, budget and stopping

```
observation = opening message (task, source, reference summary, memory, budget)
repeat:
    reply = model.ask(system_prompt, transcript)
    action = parse(reply)
    observation = run(action)          # every observation ends with the
                                       # leaderboard and the remaining budget
until finish, or a budget is used up, or patience runs out
final re-measurement (4.5)
```

Named constants, all overridable from the command line:

| Constant | Default | Meaning |
|---|---|---|
| `MAX_TURNS` | 12 | LLM calls per task |
| `MAX_EVALUATIONS` | 16 | Candidates evaluated per task |
| `PATIENCE` | 2 | `evaluate` turns in a row without improvement before the harness stops the run |
| `IMPROVEMENT_MARGIN` | 1.05 | A candidate only counts as better if its speedup exceeds the best by this factor |

The margin exists because timing is noisy: on the L4, two identical programs
measured between 0.99x and 1.07x of each other. On a laptop CPU the spread is
far larger (0.8x to 1.3x), so CPU runs are for checking that the code works,
not for conclusions.

### 4.5 Search and the final re-measurement

- The harness stores every evaluated candidate: id, idea, source, result.
- Every observation ends with a leaderboard: the top 3 correct candidates (id,
  idea, speedup), and the full source of the best one.
- The system prompt tells the model to start with several clearly different
  ideas, then build on the best ones, and not to resubmit rejected code.

**Final re-measurement (required).** Picking the maximum of many noisy
measurements overstates the winner. When the loop ends, take the top 3 correct
candidates and evaluate each again with fresh paired timing and more repeats
(default 50), on a third input case built from a new seed. The speedup the run
reports is the re-measured one. Store both numbers.

### 4.6 Memory

- File: `data/rewrite_memory.jsonl`, tracked in git so the team shares it.
- After a run whose re-measured best speedup is at least 1.2x, make one extra
  LLM call asking for a lesson. Also store lessons about ideas that were
  correct but not faster; "this did not help on the L4" is useful.
- Entry fields: `rule` (one or two general sentences), `when` (what to look
  for in the source or compiled form), `evidence` (task, device, re-measured
  speedup, date), `kind` (`worked` or `did_not_help`).
- At the start of a run, show the stored rules in the opening message. Version
  one shows all of them, newest first, capped at 20. No embeddings.
- Modes: `off`, `read`, `readwrite`. Experiments that measure the agent itself
  run with `off`. The memory experiment (6.2) must never show a task a lesson
  that came from the same task.

## 5. Feedback details

### 5.1 Opening message

Task name, target device, input shapes and dtypes, tolerances, `source_text`,
the reference's timing and compiled-form summary, stored rules (if memory is
on), the action format, and the budget.

### 5.2 Timing feedback

For each evaluated candidate: reference ms, candidate ms, speedup, and whether
it beat the best by the margin.

### 5.3 Correctness feedback

When values differ, report more than pass/fail:
- the largest error as a multiple of the allowed error
  (`abs(candidate - reference) / (atol + rtol * abs(reference))`),
- the share of elements outside the tolerance,
- the index of the worst element with both values,
- which input case failed (first or held-out).

A candidate at 1.3 times the allowed error needs a different fix than one at
10,000 times; the model should be able to tell.

### 5.4 Compiled-form summary

The optimized HLO of `transformer_block` is far too long to paste. Return:
- an operation count (`dot: 3, fusion: 12, while: 1, ...`),
- the difference against the reference (`dot 3 -> 1, while 1 -> 0`),
- FLOPs and bytes accessed from XLA's cost analysis, when available,
- the first `MAX_HLO_CHARS` characters of the text.

## 6. Experiments

### 6.1 Agent against baselines (the main result)

Runner: `experiments/compare_agents.py`. For each task, at the same number of
LLM calls N (default 12):

| Arm | What it is |
|---|---|
| `single` | One call to the existing agent |
| `best_of_n` | N independent single calls; the top 3 by speedup are re-measured |
| `chain` | The existing agent with N attempts |
| `tool_agent` | The new agent with `MAX_TURNS = N`, memory off |

`best_of_n` is the baseline that matters. If the tool agent does not beat N
independent samples, its tools and search are not helping.

Every arm's reported speedup comes from the same final re-measurement (4.5).
Record per task and arm: re-measured speedup, LLM calls used, evaluations
used, wall time, and the full transcript. LLM output varies, so support
`--repetitions` and report every repetition, not only the best.

Tasks: `softmax`, `fused_gemm_gelu_layernorm`, `attention`,
`transformer_block`, `gpt` from `data/programs/`, plus two suite programs as a
sanity check that the new agent still solves the easy ones.

### 6.2 Memory

Run a sequence of related tasks twice, with memory `off` and `readwrite`, and
compare LLM calls until the best candidate was found. Lower priority than 6.1.

### 6.3 Where runs happen

- Develop and test on the Mac CPU with the `small` sizes.
- GPU runs happen on the L4 VM and cost money. **Do not start, stop or create
  any VM, and do not run `gcloud`.** Finish with exact commands Joel can run
  on the VM. GPU runs need `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- For LLM calls during development use the Claude CLI adapter (it uses Joel's
  subscription). The Gemini free tier allows about 20 requests per day, far
  too few for this agent. Keep real LLM calls during development modest: a
  few full runs on `small` sizes, not sweeps.

## 7. Constraints

- **Readable, conservative code.** Joel reads everything. Small functions that
  do one thing, at most three positional parameters (group the rest in a
  dataclass), no boolean flag parameters, named constants in place of bare
  numbers, no dead code, comments only where the code cannot say it. Match the
  style of `experiments/run_suite.py` and `src/compiler/compile_time_limit.py`.
- **Tests.** `python -m unittest discover -s tests` must pass and stay fast.
  Tests are offline: use a fake `TextModel` that returns scripted replies (see
  `FakeGemini` in `tests/test_agent_loop.py`). Cover at least: each action's
  observation, an invalid reply, each stopping condition, the leaderboard, the
  final re-measurement choosing by the re-measured number, memory modes, and
  both task loaders. Turn the compile time limit off in tests that do not test
  it, because each check starts a process.
- **Secrets.** Never read, print or copy `.env`.
- **Shared folder.** Other sessions edit this repository. Work on your own
  branch. Before staging a file, read its diff and make sure every change is
  yours. Do not push and do not touch `main`.
- **Model-written code is executed in this process** (`exec`), as the existing
  agent already does. Keep the candidate namespace to `jax` and `jnp`.
- **Precision.** Registry programs use `rtol = atol = 1e-2`, loose enough that
  computing in lower precision could pass. The system prompt must say that
  candidates keep float32 arithmetic. This cannot be enforced; record the
  dtypes found in each winning candidate's compiled form so it can be checked.

## 8. Suggested files

```
src/agent/tasks.py            Task, loaders for both program sets
src/agent/text_model.py       TextModel, Gemini and Claude adapters
src/agent/actions.py          parse a reply into an action; run it; build the observation
src/agent/feedback.py         error statistics, compiled-form summary and diff
src/agent/tool_agent.py       the loop, budget, leaderboard, final re-measurement
src/agent/memory.py           read and write data/rewrite_memory.jsonl
src/agent/tool_prompts.py     the new system prompt
experiments/run_tool_agent.py   one task, one run
experiments/compare_agents.py   section 6.1
tests/test_tasks.py, test_actions.py, test_feedback.py, test_tool_agent.py, test_memory.py
```

Adjust the layout if a better one appears; keep files small.

## 9. Done means

1. All five capabilities exist and are covered by offline tests; the full test
   suite passes.
2. `python -m experiments.run_tool_agent --task transformer_block --size small --model claude`
   completes on the Mac CPU and saves a transcript.
3. `python -m experiments.compare_agents` runs all four arms on at least two
   tasks at `small` size and writes one results file with the fields in 6.1.
4. The README has a short section: what the tool agent is, the actions, the
   two commands, and where results go.
5. A final message to Joel that states: what was built, what the CPU smoke
   runs showed (as checks that it works, not as findings), anything in this
   spec you changed and why, known gaps, and the exact commands for the GPU
   runs.

## 10. Where judgment is needed

- **The system prompt.** It decides how the model uses its turns. Describe the
  situation and the goal; explain why to inspect or time before rewriting a
  large program, and why to try different ideas first. Avoid long rule lists.
- **Observation size.** Too little and the model guesses; too much and the
  transcript fills with HLO. Start small and check real transcripts.
- **If something in this spec turns out wrong** once you see real behaviour
  (for example an action the model never uses, or a budget that is clearly
  off), change it, keep it simple, and say so in the final message.
