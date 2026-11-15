# Claude Code with Fable 5.2 on attention (2026-11-15)

A general coding agent as a reference point for our tool agent: Claude Code
2.1.325 with `claude-fable-5-2`, given a shell on the VM and the same task.

**Setup** (`cc_setup.py`, `cc_job.sh`): a folder with the reference source,
a `candidate.py` to edit, and `check.py`, which runs the project's evaluator
(two input cases, 60 s compile limit, alternating timing). The prompt
(`PROMPT.md` in each run) uses the tool agent's goal and code rules word for
word. It was told to work only in its folder and not to search online;
web tools were disabled. Earlier results were moved into a root-only folder
during the run. Time limit 47 minutes. References at XLA's default matmul
precision.

**Scoring** (`cc_score.py`): the same as the comparison arms, the full check,
then a re-measurement with 50 repeats on an input it never saw.

## Clean run (run-1714-clean)

One run with the VM to itself, same setup: **2.11x** re-measured (2.11x in
`check.py`), no precision change. 12 turns, 13 minutes, 14,647 output tokens
(5,941 thinking), $1.14 at API prices. Audit: all 11 tool calls were shell
commands in its own folder or with the project's Python interpreter.

Its rewrite goes further than the query-block loop the other agents used: it
splits the causal triangle into unmasked squares of 1024, 512, 256 and 128
below the diagonal plus masked 128x128 diagonal tiles, one batched matmul per
size, so about 53% of the score matrix is computed. It also replaces the
row-max pass with an upper bound on each row's scores (softmax does not change
when every score in a row is shifted by the same amount), with an exact
fallback through `lax.cond` when that bound could cause underflow, and gets
the softmax denominator from a ones column appended to V.

| attention, default precision | Re-measured speedup |
|---|---|
| single call (gpt-6.1-sol x3, Fable x1) | 1.56x to 1.81x |
| tool agent, gpt-6.1-sol (3 runs) | 3.24x, 1.96x, 1.81x |
| tool agent, Fable (1 run) | 2.13x |
| Claude Code, Fable (clean run) | 2.11x |
| Claude Code, Fable (2 overlapping runs) | 1.25x, 1.36x |

## Repetitions (2026-11-15, `fable_batch.sh`, `log-fable-batch.txt`)

Run one after the other, nothing else on the VM. All with claude-fable-5-2,
references at default precision, the subscription login (`apiKeySource:
none` in every log). The single call and tool agent records are in
`../e4-claude-fable-5.2-batch.jsonl`.

| attention | first run | repetition 2 | repetition 3 |
|---|---|---|---|
| single call | 1.56x | 1.38x | 1.46x |
| tool agent | 2.13x | 2.08x | 2.12x |
| Claude Code | 2.11x | 2.07x (`cc-attention-1115-2107`) | 1.51x (`cc-attention-1115-2207`) |

| transformer_block | result |
|---|---|
| tool agent | 1.13x (8 calls) |
| Claude Code | 1.14x (`cc-transformer_block-1115-2313`) |

Claude Code runs:

| Folder | Re-measured | Turns | Time | Output tokens (thinking) | API-price equivalent |
|---|---|---|---|---|---|
| cc-attention-1115-2107 | 2.07x | 24 | 29 min | 25,424 (11,650) | $1.91 |
| cc-attention-1115-2207 | 1.51x | 22 | 37 min | 19,817 (9,018) | $1.62 |
| cc-transformer_block-1115-2313 | 1.14x | 14 | 19 min | 14,665 (6,124) | $1.24 |

The "API-price equivalent" is Claude Code's own estimate of what the tokens
would cost at API prices. Nothing was billed: the runs counted against the
subscription's usage limit.

In the 1.51x run Claude Code measured that 16 query blocks ran at about 1.9x
but compiled for 101 s, over the 60 s limit, and settled for 7 blocks.

Audit: 57 tool calls in the three runs, all shell commands in the run's own
folder or with the project's Python interpreter. No winning candidate in this
batch changes the precision.

## The two overlapping runs

**Caveat: the two runs ran at the same time on the same VM** (the job was
started twice by mistake). They shared the GPU and the CPU, which slows
timing and, above all, compilation. Both agents named the 60 s compile limit
as what stopped them from using more query blocks, so these numbers are
probably lower than a run on its own would give.

| Run | Re-measured speedup | Turns | Time | Output tokens (thinking) | API-price equivalent |
|---|---|---|---|---|---|
| 1622 | 1.25x | 20 | 36 min | 16,253 (6,013) | $1.32 |
| 1623 | 1.36x | 13 | 33 min | 12,267 (4,146) | $1.02 |

Both found the causal block rewrite with 4 query blocks, no precision change.
For comparison, at default precision: the single call got 1.56x to 1.81x and
our tool agent 1.81x to 3.24x (gpt-6.1-sol and Fable).

**Audit of the logs** (`claude_log.jsonl` holds every tool call): all 31
calls were shell commands. Outside their own folder they only used the
project's Python interpreter and their own task-output files under
`/tmp/claude-1002/`. Neither run read the hidden results or used `sudo`.
Run 1622 noticed the other job on the GPU and looked up which process it was
(`ps`, `readlink /proc/<pid>/cwd`), but did not open that job's files.
