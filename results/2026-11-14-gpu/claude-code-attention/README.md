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
