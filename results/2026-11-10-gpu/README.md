# GPU results, 2026-11-10

All runs: one NVIDIA L4 (spot `g2-standard-4`, `asia-east1-a`),
`JAX_DEFAULT_MATMUL_PRECISION=highest`, 20 timing repeats unless noted. The
code was commit `e4df9c7` with the suite's 9 programs at the time.

| File | What it is |
|---|---|
| `size-sweep.jsonl` | Each program at several sizes, 10 repeats; used to pick the benchmark sizes. |
| `expert-only.jsonl` | Expert speedup of all 9 programs at the chosen sizes. No LLM. |
| `agent-gemini-3.6-flash.jsonl` | The agent on 6 programs, 3 attempts each. Every attempt's code, verdict and timings. |
| `agent-gemini-3.6-flash.log` | Console output of that run, including Gemini retries. |
| `agent-gemini-3.8-flash-aborted.jsonl` | First agent attempt with gemini-3.8-flash. Gemini returned 503/504 on every retry for 2 of 3 programs, so the run was stopped and repeated with 3.6. |

Agent run with gemini-3.6-flash:

| Program | Expert | Agent's best | % of expert | Solved at |
|---|---|---|---|---|
| attention | 34.21x | 34.22x | 100% | attempt 0 |
| embed_tokens | 54.88x | 57.75x | 105% | attempt 0 |
| scale_rows_and_columns | 24.06x | 25.29x | 105% | attempt 0 |
| matrix_chain_times_vector | 20.42x | 21.30x | 104% | attempt 0 |
| kronecker_product_times_vector | 13.01x | 12.86x | 99% | attempt 0 |
| prefix_sums | 9.27x | - | - | daily quota used up |

Shares above 100% are within the timing spread. The agent found the expert's
rewrite on its first attempt for every program it reached. Of 12 attempts, one
reply was not valid JSON and one candidate failed to compile.
