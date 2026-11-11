# GPU results, 2026-11-11

Same setup as `../2026-11-10-gpu/`: one NVIDIA L4 (spot `g2-standard-4`,
`asia-east1-a`), `JAX_DEFAULT_MATMUL_PRECISION=highest`, 20 timing repeats.

| File | What it is |
|---|---|
| `size-sweep-new-programs.jsonl` | The four new programs at several sizes, 10 repeats. For `moving_window_sums` it also records the error of the running-total rewrite. |
| `agent-gemini-3.6-flash.jsonl` | The agent on the four new programs and three not run on Nov 10, 2 attempts each. |
| `agent-gemini-3.6-flash.log` | Console output of that run, including Gemini retries. |

Agent run with gemini-3.6-flash (2 attempts per program):

| Program | Expert | Agent's best | % of expert | Solved at | Second attempt |
|---|---|---|---|---|---|
| moving_window_sums | 17.95x | 18.14x | 101% | attempt 0 | rejected: running-total rewrite failed the correctness check |
| circulant_times_vectors | 4.09x | 4.09x | 100% | attempt 0 | 1.14x (kept the matrix product) |
| low_rank_plus_diagonal_times_vectors | 24.15x | 25.43x | 105% | attempt 0 | none, Gemini stopped |
| matrix_polynomial_times_vector | 20.55x | 20.70x | 101% | attempt 0 | 20.3x |
| prefix_sums | 9.39x | 9.11x | 97% | attempt 0 | 8.6x |
| trace_of_product | 8.18x | 8.34x | 102% | attempt 0 | 8.2x |
| evaluate_polynomial | 3.18x | - | - | - | daily quota used up |

Across both days the agent found the expert's rewrite on its first attempt for
all 11 programs it reached. No second attempt was clearly better than the
first, and one second attempt (`moving_window_sums`) was a plausible but
numerically wrong rewrite that the correctness check rejected. Shares above
100% are within the timing spread.

The free-tier daily quota ran out after 11 answered requests: requests that
Gemini answered with 503 appear to count against it.
