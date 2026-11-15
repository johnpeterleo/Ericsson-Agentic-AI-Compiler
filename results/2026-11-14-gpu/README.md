# GPU results, 2026-11-14

All runs: one NVIDIA L4 (spot `g2-standard-4`, `asia-east1-a`),
`JAX_DEFAULT_MATMUL_PRECISION=highest`, code at commit `6a612ed`. The code was
frozen before these runs; nothing was changed after seeing results.
`python results/2026-11-14-gpu/summarize.py` prints the E3 and E4 tables from
the records.

| File | What it is |
|---|---|
| `e1-suite-gemini-3.6-flash.jsonl` | E1: the last two suite programs, gemini-3.6-flash, 2 attempts each |
| `e2-timing-spread-*.json` | E2: speedup reported for two identical programs, 90 trials each |
| `e3-planted-*.jsonl` | E3: 3 planted programs x 5 arms x 3 repetitions, Codex `gpt-6-luna` (medium reasoning), 12 calls per arm |
| `e4-realistic-kernels*.jsonl` | E4: programs in `data/programs/`, single call and tool agent, 1 repetition |
| `e4-realistic-kernels-gpt-6.1-sol.jsonl` | E4 again with Codex `gpt-6.1-sol` (medium reasoning), same code and settings |
| `e4-attention-repetitions-gpt-6.1-sol.jsonl` | Two more tool-agent runs on attention, matmul precision highest |
| `e4-default-precision-gpt-6.1-sol.jsonl` | E4 with the references at XLA's default matmul precision |
| `e4-default-precision-repetitions-gpt-6.1-sol.jsonl` | Repetitions 2 and 3 on attention and transformer_block, default precision (run 2026-11-15) |
| `e4-claude-fable-5.2.jsonl` | softmax and attention with Claude Fable 5.2, default precision (run 2026-11-15) |
| `e4-claude-fable-5.2-batch.jsonl` | More Fable repetitions on attention and transformer_block (2026-11-15); Claude Code runs in `claude-code-attention/` |
| `log-*.txt` | Console output of the runs |

## E1: suite, one call is enough

`evaluate_polynomial`: expert 3.24x, agent 3.11x at attempt 0. `matrix_power`:
expert 2.93x, agent 2.91x at attempt 0. With the Nov 10 and Nov 11 runs,
gemini-3.6-flash found the expert's rewrite on its first attempt for all 13
suite programs.

## E2: timing spread on the L4

5th to 95th percentile of the speedup the evaluator reports for two identical
programs: 0.992x to 1.016x (scan MLP) and 0.972x to 1.025x (vectorized MLP).
A speedup below about 1.03x is not evidence of an improvement.

## E3: planted programs, five arms at 12 LLM calls

Re-measured speedup (50 repeats, an input the search never saw) for
repetitions 1, 2, 3. "none" means no candidate passed the correctness check.
The expert column is the clean version of each program.

| Program | Expert | single | best_of_n | chain | chain_original | tool_agent |
|---|---|---|---|---|---|---|
| causal_self_attention | 5.6x | none, none, none | 4.95x, 5.72x, 4.84x | 1.00x, 4.77x, 4.84x | 5.58x, none, none | 6.59x, 4.74x*, 4.89x |
| embed_and_transform | 85x | none, none, 1.55x | 94.8x, 93.0x, 97.3x | 95.1x, 94.9x, 89.9x | 93.2x, 89.7x, 94.2x | 90.3x, 90.5x, 92.4x |
| local_standardize | 75x | 42.8x, none, none | 50.3x, 45.8x, 70.9x | 89.4x, 51.1x, 46.4x | 76.1x, none, 79.5x | 50.0x, none*, 71.0x |

LLM calls used per run: single 1; best_of_n and chain 12; chain_original 1 to
10; tool_agent 2 to 5, except one run of 11.

\* Rerun. In repetition 2 the tool agent's run on these two programs was
skipped because Codex returned an empty reply, which the runner treats as
"model unavailable". The runs were repeated with the same code and settings
(`e3-planted-tool-agent-reruns.jsonl`). Codex returned an empty reply in 6 of
the 21 tool-agent runs started across E3 and E4.

What the table supports:

- One call is not enough here: the single call produced a passing rewrite in 2 of 9 runs.
- With several calls, no arm is clearly better than the others. The spread between repetitions of one arm is as large as the gaps between arms.
- best_of_n is the steadiest arm. The chains and the tool agent each have runs with no gain.
- The tool agent reaches the same range with 2 to 5 calls instead of 12, and it sometimes finishes while more calls would have helped.
- chain_original (the November prompt, which names tools the model lacks) stopped without a passing candidate in 3 of 9 runs.
- In repetition 1, the chain on causal_self_attention misread the reference and built on that reading for 7 attempts.

## E4: realistic kernels, nothing to gain over XLA

| Program | single | tool_agent | Note |
|---|---|---|---|
| matmul | no attempt | 2.06x | Not a real gain: the candidate sets `precision=DEFAULT` on its products, which lowers the arithmetic precision; the 1e-2 tolerance lets it pass |
| softmax | 0.66x | 1.00x | |
| fused_gemm_gelu_layernorm | 0.01x | 1.00x | |
| attention | none | 0.70x | Tool agent from the rerun (two more empty Codex replies first) |
| transformer_block | 0.88x | 0.99x | |
| gpt | skipped | skipped | The reference takes over 60 s to compile at this size |

A value below 1x means the best passing candidate was slower than the
reference, so the reference would be kept.

## E4 with a stronger model: gpt-6.1-sol

Same code, settings and 12-call budget as the gpt-6-luna runs, 1 repetition.
Each run was retried when Codex returned an empty reply.

| Program | single | tool_agent | Note |
|---|---|---|---|
| softmax | none | 1.01x | Inside the timing spread; 16 evaluations |
| fused_gemm_gelu_layernorm | 0.98x | 2.34x | Not a real gain: every candidate sets `precision='TF32_TF32_F32'` on the matrix product |
| attention | none | **1.94x** | A real gain, no precision change: each block of queries only computes scores for the keys up to its own position, and heads are processed in groups. Search speedups went 0.95x, 1.22x, 1.41x, 2.01x over 16 evaluations |
| transformer_block | none | 0.96x | 10 evaluations |
| matmul | 0.86x | 1.00x | The tool agent did not change the precision this time |
| gpt | skipped | skipped | The reference takes over 300 s to compile at this size (limit raised from 60 s for this program) |

The attention result is one run and should be repeated before it is reported
as more than an example.

The precision results point at the setup, not at the agent: the references
run with `JAX_DEFAULT_MATMUL_PRECISION=highest`, while these programs'
tolerance (1e-2) was chosen to allow TF32. A candidate that asks for TF32
then looks about 2x faster on any program dominated by a matrix product.

## E4 follow-ups with gpt-6.1-sol

**Attention again, precision highest (2 more tool-agent runs): 1.82x and
1.81x.** Both winning candidates combine the causal block rewrite with
`TF32_TF32_F32` precision, so they mix a real gain with the precision trade.
In the first of them the best candidate without a precision change reached
1.43x during the search. Only the first run (1.94x) is free of it.

**References at XLA's default matmul precision** (the environment variable is
not set), 1 repetition. Asking for lower precision no longer pays, and no
winning candidate changes the precision:

| Program | single | tool_agent | Note |
|---|---|---|---|
| attention | 1.59x | **3.24x** | Causal block rewrite: 512-row query blocks only score the keys up to their own position, heads in groups of four |
| transformer_block | none | **1.54x** | The same causal block rewrite applied to the attention inside the block |
| fused_gemm_gelu_layernorm | 0.95x | 1.00x | The 2.34x seen at precision highest is gone |
| matmul | 0.28x | 1.00x | |
| softmax | none | 1.01x | Inside the timing spread |

So with a fair reference, the tool agent with gpt-6.1-sol beat XLA on the two
programs that contain causal attention, and on nothing else. Each number is
one run. The gain is one idea: XLA computes the scores that the causal mask
then throws away, and the rewrite skips them.

### Repetitions at default precision (attention and transformer_block)

Re-measured speedups for repetitions 1, 2, 3 with gpt-6.1-sol. No winning
candidate changes the precision.

| Program | single | tool_agent |
|---|---|---|
| attention | 1.59x, 1.81x, 1.59x | 3.24x, 1.96x, 1.81x |
| transformer_block | none, none, none | 1.54x, 1.16x, 1.16x |

- Attention: both arms beat XLA in every run. The tool agent is ahead of the single call in each repetition, clearly in the first (3.24x) and narrowly in the other two.
- transformer_block: the single call never produced a passing rewrite; the tool agent gained in every run (above the 1.03x timing spread), by 16% in two of three.

## E4 with Claude Fable 5.2 (2026-11-15)

Claude Fable 5.2 through the Claude Code CLI, references at XLA's default
matmul precision, 12-call budget, 1 repetition. The code is commit `6a612ed`
with one change: `REQUEST_TIMEOUT_MS` raised from 120 s to 900 s, so long
Fable replies are not cut off. Only two programs were run, because each Fable
call used a large share of the subscription's usage limit.

| Program | single | tool_agent | Note |
|---|---|---|---|
| softmax | 0.72x | 1.00x | 3 calls, then the agent finished: nothing to gain, as with gpt-6.1-sol |
| attention | 1.56x | **2.13x** | Causal block rewrite (12 query blocks, each scoring only its key prefix), no precision change. 7 calls, 15 evaluations |

Attention across the two models, default precision:

| Model | single | tool_agent |
|---|---|---|
| gpt-6.1-sol (3 runs) | 1.59x, 1.81x, 1.59x | 3.24x, 1.96x, 1.81x |
| claude-fable-5.2 (1 run) | 1.56x | 2.13x |

Both models find the same rewrite, and in every run the tool agent ends
ahead of the single call.
