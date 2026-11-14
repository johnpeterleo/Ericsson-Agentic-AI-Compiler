"""Print the comparison tables from the agent-comparison records in this folder.

    python results/2026-11-14-gpu/summarize.py
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

FOLDER = Path(__file__).parent
ARMS = ("single", "best_of_n", "chain", "chain_original", "tool_agent")


def load_runs(pattern):
    """Finished runs as {(task, arm): [record, ...]}, in file order."""
    runs = defaultdict(list)
    for path in sorted(FOLDER.glob(pattern)):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["status"] == "done":
                runs[record["task"], record["arm"]].append(record)
    return runs


def speedup_text(record):
    speedup = record["remeasured_speedup"]
    return "none" if speedup is None else f"{speedup:.2f}x"


def print_table(runs):
    tasks = sorted({task for task, _ in runs})
    for task in tasks:
        print(f"\n{task}")
        for arm in ARMS:
            records = runs.get((task, arm), [])
            if not records:
                continue
            # A run with no correct candidate keeps the reference, which is 1.0x.
            values = [record["remeasured_speedup"] or 1.0 for record in records]
            calls = [record["llm_calls"] for record in records]
            print(
                f"  {arm:15s} {', '.join(speedup_text(r) for r in records):28s} "
                f"median {median(values):6.2f}x   calls {', '.join(map(str, calls))}"
            )


if __name__ == "__main__":
    print("E3: planted programs, 12 calls per arm, re-measured speedup per repetition")
    print_table(load_runs("e3-*.jsonl"))
    print("\nE4: programs in data/programs/, 1 repetition")
    print_table(load_runs("e4-*.jsonl"))
