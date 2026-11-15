#!/bin/bash
# More Fable repetitions, in rounds that each form a complete comparison:
#   round 1 and 2: attention, single call + tool agent + Claude Code
#   round 3:       transformer_block, tool agent + Claude Code
# Everything runs one after the other, references at XLA's default matmul precision.
# The batch stops as soon as a run reports the usage limit.
# Usage: nohup bash ~/fable_batch.sh > ~/fable_batch.out 2>&1 < /dev/null &
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
unset JAX_DEFAULT_MATMUL_PRECISION
PY=$HOME/Ericsson-Agentic-AI-Compiler/.venv/bin/python
trap 'echo "BATCH_DONE $(date -u +%H:%M)"; (sleep 900; sudo shutdown -h now) > /dev/null 2>&1 &' EXIT

stop_if_limited() {  # stop_if_limited <text>
  if echo "$1" | grep -qiE "usage limit|limit reached|Stopping:|hit your limit"; then
    echo "USAGE LIMIT reached, stopping the batch $(date -u +%H:%M)"
    exit 0
  fi
}

agent_run() {  # agent_run <task> <arm>: our runner, retried when the model returns nothing
  for try in 1 2 3; do
    out=$(cd ~/frozen-6a612ed-fable && $PY -m experiments.compare_agents --size default --model claude \
      --model-name claude-fable-5-2 --calls 12 --repetitions 1 --arms $2 --tasks $1 2>&1 | grep -v -i "warning")
    echo "$out" | grep -E "re-measured|skipped|Stopping|Saving"
    stop_if_limited "$out"
    echo "$out" | grep -qE "re-measured|Compilation took" && break
  done
}

claude_code_run() {  # claude_code_run <task>
  out=$(bash ~/cc_job.sh $1 2>&1)
  echo "$out" | grep -E "FOLDER|remeasured|claude exit|hidden|restored"
  folder=$(echo "$out" | grep -oE "FOLDER [^ ]+" | cut -d" " -f2)
  [ -z "$folder" ] && return
  # Only the run's final record counts: an error result that mentions a limit means the usage ran out.
  verdict=$(python3 -c "
import json, sys
result = None
for line in open(sys.argv[1]):
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get('type') == 'result':
        result = event
if result is not None and result.get('is_error') and 'limit' in str(result.get('result', '')).lower():
    print('usage limit')
" "$folder/claude_log.jsonl" 2>/dev/null)
  stop_if_limited "$verdict"
}

for round in 1 2; do
  echo "=== round $round: attention $(date -u +%H:%M)"
  agent_run attention single
  agent_run attention tool_agent
  claude_code_run attention
done
echo "=== round 3: transformer_block $(date -u +%H:%M)"
agent_run transformer_block tool_agent
claude_code_run transformer_block
