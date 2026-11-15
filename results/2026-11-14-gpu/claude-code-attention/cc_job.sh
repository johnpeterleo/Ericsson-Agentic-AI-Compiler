#!/bin/bash
# Claude Code with Fable on one task, in a folder of its own, with earlier results hidden.
# Usage: cc_job.sh <task>
task=$1; folder=$HOME/cc-$task-$(date -u +%m%d-%H%M)
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1
unset JAX_DEFAULT_MATMUL_PRECISION
PY=$HOME/Ericsson-Agentic-AI-Compiler/.venv/bin/python
HIDDEN=/var/tmp/hidden-results
restore() {
  for item in $(sudo ls $HIDDEN 2>/dev/null); do sudo mv "$HIDDEN/$item" "$(echo "$item" | sed 's#__#/#g')"; done
  sudo rmdir $HIDDEN 2>/dev/null
  echo "earlier results restored"
}
trap 'restore; echo "JOB_DONE $(date -u +%H:%M)"' EXIT

# Hide everything that holds earlier candidates, in a folder only root can read.
sudo mkdir -m 700 $HIDDEN
for path in $HOME/frozen-6a612ed/artifacts $HOME/frozen-6a612ed-fable/artifacts $HOME/Ericsson-Agentic-AI-Compiler/artifacts \
            $HOME/tool-agent $HOME/tool-agent-new $HOME/gpu-results.tgz $HOME/branch.tar.gz $HOME/fixes.py $HOME/planted_gpu.py \
            $HOME/mlp-results $HOME/gpu-samples /tmp/sol /tmp/fu /tmp/rp \
            $HOME/frozen-6a612ed/results $HOME/frozen-6a612ed-fable/results $HOME/Ericsson-Agentic-AI-Compiler/results; do
  [ -e "$path" ] && sudo mv "$path" "$HIDDEN/$(echo "$path" | sed 's#/#__#g')"
done
for log in $HOME/frozen-6a612ed/*.out $HOME/frozen-6a612ed-fable/*.out; do
  [ -e "$log" ] && sudo mv "$log" "$HIDDEN/$(echo "$log" | sed 's#/#__#g')"
done
echo "hidden: $(sudo ls $HIDDEN | wc -l) items"

$PY ~/cc_setup.py $task $folder || exit 1
cd $folder
echo "=== Claude Code started $(date -u +%H:%M)"
timeout 2820 claude -p "$(cat PROMPT.md)" --model claude-fable-5-2 --permission-mode bypassPermissions \
  --disallowedTools WebFetch WebSearch --output-format stream-json --verbose > claude_log.jsonl 2> claude_stderr.txt
echo "claude exit code $? at $(date -u +%H:%M)"
echo "=== scoring"
$PY ~/cc_score.py $task $folder > score.json 2> score_stderr.txt; tail -c 2000 score_stderr.txt | grep -iv warning | tail -3
python3 -c "
import json
s = json.load(open('score.json'))
print({k: v for k, v in s.items() if k != 'source'})
"
echo "FOLDER $folder"
