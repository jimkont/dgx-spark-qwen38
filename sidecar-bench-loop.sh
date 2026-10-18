#!/usr/bin/env bash
# Self-resuming quality suite for the sidecar experiment (issue #42).
# Survives engine aborts: bench-systemone.py writes predictions as they arrive
# and resumes by id, so every retry continues where the last run died.
# Usage: SIDECAR_MODEL=<served name> ./sidecar-bench-loop.sh <target-name> <url> [task ...]
#   ./sidecar-bench-loop.sh s08b http://127.0.0.1:30010 public inject gdpr
set -uo pipefail
name="${1:?usage: sidecar-bench-loop.sh name url [tasks...]}"
url="${2:?usage: sidecar-bench-loop.sh name url [tasks...]}"
shift 2
if [ "$#" -gt 0 ]; then tasks=("$@"); else tasks=(public inject gdpr boolq xnli-fr mmmlu-fr mmlu-pro); fi
MODEL="${SIDECAR_MODEL:-}"
export PATH=/home/jim/venvs/bench/bin:$PATH
DATA="${SIDECAR_DATA:-/home/jim/llm/dgx-spark-qwen38/bench-systemone-data}"
cd /home/jim/llm/dgx-spark-qwen38
for t in "${tasks[@]}"; do
  until ./bench-systemone.py --data "$DATA" run --task "$t" --target "$name=$url" ${MODEL:+--model "$MODEL"}; do
    tries=$((tries + 1))
    if [ "$tries" -ge 60 ]; then echo "$(date -u +'%F %T') bench-loop: giving up on $t after $tries tries"; break; fi
    echo "$(date -u +'%F %T') bench-loop: $t died (try $tries), resuming in 20 s"
    sleep 20
  done
  echo "$(date -u +'%F %T') bench-loop: $t done"
done
echo "bench-loop: suite complete for target $name"
