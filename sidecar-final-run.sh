#!/usr/bin/env bash
# One GPU sidecar arm end to end on a free box: boot, sample memory, run the
# full pinned suite into $SIDECAR_DATA, stop. Usage:
#   SIDECAR_DATA=<dir> ./sidecar-final-run.sh <arm> <hf-model> <rev> [quant]
set -uo pipefail
arm="${1:?arm}"; model="${2:?hf model}"; rev="${3:?rev}"; quant="${4:-}"
cd /home/jim/llm/dgx-spark-qwen38
: "${SIDECAR_DATA:?set SIDECAR_DATA}"
csv="$SIDECAR_DATA/mem-$arm.csv"
log="$SIDECAR_DATA/engine-$arm.log"
for _ in $(seq 1 30); do docker rm -f qwen38-sidecar >/dev/null 2>&1; docker ps -aq --filter name=qwen38-sidecar | grep -q . || break; sleep 2; done
./sidecar-memsample.sh "$csv" 14400 >/dev/null 2>&1 &
samp=$!
t0=$(date +%s)
SIDECAR_MODEL="$model" SIDECAR_REV="$rev" SIDECAR_QUANT="$quant" SIDECAR_NAME="$arm" SIDECAR_NO_KEY=1 \
  ./sidecar-launch.sh > "$log" 2>&1 &
ready=0
for i in $(seq 1 180); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:30010/v1/models)" = 200 ] && { ready=1; break; }
  [ $i -gt 6 ] && [ -z "$(docker ps -q --filter name=qwen38-sidecar)" ] && break
  sleep 5
done
if [ "$ready" != 1 ]; then echo "$(date -u +%T) $arm: engine did not come up"; tail -5 "$log"; kill $samp; exit 1; fi
echo "$(date -u +%T) $arm: ready after $(( $(date +%s) - t0 )) s"
SIDECAR_MODEL="$arm" ./sidecar-bench-loop.sh "$arm" http://127.0.0.1:30010
docker stop -t 20 qwen38-sidecar >/dev/null 2>&1
for _ in $(seq 1 30); do docker ps -aq --filter name=qwen38-sidecar | grep -q . || break; sleep 2; done
kill $samp 2>/dev/null
echo "$(date -u +%T) $arm: done"
