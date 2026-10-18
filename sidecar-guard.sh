#!/usr/bin/env bash
# Watchdog for the sidecar lanes (issue #42): keeps the serving lane from
# freezing while second engine(s) run beside it. Watches every container
# named qwen38-sidecar* (the ad-hoc one, qwen38-sidecar-tiny, -small), and
# stops them all when the box's memory is exhausted. Polls every 2 s.
# Warning line (default 9.5 GiB): a BF16 weight load dips the whole box to
# about 3.2 GiB for under two minutes; that transient is logged, not fatal.
# Hard floor (default 2.5 GiB): below it a dip is NOT the boot transient
# anymore - stop the sidecar. earlyoom is inactive on this box; the real
# failure mode is unified-memory exhaustion freezing the machine, so the
# hard floor is the guard.
set -uo pipefail
WARN_GIB="${SIDECAR_MEM_WARN_GIB:-9.5}"
FLOOR_GIB="${SIDECAR_MEM_FLOOR_GIB:-2.5}"
WARN_KIB=$(awk "BEGIN{printf \"%d\", $WARN_GIB * 1048576}")
FLOOR_KIB=$(awk "BEGIN{printf \"%d\", $FLOOR_GIB * 1048576}")
log() { echo "[$(date -u +'%F %T')] sidecar-guard: $*"; }
log "guard up, warn ${WARN_GIB} GiB, kill ${FLOOR_GIB} GiB"
while :; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  ids=$(docker ps -q --filter name=qwen38-sidecar 2>/dev/null || true)
  if [ -n "$ids" ] && [ "$a" -lt "$FLOOR_KIB" ]; then
    log "MemAvailable $((a / 1048576)) GiB under hard floor; stopping the sidecar(s) to protect the serving lane"
    for c in $ids; do docker stop -t 10 "$c" >/dev/null 2>&1; log "stopped $c"; done
    log "sidecar(s) stopped by guard; investigate before restarting"
    exit 0
  fi
  if [ -n "$ids" ] && [ "$a" -lt "$WARN_KIB" ] && [ "$a" -ge "$FLOOR_KIB" ] && [ $(($(date +%s) - ${last_warn:-0})) -ge 60 ]; then
    last_warn=$(date +%s)
    log "MemAvailable $((a / 1048576)) GiB under the warning line (a boot transient is expected to recover; if this persists at steady idle, stop the sidecar(s): they do not fit beside the lane)"
  fi
  sleep 2
done
