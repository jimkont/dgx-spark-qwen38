#!/usr/bin/env bash
# MemAvailable + per-engine RSS sampler for the sidecar experiment (issue #42).
# Usage: ./sidecar-memsample.sh <out.csv> [seconds]   (default 3600)
# One line every 2 s: epoch, MemAvailable_kib, docker stats for the flash and
# sidecar containers (names tolerantly missing).
set -euo pipefail
out="${1:?usage: sidecar-memsample.sh out.csv [seconds]}"
secs="${2:-3600}"
echo "memsample: writing $out for ${secs}s"
end=$(( $(date +%s) + secs ))
echo "epoch,memavailable_kib,flash_rss_gib,sidecar_rss_gib" > "$out"
while [ "$(date +%s)" -lt "$end" ]; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  cstats() { # container RSS in GiB (docker stats), nan if the container is absent
    local v
    v=$(docker stats --no-stream --format '{{.MemUsage}}' "$1" 2>/dev/null | awk '{print $1}' || true)
    [ -z "$v" ] && { echo nan; return; }
    awk -v v="$v" 'BEGIN{ if (v ~ /GiB/) {gsub(/GiB/,"",v); printf "%.2f", v}
      else if (v ~ /MiB/) {gsub(/MiB/,"",v); printf "%.2f", v/1024}
      else if (v ~ /KiB/) {gsub(/KiB/,"",v); printf "%.2f", v/1048576}
      else printf "nan" }'
  }
  fr=$(cstats qwen38-flash)
  sr=$(cstats qwen38-sidecar)
  echo "$(date +%s),$a,$fr,$sr" >> "$out"
  sleep 1
done
