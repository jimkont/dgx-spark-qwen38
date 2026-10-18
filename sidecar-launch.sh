#!/usr/bin/env bash
# Sidecar engine for /v1/systemone (issue #42): Qwen3.5-2B beside the Flash lane.
# Text-only (no vision), no speculation, no CUDA graphs, small context and a
# pinned token budget so it cannot eat the pool the Flash lane sizes from.
# Port 30010, its own name so it never touches qwen38-flash.
set -euo pipefail

MODEL="${SIDECAR_MODEL:-Qwen/Qwen3.5-2B}"
REV="${SIDECAR_REV:-15852e8c16360a2fea060d615a32b45270f8a8fc}"
QUANT="${SIDECAR_QUANT:-}"          # e.g. compressed-tensors for the NVFP4 export
CTX="${SIDECAR_CTX:-16384}"
MAXTOK="${SIDECAR_MAX_TOTAL_TOKENS:-16384}"
MEMFR="${SIDECAR_MEM_FRACTION:-0.60}"   # rest = F_free*f - W - mm, F_free is what's left beside Flash
MAXRUN="${SIDECAR_MAX_RUNNING_REQUESTS:-8}"
MAMBA="${SIDECAR_MAX_MAMBA_CACHE_SIZE:-32}"   # 18.63 MB/req, 3 slots/req with radix on
NAME="${SIDECAR_NAME:-qwen3.5-2b-sidecar}"
CONTAINER="${SIDECAR_CONTAINER:-qwen38-sidecar}"
PORT="${SIDECAR_PORT:-30010}"

QUANT_ARGS=()
[ -n "$QUANT" ] && QUANT_ARGS=(--quantization "$QUANT")
NOKEY="${SIDECAR_NO_KEY:-}"
# No-key mode (SIDECAR_NO_KEY=1): omit the secrets config so the engine runs
# unauthenticated — for local testing only, never for an exposed host.
CONF_ARGS=()
[ -z "$NOKEY" ] && CONF_ARGS=(--config /out/engine-secrets.yaml)

avail_kib() { awk '/MemAvailable/{print $2}' /proc/meminfo; }
echo "qwen38-sidecar: host MemAvailable at launch $(( $(avail_kib) / 1048576 )) GiB (Flash lane must be up first)"

exec /usr/bin/docker run --rm --name "$CONTAINER" --gpus all \
  --memory 12g --memory-swap 12g --shm-size 2g --network host --ipc=host \
  -v /home/jim/.cache/huggingface:/root/.cache/huggingface \
  -v /home/jim/.config/qwen38:/out \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 \
  -e SGLANG_VLM_CACHE_SIZE_MB=0 \
  qwen38-pinned:flash-cb6ed363800a \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" --revision "$REV" --tp-size 1 \
    --served-model-name "$NAME" \
    "${QUANT_ARGS[@]}" \
    --language-only \
    --page-size 64 \
    --mem-fraction-static "$MEMFR" \
    --max-total-tokens "$MAXTOK" \
    --context-length "$CTX" \
    --chunked-prefill-size 2048 \
    --max-running-requests "$MAXRUN" \
    --max-mamba-cache-size "$MAMBA" \
    --disable-cuda-graph \
    --disable-flashinfer-autotune \
    --enable-metrics \
    "${CONF_ARGS[@]}" \
    --host 127.0.0.1 --port "$PORT"
