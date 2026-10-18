#!/usr/bin/env bash
# Install the Qwen3.5 sidecar lanes for /v1/systemone (issue #42): one small
# SGLang engine per sidecar on its own port, each a systemd unit that coexists
# with the text lane(s) instead of competing for the door. Idempotent; safe to
# run on its own or from install.sh.
#
#   ./install-sidecar.sh                 both sidecars (default)
#   ./install-sidecar.sh --tiny          the 0.8B only,  port 30011
#   ./install-sidecar.sh --small         the 2B only,    port 30010
#   ./install-sidecar.sh --quant nvfp4   serve the pinned NVFP4 exports instead
#                                       of the BF16 weights; the rendered launch
#                                       script then passes
#                                       --quantization compressed-tensors
#   ./install-sidecar.sh --no-key        no API key on the engine (local dev)
#   ./install-sidecar.sh --no-smoke      start it, skip the /v1/systemone probe
#   ./install-sidecar.sh --uninstall [tiny|small|all]
#
# After each install or uninstall it also rewrites ~/.config/qwen38/sidecars and
# restarts the keepalive proxy, so /v1/systemone routes by model name to the
# sidecars (proxy v6.37).
#
# Measured on this box beside the Flash-Next lane (docs/sidecar.md): ~10 GiB
# steady each (the flash lane's 13.5 GiB floor reads 3.3 GiB with the 0.8B up,
# 2.8 with the 2B; suite-load peaks reach ~11.5). Both fit on the 27B lane.
# The guard (sidecar-guard.sh) watches MemAvailable at runtime.
set -euo pipefail

TINY=1; SMALL=1; QUANT=""; NO_KEY=0; NO_SMOKE=0; UNINSTALL=""
prev=""
for arg in "$@"; do
  case "$prev" in
    --quant)
      case "$arg" in
        nvfp4) QUANT=nvfp4 ;;
        *) echo "usage: --quant nvfp4" >&2; exit 1 ;;
      esac
      ;;
    --uninstall)
      case "$arg" in
        tiny|small|all) UNINSTALL="$arg" ;;
        *) echo "usage: --uninstall [tiny|small|all]" >&2; exit 1 ;;
      esac
      ;;
  esac
  case "$arg" in
    --tiny)      SMALL=0 ;;
    --small)     TINY=0 ;;
    --no-key)    NO_KEY=1 ;;
    --no-smoke)  NO_SMOKE=1 ;;
    --quant|--uninstall) : ;;
    -h|--help)   grep '^# ' "$0" | sed 's/^# //;s/^# $//'; exit 0 ;;
    -*)
      case "$arg" in
        --tiny|--small|--no-key|--no-smoke|--quant|--uninstall) : ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
      esac
      ;;
  esac
  prev="$arg"
done
if [ "$prev" = "--quant" ] || [ "$prev" = "--uninstall" ]; then
  echo "usage: $prev needs a value" >&2; exit 1
fi
if [ -n "$UNINSTALL" ]; then
  case "$UNINSTALL" in
    tiny)  UNINSTALL_SIZES="tiny" ;;
    small) UNINSTALL_SIZES="small" ;;
    all)   UNINSTALL_SIZES="tiny small" ;;
  esac
fi

if [ "$(id -u)" -eq 0 ]; then
  echo "install as your user, not root (systemd units run as your user): ./install-sidecar.sh $*" >&2
  exit 1
fi
command -v docker >/dev/null || { echo "docker is not available" >&2; exit 1; }

HOME_DIR="$HOME"
CONFIG_DIR="$HOME/.config/qwen38"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

step(){ echo; echo "--- $* ---"; }
die(){ echo "install-sidecar: $*" >&2; exit 1; }

# ── Pinned revisions (the same commits already in the local HF cache) ────────
TINY_BF16=Qwen/Qwen3.5-0.8B
TINY_BF16_REV=2fc06364715b967f1860aea9cf38778875588b17
SMALL_BF16=Qwen/Qwen3.5-2B
SMALL_BF16_REV=15852e8c16360a2fea060d615a32b45270f8a8fc
TINY_NVFP4=AxionML/Qwen3.5-0.8B-NVFP4
TINY_NVFP4_REV=2ac1e750cda67cc8538d731f6216f77b9c3a6f72
SMALL_NVFP4=AxionML/Qwen3.5-2B-NVFP4
SMALL_NVFP4_REV=6f2ad4febe92be529de499b08f0020f5ff06bf25

get_model(){
  case "$1" in
    tiny)  if [ -n "$QUANT" ]; then echo "$TINY_NVFP4";  else echo "$TINY_BF16";  fi ;;
    small) if [ -n "$QUANT" ]; then echo "$SMALL_NVFP4"; else echo "$SMALL_BF16"; fi ;;
  esac
}
get_rev(){
  case "$1" in
    tiny)  if [ -n "$QUANT" ]; then echo "$TINY_NVFP4_REV";  else echo "$TINY_BF16_REV";  fi ;;
    small) if [ -n "$QUANT" ]; then echo "$SMALL_NVFP4_REV"; else echo "$SMALL_BF16_REV"; fi ;;
  esac
}
get_served(){    case "$1" in tiny) echo qwen3.5-0.8b-sidecar ;; small) echo qwen3.5-2b-sidecar ;; esac; }
get_port(){      case "$1" in tiny) echo 30011 ;; small) echo 30010 ;; esac; }
get_unit(){      case "$1" in tiny) echo qwen38-sidecar-tiny.service ;; small) echo qwen38-sidecar-small.service ;; esac; }
get_container(){ case "$1" in tiny) echo qwen38-sidecar-tiny ;; small) echo qwen38-sidecar-small ;; esac; }
get_launch(){    case "$1" in tiny) echo launch-sidecar-tiny.sh ;; small) echo launch-sidecar-small.sh ;; esac; }

# The keepalive proxy routes /v1/systemone by the model field (v6.37) from one
# "<served name> <base url>" line per sidecar in $CONFIG_DIR/sidecars, read at
# start. Rewrite it from the sidecar units that exist, then restart the proxy
# so the table matches the lanes.
sync_sidecars(){
  mkdir -p "$CONFIG_DIR"
  local s
  { for s in tiny small; do
      if [ -f "/etc/systemd/system/$(get_unit $s)" ]; then
        printf '%s http://127.0.0.1:%s\n' "$(get_served $s)" "$(get_port $s)"
      fi
    done
  } > "$CONFIG_DIR/sidecars"
  if [ -f /etc/systemd/system/qwen38-keepalive.service ]; then
    sudo systemctl restart qwen38-keepalive \
      || echo "note: could not restart qwen38-keepalive; restart it so /v1/systemone routes to the sidecar(s)"
  else
    echo "note: the keepalive proxy is not installed; /v1/systemone is served by each engine itself on its port"
  fi
}

# ── Uninstall path ───────────────────────────────────────────────────────────
if [ -n "$UNINSTALL" ]; then
  for s in $UNINSTALL_SIZES; do
    unit="$(get_unit $s)"
    step "Uninstalling the $(get_served $s) sidecar"
    sudo systemctl disable --now "$unit" 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/$unit"
    rm -f "$CONFIG_DIR/$(get_launch $s)"
    echo "unit $unit removed; weights remain in $HF_CACHE (remove the matching models--*--Qwen3.5* dir to reclaim disk)"
  done
  sudo systemctl daemon-reload
  sync_sidecars
  echo "done."
  exit 0
fi

# ── Preflight ────────────────────────────────────────────────────────────────
step "Preflight"
avail_gib(){ awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
AVAIL="$(avail_gib)"
if [ "$TINY" -eq 1 ] && [ "$SMALL" -eq 1 ]; then NEED=24
elif [ "$TINY" -eq 1 ]; then NEED=12
else NEED=13; fi
[ "$AVAIL" -ge "$NEED" ] || die "$AVAIL GiB free; this needs about $NEED GiB (steady footprint plus boot-dip headroom). Free up memory or install fewer (--tiny / --small)."
if systemctl is-active --quiet qwen38-flash.service && [ "$TINY" -eq 1 ] && [ "$SMALL" -eq 1 ]; then
  die "the flash lane is active and BOTH sidecars need ~21 GiB steady against its ~13.5 GiB floor. Pick one: ./install-sidecar.sh --tiny or --small (the 27B lane, with ~29 GiB free, takes both)."
fi
if [ "$NO_KEY" -eq 0 ] && [ ! -f "$CONFIG_DIR/api-key" ]; then
  die "no key to serve with: $CONFIG_DIR/api-key is missing. Run ./install.sh first, or pass --no-key for a local unkeyed engine."
fi
IMAGE="$(docker inspect -f '{{.Config.Image}}' qwen38-sglang 2>/dev/null || true)"
[ -n "$IMAGE" ] || IMAGE="$(docker inspect -f '{{.Config.Image}}' qwen38-flash 2>/dev/null || true)"
[ -n "$IMAGE" ] || IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^qwen38-pinned:' | head -1 || true)"
[ -n "$IMAGE" ] || die "no SGLang image found (tried qwen38-sglang, qwen38-flash, then qwen38-pinned:*). Run ./install.sh first."
echo "engine image: $IMAGE"

mkdir -p "$CONFIG_DIR"
TOTAL=0
[ "$TINY" -eq 1 ] && TOTAL=$((TOTAL + 1))
[ "$SMALL" -eq 1 ] && TOTAL=$((TOTAL + 1))
N=0
for s in $( [ "$TINY" -eq 1 ] && echo tiny ) $( [ "$SMALL" -eq 1 ] && echo small ); do
  N=$((N + 1))
  model="$(get_model $s)"; rev="$(get_rev $s)"; port="$(get_port $s)"
  unit="$(get_unit $s)"; container="$(get_container $s)"; served="$(get_served $s)"
  step "$N/$TOTAL sidecar '$served' ($model @ ${rev:0:12}, port $port)"

  # 1. Weights: one-time, resumable. A complete local cache needs no network and no
  #    writable lock dirs (containers download as root into the shared cache), so
  #    the pinned revision's cache is checked first; only a miss goes to the network.
  HF_HOME="$HF_CACHE" HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_DISABLE_XET=1 \
    MODEL_REPO="$model" MODEL_REV="$rev" \
    python3 - <<'PY' || die "weight download failed: no internet, HuggingFace throttling an unauthenticated download (set HF_TOKEN), a pinned revision removed upstream, or a too-small disk. Re-running resumes."
import os, time
from huggingface_hub import snapshot_download
repo, rev = os.environ["MODEL_REPO"], os.environ["MODEL_REV"]
path = None
try:
    path = snapshot_download(repo, revision=rev, local_files_only=True)
    print("   weights already in the cache:", path, flush=True)
except Exception:
    pass
if path is None:
    for attempt in range(1, 5):
        try:
            path = snapshot_download(repo, revision=rev)
            print("   weights in", path, flush=True)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"   attempt {attempt} stopped ({type(e).__name__}); resuming in 10 s", flush=True)
            time.sleep(10)
sha = os.path.basename(path.rstrip("/"))
ref = os.path.join(os.path.dirname(os.path.dirname(path.rstrip("/"))), "refs", "main")
if len(sha) == 40 and sha == rev:
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    old = open(ref).read().strip() if os.path.exists(ref) else ""
    if old != sha:
        with open(ref, "w") as f:
            f.write(sha)
        print(f"   refs/main -> {sha[:12]}", flush=True)
PY

  # 2. Port: refuse to steal it.
  if ss -ltn 2>/dev/null | grep -qE "[.:]$port( |$)"; then
    die "port $port is already in use. The ad-hoc 'qwen38-sidecar' container from a manual run is the usual holder: stop it first (docker stop qwen38-sidecar) so this managed unit can take the port."
  fi

  # 3. Render the launch script (a plain file: no systemd-vs-bash escaping).
  launch="$CONFIG_DIR/$(get_launch $s)"
  if [ "$NO_KEY" -eq 1 ]; then secrets_call=""; conf_args=""; else
    secrets_call="/bin/bash $HOME_DIR/.config/qwen38/engine-secrets.sh"
    conf_args="--config /out/engine-secrets.yaml"
  fi
  # --quant nvfp4 serves the AxionML export; SGLang's own flag name for it is
  # compressed-tensors (the same one the ad-hoc sidecar-launch.sh takes).
  quant_args=""
  [ -n "$QUANT" ] && quant_args="--quantization compressed-tensors"
  sed -e "s|__SIDECAR_MODEL__|$model|g" \
      -e "s|__SIDECAR_REV__|$rev|g" \
      -e "s|__SIDECAR_QUANT_ARGS__|$quant_args|g" \
      -e "s|__SIDECAR_NAME__|$served|g" \
      -e "s|__SIDECAR_PORT__|$port|g" \
      -e "s|__SIDECAR_CTX__|16384|g" \
      -e "s|__SIDECAR_MAXTOK__|16384|g" \
      -e "s|__SIDECAR_MEM_FRACTION__|0.60|g" \
      -e "s|__SIDECAR_MAX_RUNNING_REQUESTS__|8|g" \
      -e "s|__SIDECAR_MAX_MAMBA_CACHE_SIZE__|32|g" \
      -e "s|__SIDECAR_CONTAINER__|$container|g" \
      -e "s|__HF_CACHE__|$HF_CACHE|g" \
      -e "s|__CONFIG_DIR__|$CONFIG_DIR|g" \
      -e "s|__IMAGE__|$IMAGE|g" \
      -e "s|__SIDECAR_SECRETS_CALL__|$secrets_call|g" \
      -e "s|__SIDECAR_CONF_ARGS__|$conf_args|g" \
      "$REPO_DIR/qwen38-sidecar-launch.sh.template" > "$launch"
  chmod 0700 "$launch"

  # 4. Render + install the unit, enable and start it.
  sed -e "s|__PORT__|$port|g" \
      -e "s|__CONTAINER__|$container|g" \
      -e "s|__LAUNCH__|$(get_launch $s)|g" \
      -e "s|__USER__|$(id -un)|g" \
      -e "s|__GROUP__|$(id -gn)|g" \
      -e "s|__HOME__|$HOME_DIR|g" \
      "$REPO_DIR/qwen38-sidecar.service.template" > "$CONFIG_DIR/.sidecar-unit.$s"
  sudo install -m 0644 -o root -g root "$CONFIG_DIR/.sidecar-unit.$s" "/etc/systemd/system/$unit"
  rm -f "$CONFIG_DIR/.sidecar-unit.$s"
  sudo systemctl daemon-reload
  sudo systemctl enable "$unit"
  echo "starting $unit"
  sudo systemctl start "$unit"

  # 5. Health: the 0.8B loads in ~30 s, the 2B took 183 s on this box; allow 10 min.
  ok=0
  for i in $(seq 1 60); do
    curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    systemctl is-active --quiet "$unit" || break
    sleep 10
  done
  [ "$ok" -eq 1 ] || die "$unit is not healthy after 10 min. Check: journalctl -u $unit -n 40"
  echo "healthy on :$port"

  # 6. Smoke: one /v1/systemone call against the engine's own port, as a Jev
  #    client would meet it.
  if [ "$NO_SMOKE" -eq 1 ]; then
    echo "smoke skipped (--no-smoke)"
  else
    auth=()
    [ "$NO_KEY" -eq 1 ] || auth=("-H" "Authorization: Bearer $(cat "$CONFIG_DIR/api-key")")
    code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/systemone" \
      ${auth[@]+"${auth[@]}"} \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$served\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"Is the request asking for a refund?\",\"criteria\":{\"true\":\"yes, a refund\",\"false\":\"no\"}}},\"state\":\"A customer wrote: I want my money back for the broken widget.\"}")"
    [ "$code" = 200 ] || die "the /v1/systemone probe answered $code, not 200. Check: journalctl -u $unit -n 40"
    echo "answered a /v1/systemone probe (HTTP $code)"
  fi
done

step "Routing /v1/systemone to the sidecars"
sync_sidecars

step "Done"
{ [ "$TINY" -eq 1 ]; } && echo "  tiny:  qwen3.5-0.8b-sidecar -> http://127.0.0.1:30011 (unit qwen38-sidecar-tiny.service)"
{ [ "$SMALL" -eq 1 ]; } && echo "  small: qwen3.5-2b-sidecar  -> http://127.0.0.1:30010 (unit qwen38-sidecar-small.service)"
echo "  memory guard (optional): ./sidecar-guard.sh"
[ "$TINY" -eq 1 ] && echo "  /v1/systemone: calls named qwen3.5-0.8b-sidecar route to the tiny engine"
[ "$SMALL" -eq 1 ] && echo "  /v1/systemone: calls named qwen3.5-2b-sidecar route to the small engine"
echo "  uninstall:               ./install-sidecar.sh --uninstall [tiny|small|all]"
