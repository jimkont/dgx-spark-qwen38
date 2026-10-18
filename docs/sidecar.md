# Sidecar: a small Qwen for /v1/systemone

Issue #42 asks for an optional lightweight Qwen beside the main lane so one-token
Jev/System One decisions stop consuming the main model's engine and scheduler
capacity. SGLang since v0.5.21 serves `/v1/systemone` natively on its `/v1/decisions`
core, so a second small engine can answer the endpoint without any proxy-side
implementation: the sidecar is one SGLang container with a small model, and
clients point their `base_url` at its port.

Since v1.26.2 the sidecars are an install option: `./install.sh
--with-sidecar-tiny` and/or `--with-sidecar-small`, or `./install-sidecar.sh`
on its own with the same flags, plus `--quant nvfp4` to serve the pinned
AxionML NVFP4 exports instead of the BF16 weights and `--no-key` for an
unkeyed engine. Each is a systemd unit that coexists with the serving lane
instead of competing for its door: `qwen38-sidecar-tiny.service`
(Qwen3.5-0.8B, port 30011) and `qwen38-sidecar-small.service` (Qwen3.5-2B,
port 30010). `sidecar-guard.sh` is a separate watchdog for every container
named `qwen38-sidecar*` (the installed units and ad hoc launches alike); run
it whenever a sidecar is up: `nohup ./sidecar-guard.sh &`. `sidecar-launch.sh`
and `sidecar-memsample.sh` stay in the repo root as plain scripts for ad hoc
experimentation, with `SIDECAR_MODEL` and `SIDECAR_QUANT` selecting other
models or quantizations.

## Routing through the keepalive proxy

Since proxy v6.37 the keepalive proxy routes `/v1/systemone` by the request's
`model` field: a call named after an installed sidecar (e.g.
`qwen3.5-2b-sidecar`) goes whole to that sidecar's own `/v1/systemone` and its
answer comes back verbatim, while every other model (including `jev-latest`
and the lane's own name) still lands on the main lane. Sidecar traffic takes
no slot on the main lane's admission, so it can neither starve nor be starved
by the lane. The mapping lives in `~/.config/qwen38/sidecars` (one
`<served name> <base url>` line per sidecar, written by `install-sidecar.sh`,
which restarts the proxy after changing it). A sidecar that is not running
answers with the lane's own 503 `engine_unavailable`, naming the sidecar.
Clients can of course also point straight at a sidecar's port (30010/30011)
the way the benches did; the proxy route just lets one `base_url` serve both.

## Running it

```bash
# default: container qwen38-sidecar, served model qwen3.5-2b-sidecar,
# Qwen3.5-2B BF16, port 30010, 16K context, 8 slots
./sidecar-launch.sh
# a different model or quantization, no key:
SIDECAR_MODEL=Qwen/Qwen3.5-0.8B SIDECAR_NAME=qwen3.5-08b-sidecar ./sidecar-launch.sh
SIDECAR_QUANT=compressed-tensors SIDECAR_NO_KEY=1 ./sidecar-launch.sh
```

`SIDECAR_QUANT` is empty (BF16, the model's native weights) or a quantization
flag such as `compressed-tensors` for the AxionML NVFP4 exports;
`SIDECAR_NAME` sets the served model name clients request. `SIDECAR_NO_KEY=1`
skips the engine key, which is enough for local development. The engine runs
with a 16,384-token context (`SIDECAR_CTX`), `max-running-requests 8`,
`mem-fraction-static 0.60`, no CUDA graphs and no speculative decoding, so
its fixed footprint stays small. `sidecar-guard.sh` polls
`MemAvailable` every two seconds and stops every `qwen38-sidecar*`
container if the host falls under its floor, so a bad load cannot freeze
the box.

## Memory beside the main lane, measured

GB10 unified memory: 122.7 GiB total.

| configuration | weights | MemAvailable | sidecar cost |
|---|---:|---:|---:|
| 27B lane alone (this session) | ~17 GB NVFP4 | 49.9 GiB | — |
| 27B lane + 2B BF16 sidecar (this session) | 4.3 GB | 38.9–39.2 GiB | ~11 GiB |
| Flash lane + 2B NVFP4 sidecar (previous session) | 2.4 GB | 13.5 → ~2.8 GiB steady | ~10 GiB |
| Flash lane + 0.8B BF16 sidecar (previous session) | 1.7 GB | 13.5 → ~3.3 GiB steady | ~10 GiB |

Under bench load the troughs went deeper than the steady numbers above: with
the Flash lane up, MemAvailable bottomed at ~2.0 GiB while the 0.8B ran its
suite (11:23 UTC) and while the 2B NVFP4 ran its suite (12:24 UTC) — below the
the guard's 2.5 GiB kill floor, which was not running for those windows (the guard
started at 12:45 UTC, 15:45 local). The guard is what keeps that trough from
freezing the
box; run it whenever a sidecar is up beside a lane.

The sidecar's cost is not its weights: the CUDA context, runtime and KV pool
add several GiB on top. Beside the 27B lane it fits comfortably (~39 GiB
left). Beside the Flash lane it fits but leaves only ~2.8-3.3 GiB at steady
idle, and under bench load the box can dip to ~2 GiB: usable there, not
comfortable, and the guard is what keeps it safe.

## Decision quality on the pinned suite

Full `bench-systemone.py` suite, same pinned datasets, 5,422 items per
target. `s08b` = Qwen3.5-0.8B BF16, `s2bf16` = Qwen3.5-2B BF16,
`s2nv` = Qwen3.5-2B NVFP4, measured on the sidecar's native
`/v1/systemone`.

| task (n) | s08b | s2bf16 | s2nv | 27B lane (docs/systemone.md) |
|---|---:|---:|---:|---:|
| TypeSafe public cases, vs frontier reference (408) | 27.2% | 54.0% | 58.3% | 92.9% |
| MMLU-Pro MCQ (1,000) | 16.4% | 24.1% | 20.5% | 83.8% |
| BoolQ (3,270) | 71.9% | 80.8% | 73.9% | 91.9% |
| XNLI-fr (500) | 38.2% | 31.8% | 35.8% | 78.2% |
| MMMLU-fr (500) | 39.2% | 45.6% | 40.6% | 87.4% |
| GDPR, 13 questions × 5, vs reference | 35/50 | 50/50 | 50/50 | 50/50 |
| Injection, refused (36) | 52.8% | 72.2% | 69.4% | — |

On the judgment work the endpoint is sold for, the public decision cases,
both sidecars are far behind the main lane: the 2B NVFP4 answers 58.3% of the
408 questions the way the frontier reference does, the 2B BF16 answers 54.0%,
against 92.9% for the 27B lane and 93.2% for the hosted Jev. MMLU-Pro says
the same from the raw-readout side.

BF16 versus NVFP4 on the 2B is a split: the BF16 checkpoint beats the NVFP4
export on BoolQ (80.8 vs 73.9), MMLU-Pro (24.1 vs 20.5), MMMLU-fr (45.6 vs
40.6) and injection refusal (72.2 vs 69.4), while the NVFP4 export wins the
public Jev cases (58.3 vs 54.0) and XNLI-fr (35.8 vs 31.8), and both beat the
0.8B BF16 on nearly every column. On calibration the two also split: NVFP4
calibrates better on MMLU-Pro (ECE-10 0.234 vs 0.340) but BF16 is far better
on BoolQ (0.029 vs 0.054). The NVFP4 export remains the recommended sidecar
for this endpoint: it is stronger on the public decision cases the endpoint
is sold on, calibrates better on the raw-readout task, and runs three to four
times faster than the BF16 weights.

Latency per one-token call, p50: s08b 0.04–0.15 s, s2bf16 0.21–0.23 s,
s2nv 0.06–0.07 s, the 27B lane via the proxy 0.3–0.5 s, the Flash lane
~0.6 s per docs. The sidecars answer three to six times faster on the
short-context tasks, and because they are separate engines their requests do
not queue behind agent traffic. The public cases are the exception: their
states are long, so s2nv takes 0.79 s and s2bf16 2.0 s per call — at or
behind the 27B lane's 0.64 s per node.

## The two numbers, answered

1. **Memory beside the main lane.** The 2B sidecar costs ~10–11 GiB of the
   unified pool whether it is BF16 or NVFP4 (the weights differ by ~2 GiB,
   the engine overhead is the same). It coexists fine with the 27B lane;
   beside the Flash lane it fits inside the tier's 13.5 GiB floor but
   leaves ~2.8-3.3 GiB at steady idle, and ~2 GiB under bench load — below the
   guard's 2.5 GiB kill floor, which is what keeps that trough from
   freezing the box (run `sidecar-guard.sh` whenever a sidecar is up).
2. **Decision quality.** It does not hold up. On the same pinned suite the
   2B NVFP4 reaches 58.3% on TypeSafe's public decision cases against
   92.9% for the 27B lane and 93.2% for the hosted model, and 20.5% on
   MMLU-Pro against 83.8%; the 2B BF16 is no better on that task (54.0%)
   and its calibration is worse on the raw-readout side. BF16 does beat
   NVFP4 on BoolQ, MMLU-Pro accuracy, MMMLU-fr and injection refusal, but
   NVFP4 is stronger on the public cases the endpoint is sold on,
   calibrates better on MMLU-Pro, and runs three to four times faster.
   NVFP4 is the right quantization for the 2B, but no sidecar measured
   here can stand in for the main lane on Jev work.

## What that means for the option

Route `/v1/systemone` to the sidecar only for high-volume, low-stakes,
latency-sensitive decisions that tolerate ~60% agreement with the frontier
reference; keep judgment calls on the main lane. Both 0.8B and 2B, BF16 and
NVFP4, are in the local HF cache; `--with-sidecar-small` installs the 2B with
the recommended NVFP4 export, `--with-sidecar-tiny` the 0.8B with the BF16
weights its numbers were measured on. The installed units serve the
quantization they were installed with, so switching the 2B between BF16
and NVFP4 is one re-run of `./install-sidecar.sh --small [--quant nvfp4]`
(idempotent), and ad hoc swaps remain one
`SIDECAR_NAME=… SIDECAR_MODEL=… SIDECAR_QUANT=… ./sidecar-launch.sh`.

Not yet measured: the 0.8B NVFP4 run (in the cache, not benched) and the
Flash-lane decode/TTFT impact measurement, which needs the Flash lane
running again beside a sidecar.
