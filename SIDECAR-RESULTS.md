# Sidecar experiment: Qwen3.5-0.8B / 2B for /v1/systemone (issue #42)

Status: **final.** All four arms re-benched on 2026-10-12 with the box free
(lane unloaded; 5,422 items per arm, 0 error rows), after the provisional
2026-10-11 runs. The CPU arm reproduced the provisional accuracies exactly;
the GPU arms moved by at most 1.6 points. Memory fit was also re-verified
beside the shipped flash-next concurrency tier: **on that install neither the
GPU sidecar nor the CPU sidecar fits** (§1.1, §5).

Measured on this box, Flash-Next lane at the tier as shipped on 2026-10-10/11
(~13.5 GiB host MemAvailable at its floor, docs/flash-lane.md), one SGLang engine per
`sidecar-launch.sh` container (port 30010, 16K context, no CUDA graphs, no
speculation, `--disable-flashinfer-autotune`, page-size 64, its own
`/v1/systemone` from the pinned sglang). The CPU arm is llama.cpp
`llama-server` (:8099) + `sidecar-cpu-shim.py` (:8098) reproducing the same
`/v1/systemone` protocol — SGLang has no CPU build, §3. Bench:
`bench-systemone.py`, the same pinned datasets as the lane's published
numbers (docs/systemone.md), one-token readout, concurrency 4.
`sidecar-guard.sh` (MemAvailable watchdog) and `sidecar-memsample.sh`
protected the serving lane the whole time; `sidecar-bench-loop.sh` resumes a
bench task by item id after any abort and is re-runnable per arm.

The final pass ran each GPU sidecar alone on the free box with
`sidecar-final-run.sh` (data in `bench-final-data/`); the accuracy, latency
and alone-on-a-free-box footprint numbers in §1.2, §2 and §4 are from it.
The §1 table and §1.1 are the earlier lane-contended measurements.

The two numbers that decide it, from the maintainer:

## 1. Memory beside Flash-Next (GPU sidecar), 2026-10-10/11 — fits but leaves no margin

_(Measured against the pre-update lane install. The re-verified answer for
the shipped concurrency tier is §1.1: a second engine cannot boot there.)_

| sidecar | weights | boot transient (MemAvailable) | steady floor | steady footprint |
|---|---|---|---|---|
| 0.8B BF16 | 1.75 GB | dip to ~3.5 GiB, <1 min | ~3.9 GiB | ~8.5–9 GiB total |
| 2B BF16 | 4.5 GB | **13 → under 3.25 GiB for ~90 s**, recovers | ~8.5 GiB (idle window) | ~10 GiB total |
| 2B NVFP4 | 2.58 GB | dip to ~6 GiB | ~3.1–3.5 GiB | ~9.2 GiB total |

- The floor that matters is the **transient**, not the steady state: a 2B BF16
  boot beside a live lane goes under 3.25 GiB — exactly where live agent
  sessions started aborting (this session included). Two guarded boots died
  there; the unguarded one recovered, which is what the sampler shows.
- Steady-state cost is dominated by the second engine itself: a CUDA context
  plus kernels and pools costs ~6–7 GiB *before any weights*. That is why the
  0.8B floor (~3.9) is barely better than the 2B NVFP4 floor (~3.1–3.5): the
  weight saving shows up in the transient, not in steady state.
- A ~3.2 GiB floor on a box whose measured freeze zone starts right there is
  not a margin. With any spike on the Flash lane (a big prefill, a swap
  pressure, this box's 25–40 GB untracked allocations, docs/gb10-memory.md)
  the machine stalls. earlyoom is inactive on this box; the guard is the only
  protection, and it works by killing the sidecar, not by protecting the
  lane's spikes.
- Flash KV capacity is unchanged (the sidecar boots after the lane, the pool
  is sized at lane launch). The impact on Flash decode/TTFT is the memory
  pressure itself: with the 2B sidecar up, live agent traffic on the lane
  aborted intermittently.

### 1.1 Re-verified beside the shipped concurrency tier (2026-10-12): the GPU sidecar cannot boot

The lane was reloaded today at the docs/flash-lane.md shipped tier
(`concurrency`: 176B hybrid MoE, `--mem-fraction-static 0.85`, 262,144-token
window, 8 requests / 40 mamba slots, NEXTN, PLE table on NVMe). Baseline:
MemAvailable 14.15–14.30 GiB (2 s samples, n=29), consistent with the
~13.5 GiB floor in the table above. Then, same `sidecar-launch.sh` shape:

- **3/3 boot attempts of 2B NVFP4 died 30–40 s in, before any weight load**,
  at `handle_gpu_memory_settings → get_device_memory_capacity →
  cuDevicePrimaryCtxRetain → CUDA_ERROR_OUT_OF_MEMORY`. The sidecar RSS
  peaked at 1.02 GiB; host MemAvailable never dipped below 13.16 GiB — the
  host still had headroom; the device did not.
- The device reports only **1.38–1.70 GB free of the 130.66 GB (121.7 GiB)
  unified pool** while the lane is up. A second SGLang engine needs ~6–7 GiB
  of CUDA context before any weights (bullet above), so on the shipped tier
  the engine cannot even start. Host MemAvailable (14 GiB) overstates what a
  second CUDA context can use: the lane's static pool (0.85 of device free at
  its boot) is sized from *device* memory and keeps it.
- Ruled out as the cause: the sidecar's `--memory 12g` cgroup cap (a bare
  torch context init succeeds in the identical container shape with and
  without the cap), lane load (it fails under a conc-6 request storm and
  with the lane quiet), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`.
- On the shipped tier the repo's one-engine-at-a-time rule (`Conflicts=`
  units, sglang#40948, engine-preflight.sh) is therefore a measured
  constraint, not just policy: at `concurrency` there is no room beside the
  lane for any second CUDA engine, 2B or otherwise.

### 1.2 The sidecars alone on a free box (2026-10-12, final pass)

Lane unloaded, box at 118.5 GiB MemAvailable before each boot. Each arm
booted through `sidecar-launch.sh`, ran the full suite, sampled every ~2–3 s:

| sidecar | ready after | MemAvailable, steady median | lowest during run | footprint (baseline − median / − lowest) | container RSS max (`docker stats`) |
|---|---:|---:|---:|---:|---:|
| 0.8B BF16 | 35 s | 109.9 GiB | 108.5 GiB | 8.6–9.9 GiB | 5.10 GiB |
| 2B BF16 | 45 s | 107.3 GiB | 105.4 GiB | 11.2–13.1 GiB | 8.68 GiB |
| 2B NVFP4 | 45 s | 109.2 GiB | 107.9 GiB | 9.3–10.6 GiB | 5.96 GiB |

- These agree with the 2026-10-10/11 steady footprints in the §1 table
  (~8.5–9 / ~10 / ~9.2 GiB); the 2B BF16 comes out higher (11–13 GiB).
  Footprint here is the host-side MemAvailable drop, which includes the page
  cache of the loaded weights; the container RSS column is smaller than the
  host drop for every arm, so part of a CUDA engine's cost does not show in
  its cgroup.
- On a free box all three are cheap. The problem in §1/§1.1 is only that the
  lane leaves no device memory for them.

## 2. Decision quality — the bar is not met by any of them

`bench-systemone.py` full suite (5,422 items per arm) against the lane's
published numbers (docs/systemone.md table). One-token readout, no
temperature fit. CPU arm = Q8_0 GGUF 2B on llama.cpp.

| task | hosted Jev | lane (Flash) raw / 2 orders | 0.8B BF16 | 2B BF16 | 2B NVFP4 | 2B Q8_0 CPU |
|---|---:|---|---:|---:|---:|---:|
| BoolQ 3,270 | 91.9% | 86.7% / 89.3% | 72.0% | 80.7% | 73.9% | 80.2% |
| MMLU-Pro 1,000 | 83.8% | 58.1% / 62.1% | **16.4%** | 24.2% | 20.5% | 25.3% |
| XNLI-fr 500 | 78.2% | 71.2% / 71.4% | 39.0% | 31.4% | 35.8% | 31.4% |
| MMMLU-fr 500 | 87.4% | 73.0% / 74.6% | 38.8% | 45.4% | 40.6% | 45.6% |
| TypeSafe public, 408 q vs references | 93.2% | 90.6–90.9% / 92.9% | **26.9%** | 53.4% | **59.9%** | **68.0%** |
| inject, 36 (18 clean / 18 injected) | — | — | 72.2 / 33.3 | 94.4 / 50.0 | 100.0 / 38.9 | 83.3 / 55.6 |
| GDPR 13×5 (54k-char article, batched) | 50/50 | 50/50 | 35/50 | 50/50 | 50/50 | **30/50** |

- The 0.8B is a coin flip on multi-option questions (MMLU-Pro 16.4% with up to
  10 options ≈ chance) and answers 27% of TypeSafe's public reference
  questions. "Much smaller footprint" yes; "usable" no.
- The 2B closes the gap on BoolQ-ish judgment (2B BF16 80.7%, 2B NVFP4 73.9%)
  but on the task the endpoint exists for — TypeSafe's 20 public cases against
  the frontier references — it lands at 53–60% versus the lane's 92.9%. That is
  not "answer for the big one"; it is a different, weaker decision-maker.
- The CPU Q8_0 sits in the same weaker band, and on the multi-option tasks it
  actually beats the 4-bit GPU arms (TypeSafe 68.0 vs 59.9, MMLU-Pro 25.3 vs
  20.5): an effective 8-bit is closer to BF16 than NVFP4 is. But it loses the
  one thing both 2B GPU arms kept on the GDPR article: 30/50, answering
  "false" to 4 of 10 questions (the 72-hour breach notification, non-EU
  applicability, right to erasure, data portability) — wrong on all 5 repeats
  each, so consistent boundary flips, not noise.
- BF16 vs NVFP4 for the 2B: no consistent winner (public 53.4 vs 59.9 at
  n=309, not significant; MMLU-Pro 24.2 vs 20.5; BoolQ 80.7 vs 73.9, where
  BF16 is ahead). On a free box NVFP4 is ~1.5× faster per item; only NVFP4
  could ever boot beside a live lane (BF16 dipped under the freeze line in
  the §1 runs). If a 2B sidecar is ever kept, NVFP4 is the only viable form.
- Injection robustness is a non-result at n=18: every arm follows the
  injected instruction on 8–10 of 18 (0.8B 10, 2B BF16 9, 2B NVFP4 10,
  CPU 8). No fence, like the model card says.

## 3. SGLang on CPU: not possible, measured

- Pinned image `qwen38-pinned:flash-cb6ed363800a`:
  `torch 2.15.0.dev20260818+cu134`, `torch.cuda.is_available() == False`,
  `import sgl_kernel` → `ImportError: [sgl_kernel] CRITICAL: Could not load
  any common_ops library!` (missing `libcuda.so.1`).
- The PyPI `sgl_kernel-0.3.21` wheel (the only kernel distribution,
  manylinux aarch64): all 9 shared objects link against CUDA
  (`libcuda.so.1`, NCCL/mscclpp). There is no CPU build of the kernel, so
  the `/v1/systemone` endpoint — which scores raw full-vocabulary label
  masses — cannot be served by SGLang on CPU.
- The CPU arm was therefore benched by replicating SGLang's protocol on top
  of llama.cpp: `sidecar-cpu-shim.py` (:8098) over `llama-server` (:8099),
  Q8_0 2B GGUF, 32K context, `-nplog 30`. The shim reproduces the content
  rendering (choice/score/noul), the label-mass scoring (raw softmax mass per
  label token, temperature 1), and the answer shapes byte-for-byte; rendered
  prompt token counts match the GPU engine exactly (141,421 tokens on the
  GDPR item, identical for both engines). Ollama was ruled out first:
  `top_logprobs` capped at 20 and no per-token logprobs on `/api/generate`.

## 4. Latency

p50 / p95 per item, s, final pass (GPU arms alone on the free box; mean input
tokens per item in the last column):

| task | 0.8B BF16 | 2B BF16 | 2B NVFP4 | 2B Q8_0 CPU | tok/item |
|---|---:|---:|---:|---:|---:|
| public (~9 q/item) | 0.50 / 19.6 | 0.85 / 14.4 | 0.59 / 18.6 | 46.0 / 434.7 | 41,828 |
| boolq | 0.04 / 0.05 | 0.08 / 0.09 | 0.06 / 0.06 | 3.71 / 4.91 | 181 |
| mmlu-pro | 0.04 / 0.06 | 0.09 / 0.11 | 0.06 / 0.07 | 4.80 / 6.83 | 251 |
| xnli-fr | 0.04 / 0.04 | 0.08 / 0.09 | 0.06 / 0.06 | 3.69 / 3.93 | 176 |
| mmmlu-fr | 0.04 / 0.05 | 0.08 / 0.10 | 0.06 / 0.07 | 3.65 / 6.23 | 190 |
| inject | 0.04 / 0.12 | 0.05 / 0.16 | 0.06 / 0.16 | 2.02 / 2.45 | 84 |
| gdpr (item) | 0.09 / 0.50 | 0.12 / 0.63 | 0.12 / 0.52 | 11.2 / 51.8 | 20,203 |
| gdpr (13-q batched call, mean) | 0.37 | 0.45 | 0.38 | 54.1 | 141,421 |

- A GPU sidecar alone answers a one-token decision in 0.04–0.09 s p50 and the
  whole 13-question GDPR batch (141k tokens) in 0.37–0.45 s. Beside the live
  lane in the provisional run the same arms were slower (2B BF16 BoolQ p50
  0.21 s vs 0.08 s now; 2B NVFP4 0.063 vs 0.056 s). The Flash lane on the same
  small payloads measured 0.13–0.35 s warm (published p50 0.43–0.62 s; not
  re-measured in this pass, lane unloaded), so a sidecar is roughly 2–9×
  faster on one-token decisions — but it is faster at giving the wrong answer.
- CPU is 35–93× slower than 2B NVFP4 on item p50 (inject 2.02 s vs 0.057 s,
  MMLU-Pro 4.80 s vs 0.059 s, GDPR item 11.2 s vs 0.121 s) and ~140× on the
  GDPR batch (54.1 s vs 0.38 s). Prefill throughput on long prompts is
  2,600–3,570 tok/s (141,421 tokens in 54.1 s in this run, 42.2 s in the
  provisional run; a warm 10,855-token prompt in 3.04 s), and per-request
  latency has a ~2.7 s floor even for 181-token items (BoolQ p01 2.69 s) — a
  fixed per-request cost dominates short items on this stack. TypeSafe items
  (41.8k tokens each) run 46 s p50, tailing to 552 s at concurrency 4;
  multi-question items work out to ~11.5 s / ~12.5 s per question (TypeSafe
  4,689 s ÷ 408 q; GDPR 1,621 s ÷ 130 q).

## 5. Memory: the CPU arm dodges the CUDA context, not the freeze line

- No CUDA context, no second engine pool: `llama-server` RSS grew on demand
  to **11.17 GiB** (32K context; the hybrid model's KV/SSM state), the shim
  sits at 0.03 GiB.
- The full suite ran beside the **27B lane**: `MemAvailable` stayed at
  **40.0–48.6 GiB** for the whole 2.6 h (1,502 samples, 5 s cadence) — 30+
  GiB above the ~3.25 GiB freeze line. Beside that lane the CPU arm is free
  of risk.
- Re-verified 2026-10-12 beside the **flash-next concurrency tier** (host
  MemAvailable 13–14 GiB): inject ran 36/36 in 18 s (0 errors), then 100
  BoolQ items in 112 s (0 errors) took RSS 2.54 → 9.90 GiB and MemAvailable
  13.24 → **5.47 GiB in 3.8 min**, where a manual 6 GiB watchdog cut the
  engine. Projecting the full-suite RSS (11.17 GiB) the floor lands near
  **4 GiB — inside the freeze zone**. **The CPU arm does not fit beside the
  shipped flash tier at 32K context / concurrency 4.**
- So the CPU sidecar's safety is lane-config dependent, not CUDA-dependent:
  free beside the 27B lane, unsafe beside the shipped `concurrency` tier. A
  smaller llama.cpp context (less KV/SSM growth) or a lane tier that leaves
  more room could make it fit — not measured.
- Final pass on the free box (lane unloaded): over 2.09 h and 1,502 samples
  `MemAvailable` ranged 109.1–117.7 GiB, `llama-server` RSS peaked at
  **11.18 GiB** and the shim at 0.031 GiB, reproducing the 11.17 GiB maximum
  of the provisional suite. The CPU arm costs ~11 GiB whatever the lane is,
  which is why it fit beside the 27B lane and not beside the flash tier's
  ~14 GiB of headroom.

## Verdict

- 0.8B is too weak to enter: MMLU-Pro 16.4% (≈ chance on up to 10 options),
  26.9% on TypeSafe's public cases.
- On 2026-10-10/11 a 2B NVFP4 GPU sidecar **fit** beside Flash-Next and was
  **fast** (alone on a free box: 0.06 s p50, ~2–9× the lane on one-token
  decisions), but it did not meet the decision quality bar on the pinned
  suite: 53–60% vs 92.9% on TypeSafe's public cases, 20–24% vs 58–62% on
  MMLU-Pro.
- A 2B CPU sidecar is quality-equivalent to the GPU 2B arms (68.0% / 25.3%
  there — better than the 4-bit GPU arm — but 30/50 on the GDPR article, where
  both 2B GPU arms scored 50/50) and is 35–93× slower per item.
- **The 2026-10-12 re-verification voids the fit of both sidecar forms on the
  shipped concurrency tier**: the GPU sidecar cannot even attach (1.38–1.70 GB
  device free; 3/3 CUDA-OOM boot failures, §1.1), and the CPU arm's 11 GiB
  RSS eats the tier's 14 GiB host headroom down into the freeze zone (§5).
- Neither answers for the lane on `/v1/systemone` as currently defined —
  the 2B family sits at 53–68% vs 92.9% on TypeSafe's public cases regardless
  of form. If the endpoint is ever scoped down to very cheap routing-class
  calls, a sidecar only works on a lane tier that leaves room for it (2B
  NVFP4, 8–16K context, no graphs, hard MemAvailable floor, explicit "cheap
  tier, different quality" label — not a silent swap), or as a CPU fallback
  where the lane's headroom allows. On the shipped `concurrency` tier the
  measured answer to issue #42 is: there is no room.
