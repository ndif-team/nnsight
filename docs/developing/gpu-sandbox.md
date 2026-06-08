# GPU Sandbox — the chosen isolation implementation

**Status:** Built + tested (functional + safety pass) · **Date:** 2026-06-06
**Supersedes** the CPU-only transport work (harness-plan Phases 5/5b/6) for the GPU path.

## Why this design

The threat model was relaxed to **contain footguns, not defeat a determined adversary** — stop a careless
or buggy intervention from crashing the shared model server, reading host files, exfiltrating, hanging, or
OOMing it; do **not** (for now) defend against a malicious user weaponising the GPU. Under that model the
worker can keep **GPU access**, which makes the data path **zero-copy** and deletes the whole transport
problem (D2H + serialize) that made the CPU-only path cost ~110 ms/hook.

## Design

A **separate, spawned, GPU-enabled worker process** runs the user's arbitrary intervention on the real GPU
activation, while a process boundary + seccomp contain everything except the GPU.

- **Zero-copy via a shared GPU "bounce buffer."** A single CUDA tensor is shared host↔worker via CUDA IPC
  **once at spawn** (before lockdown). Per request the host does a cheap on-GPU (D2D) copy of the
  activation into the buffer + a 1-byte signal; the worker views the buffer as the activation tensor and
  runs the user op in place; the host reads the result back.

  **Measured (`perf_test.py`, A100): flat ~0.6–0.8 ms/hook, independent of size** — 0.59 ms at 0.03 MB,
  0.71 ms at 16.8 MB (~7B), 0.77 ms at 33.6 MB (~70B). The CPU-transport path scaled 1→285 ms with bytes;
  this doesn't, because the activation never leaves the GPU (per-request copy is an on-GPU D2D memcpy,
  ~0.01–0.04 ms). End-to-end on a gpt2 trace: **+1.1 ms/hook** (10.1 → 11.2 ms for one intervention; 10 → 24 ms for all 12
  layers). vs ~0.02 ms fully in-process.

  **Where the ~0.6 ms goes (decomposed, `perf_decompose.py` / `perf_ctxswitch.py`): GPU context-switching
  between the host and worker CUDA contexts — NOT process communication.** Measured: bare process round-trip
  0.04 ms, `cloudpickle.loads` 0.01 ms, one `cuda.synchronize` 0.016 ms. But a round-trip where BOTH host
  and worker do a GPU op (apply's pattern) = 0.405 ms vs 0.040 ms when only the host does → **~0.37 ms is the
  GPU switching active context host↔worker** each hook (each process has its own context on the shared
  device; they alternate every intervention).

  *Fix (measured under a scoped CUDA MPS daemon):* **CUDA MPS routes both processes' kernels through one
  shared GPU context**, eliminating the per-process switch. The both-touch-GPU round-trip dropped
  **0.405 → 0.132 ms** and the context-switch overhead **0.365 → 0.090 ms (~4×)** — confirming the diagnosis.
  So `apply()` under MPS goes from ~0.6 ms toward **~0.2 ms**, end-to-end toward ~+0.3 ms/hook. (A spin-waiting
  worker does **not** help — it only removes the ~0.04 ms IPC, not the context switch.)
  - **Fault-domain tradeoff — measured on the A100 (`fault_test.py`), better than feared:** MPS shares one
    GPU context, so in principle it couples the fault domain. In practice, **every GPU fault reachable from
    realistic buggy code was CONTAINED under MPS — the host's CUDA context survived**, same as with separate
    contexts. A buggy intervention's out-of-bounds index / embedding produces a **device-side assert** (PyTorch
    bounds-checks its indexing), and Volta+ MPS attributes that to the offending client and terminates only it.
    I could **not** trigger a genuine raw illegal-memory-access (XID 31) from normal tensor ops — they all
    became asserts. The residual coupling is the *rare* worst case (a true bad-pointer illegal access, or a
    hardware/ECC fault), which MPS docs say can hit the server. **Net:** MPS gives ~3× *and* keeps containment
    of the faults that actually happen — a viable default; separate contexts only buy the rare worst-case.
    | mode | per-hook | OOB→device-assert (the real footgun) | rare raw-illegal/HW fault |
    |---|---|---|---|
    | separate contexts | ~0.6 ms | contained (host survives) ✓ measured | contained |
    | Volta+ MPS (A100) | ~0.2 ms | **contained (host survives) ✓ measured** | shared blast radius |
  - *Why a bounce buffer and not per-request IPC:* `cudaIpcOpenMemHandle` (per-request rebuild) **fails
    after the seccomp lockdown** (`context is destroyed`). Mapping one buffer before lockdown sidesteps it;
    the per-request D2D copy is on-GPU memcpy (microseconds), not the bottleneck.
- **Footgun containment (`sandbox.py`):** after CUDA + torch are warmed, the worker installs a hand-rolled
  **seccomp-BPF** filter (no external dep) that makes new `open`/`openat`/`openat2` (filesystem),
  `socket`/`connect` (network), and `execve` fail with EPERM. CUDA keeps working — it uses already-open
  `/dev/nvidia*` fds via ioctl, not new file opens. GPU memory is capped with
  `torch.cuda.set_per_process_memory_fraction` (an OOM footgun can't exhaust the device). A per-call
  **timeout** kills a wedged worker; the **process boundary** isolates crashes and hides all host Python
  objects (weights, interleaver, other tenants).
  - *Gotcha:* `RLIMIT_AS` is unusable with CUDA — CUDA reserves tens of GB of *virtual* address space, so
    an 8 GB AS cap kills it at startup with `MemoryError`. Use the GPU memory fraction instead.
- **Pool.** CUDA isn't fork-safe → workers are spawned (not forked) and kept warm; each request goes to a
  free worker; a crashed worker is detected and replaced. (`GPUSandbox` is one worker for clarity.)

**Files:** `prototypes/mediator-sandbox/gpu_sandbox/` — `sandbox.py` (seccomp), `gpu_worker.py` (the
spawned worker), `gpu_sandbox.py` (host manager + `apply(activation, fn)`), `test_functional.py`,
`test_safety.py`. Run on a GPU: `CUDA_VISIBLE_DEVICES=N PYTHONPATH=<wt>/src <py> .../test_*.py`.

## Results

**Functional (`test_functional.py`) — PASS.** Real nnsight `model.trace()` workloads (scale, steer,
mean-ablate, per-token-norm read) on gpt2, run two ways — op inline vs op offloaded to the GPU worker —
produce **bit-identical** logits (`max|Δ|=0`); the worker survives all requests. nnsight delivers the
activation through its real machinery; only the user op runs isolated. (Core nnsight suite: 152 passed.)

**Safety (`test_safety.py`) — PASS (9/9).** Each mimicked unsafe request is contained:

| Unsafe request | Outcome |
|---|---|
| read a host secret file | `PermissionError` (open blocked); secret not returned |
| write a host file | `PermissionError`; no host file created |
| network egress (`socket`/`connect`) | `PermissionError` |
| reach a host object (`THE_MODEL.config`) | `NameError` — host objects don't exist in the worker |
| OOM (`torch.empty(1e12)` on GPU) | `OutOfMemoryError` — GPU mem fraction caps it |
| infinite loop | timeout → worker killed; host unaffected |
| hard crash (NULL-deref segfault) | worker dies; **host CUDA context intact**, crash surfaced cleanly |
| legit op after all attacks | still correct (worker survived the catchable ones) |
| fresh worker after a crash | serves correctly (pool recovery) |

## What this is vs. isn't

- **Contained:** filesystem, network, host crashes/OOM/hangs, and all host Python objects/other tenants'
  CPU-side state. The model server cannot be taken down or read by a buggy intervention.
- **Not contained (the accepted axis):** the GPU. The worker has a CUDA context and shares the activation,
  so a *deliberately malicious* user could attack the driver or read neighbouring GPU memory. Out of scope
  per the current threat model; revisit (CPU-only worker, MIG, dedicated allocations) if the model tightens.

## Remaining work

This is a working harness with an explicit offload API: `output = sandbox.apply(activation, fn)` inside a
trace. Wiring it **transparently into `model.trace()`** (so `.output`-access/ops/`.save()` route to the
worker automatically via the eproperty bridge) is the remaining productionisation — the same integration
as the transparent-trace-integration milestone, but simpler here because there is **no transport codec** (zero-copy GPU buffer).
Plus the warm-pool/scheduler and per-tenant dedicated allocations if the threat model tightens.
