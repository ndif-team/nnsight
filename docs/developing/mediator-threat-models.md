# Mediator Isolation — Threat Models & Security Hardening

**Status:** Design notes / decision artifact · **Branch:** `worktree-mediator-sandbox`
**Builds on:** [mediator-gpu-trace-integration.md](mediator-gpu-trace-integration.md) (the GPU-worker
backend + §16 triton motivation), [mediator-isolation-sandbox.md](mediator-isolation-sandbox.md) (the
earlier CPU-only threat model + AWS deployment probe).

This page is the security companion to the GPU-worker backend: *what* adversary each isolation
configuration faces, what it can and cannot contain, and the resulting **cost coupling** that makes
the "how much isolation" question concrete. It also records the one security fix landed in-branch — the
worker→host **restricted-unpickler** codec.

---

## 1. Why isolation, and why now

`model.trace()` runs **arbitrary user intervention code**. NDIF runs that on shared GPU infrastructure,
today behind an **in-process Python import whitelist** that wraps the *entire forward pass*. That
boundary is architecturally unfixable in-process (the security suite shows 10/10 in-process escapes and
8/8 cross-tenant capability leaks succeed), **and** it cannot serve **triton-kernel models** — frontier
MoE/SSM models and anything under `torch.compile` JIT-compile triton, which needs
`subprocess`/ptxas/`tempfile`/`open` exactly when the whitelist denies them (see §16 of
[mediator-gpu-trace-integration.md](mediator-gpu-trace-integration.md)). The GPU-worker backend fixes
both at once: the **trusted forward runs host-side, unrestricted** (triton works); only the **untrusted
intervention** runs in a separate, sandboxable worker.

---

## 2. Attacker model

The attacker submits **serialized Python (cloudpickle)** that the server deserializes and executes
against a loaded model. They control: (i) the **pickle payload** (gadgets can fire at deserialize time)
and (ii) the **intervention function** — arbitrary Python at hook points. They do *not* directly control
the host process — only what they reach through the channels we expose. Intent tiers:

- **Footgun** — accident: infinite loop, OOM, bad index/device assert, accidental host mutation.
- **Casual adversary** — `import os`, open files, sockets, object-graph walks.
- **Determined adversary** — pickle gadgets, C-extension escapes, GPU-driver/kernel exploits,
  cross-tenant GPU reads, side channels.

---

## 3. Assets

| | Asset | Notes |
|---|---|---|
| **A** | Host-process integrity (no RCE on the model server) | the crown jewel |
| **B** | Host filesystem | model weights, other tenants' cached data, **secrets/IAM** |
| **C** | Network | exfiltration, internal services, **IMDS credential theft** |
| **D1** | Cross-tenant **host-memory** isolation | only under co-batching |
| **D2** | Cross-tenant **GPU-memory** isolation | the hard one |
| **E** | Availability / DoS | resource exhaustion, crash blast-radius |
| **F** | Deserialization RCE | attacker pickle firing **on the host** |

D1/D2 only bite when **multiple tenants share a forward pass** (co-batching).

---

## 4. Configuration ladder + the cost coupling

The deep insight: **the protection level and the data-path cost are coupled** — closing a deeper threat
*forces* the activation onto a slower path. You cannot pick "max isolation, min latency."

| Ladder | Threats newly closed | Data path | Per-hook cost |
|---|---|---|---|
| **R0 in-process** | — | tensor never moves | ~0 (≈12 ms whole gpt2 trace) |
| **R1 worker, shared GPU** | A host-process, **D1**, **F**, footgun-DoS | stays on GPU (CUDA-IPC) | **~0.6 ms, size-independent** + spawn (warm ~0.22 s) |
| **R2 + OS jail** (seccomp / net+mount+user ns + cgroups) | B fs, C net/IMDS, privilege, RAM-DoS | **same GPU buffer** | **~same ~0.6 ms** (jail is warm-time only) |
| **R3 + no GPU in worker** | **D2**, GPU-driver/kernel surface | **GPU→CPU→GPU (PCIe)** | **~ms, size-*dependent*** (1–285 ms) |
| **R4 + CPU-mem isolation** (gVisor/microVM) | kernel-proof, mem side-channels | **socket copy** (no shared mem) | R3 + serialize + pipe (slowest) |

Two consequences:
1. **The cost cliff is R2→R3, not R1→R2.** Adding the OS jail (closes fs/net/IMDS/privilege/RAM-DoS
   against even a determined adversary) rides the *same* GPU buffer and is ~free per hook. The latency
   explosion only happens when you leave the GPU to close **D2**.
2. **The decision reduces to one bit: do you require D2 (cross-tenant GPU-memory isolation)?** If no →
   **R2** is the target (full host-OS isolation, fast path, accept the GPU-layer residual). If yes →
   you must pay **R3** for the co-batched path.

---

## 5. Security controls vs the shared-GPU design

The GPU-worker's defining feature — host and worker **share a GPU + a CUDA-IPC memory buffer** — is
itself a cross-trust channel and requires exposing the NVIDIA driver surface to the worker. Controls
split by whether they touch that path. (Lifecycle rule for all of Group A: **map device fds + init CUDA
+ map the IPC buffer + warm, THEN jail** — fds/buffer survive; mirrors the existing
lock-down-after-warm.)

**Group A — compatible (the free wins, take these):**
- **Network namespace** (no NIC) — zero GPU impact; kills exfil + IMDS. Take unconditionally.
- **Mount namespace** (ro allowlist; bind only `/dev/nvidia*` + CUDA libs) — removes weights/secrets/
  other-tenant disk from view. Highest value.
- **User namespace** (non-root, drop caps); **cgroups** (CPU/mem → OOM/forkbomb cap); **seccomp** (have it).

**Group B — in tension / incompatible (forces the CPU-transport fallback or hardware partition):**
- **PID/IPC namespace** — cross-namespace CUDA-IPC is documented to fail; `CLONE_NEWPID` must be entered
  before CUDA init, colliding with map-first. Likely cannot PID-isolate while sharing the buffer.
- **gVisor** — `nvproxy` implements a narrow NVIDIA ioctl subset; CUDA-IPC unconfirmed → likely drops to
  CPU transport.
- **Firecracker microVM** — GPU only via exclusive passthrough; can't share one GPU.
- **MIG** — the only real GPU-memory isolation, but breaks IPC sharing by construction (and unavailable
  on the T4 prod GPU).

**Residual that no OS control closes:** even with all of Group A, the worker holds a CUDA context on a
GPU also hosting the model + (under batching) other tenants' KV-cache → GPU/driver-layer cross-tenant
surface (driver bugs, IPC-handle reachability, memory-residency side channels). Only **R3 (no GPU in
worker)** or **MIG** removes it.

---

## 6. Co-batch tenant isolation

With co-batching in scope, "can a co-batched user reach another's data / entire workflow?" resolves as
follows (verified in code).

**Row-bounding is host-authoritative.** `batch_group` is computed host-side at invoke construction
(`tracing/invoker.py:81`); the host narrows reads to `self.batch_group` (`interleaver.py:1245`) and
bounds writes (`:1259`/`:1294`). The worker only names a `(path, kind, iteration)` string — it cannot
widen. Each user's intervention runs in its own worker **process** (separate address space).

**The one data-plane hole.** An **empty invoke** gets `batch_group=None`, and `narrow(None)` returns the
**entire batch** while `swap(None,…)` overwrites it (`batching.py:214/217/244/248` — `None`, `[-1]`, and
`not needs_batching` are all full-batch sentinels). So a `batch_group=None` mediator is a full-batch
read+write primitive across all co-batched tenants — the in-process suite's "`swap(None)` poisons all
tenants" leak.

**Enforce, don't guard.** `batch_group` is 100% host-computed (no user-supplied field), so this is a
**system invariant**, not input validation — but the full-batch path is reachable via the user's
*program structure* (an empty invoke is a legit single-tenant feature). The correct enforcement is
architectural: **the Batcher/Interleaver instance is the tenant boundary** — co-batch *below* the nnsight
Batcher (at the inference engine, e.g. vLLM continuous batching), never *inside* one shared Batcher.
Under that invariant all three sentinels are tenant-local and safe by construction, and
`cross_invoker`/`barrier` (per-Interleaver `_xinvoke_store`/`_barrier_acc`) close with it. The vLLM path
gets this for free (independent per-request interleavers); an HF-merged-batch path must enforce it. A
runtime reject-`None` check is defense-in-depth against our own merge bug, not the primary control.

**What a co-batched user CAN / CANNOT do.** *Cannot* (with the empty-invoke invariant + normal CUDA):
read another tenant's rows/activations (narrowed out, never sent), their prompt/code/output (separate
process), or their **live** GPU memory by pointer (separate CUDA contexts are VA/page-table isolated). So
reconstructing another user's **entire workflow is structurally closed**. *Can*, on the shared GPU
(R1/R2): scoop **uninitialized/freed GPU memory** (untargeted *fragments*, not workflows) + coarse
timing side channels — reducible on the fast path via **zero-on-free memory scrubbing** + alloc caps +
driver capability segmentation, without going to R3.

---

## 7. Worker→host RCE — finding and fix (landed in-branch)

**Finding.** The host read worker frames with `mp.Pipe.recv()` (`transport.py`), which **unpickles**.
The worker runs attacker code, so it could send a crafted pickle and get **RCE on the trusted host** —
which bypasses *every* other layer (seccomp, namespaces, row-bounding). The design fixed deserialization
for the *inbound* user payload (unpickled inside the worker) but not the *outbound* results. This is a
precondition for the isolation guarantee holding against a determined adversary.

**Fix.** Worker→host now never plain-`pickle.loads`. Tensors already travel out-of-band (GPU bounce
buffer / safetensors), so a frame is **tensor-free**; the remaining small structure is decoded with a
**restricted unpickler** (`transport._RestrictedUnpickler` / `_safe_loads`) whose `find_class` allows
**only torch `dtype`/`device`** and refuses every other class/function. `find_class` resolves a global
*before* the `REDUCE` that would call it, so a gadget (`os.system`, …) is refused before it can execute.
Supporting choices keep the allowlist minimal: the event rides as its string `.value` (no enum class),
and exceptions ride as a `(type-name, message)` sentinel (no class). **Host→worker stays normal pickle**
(host-authored, trusted).

- **Why a restricted unpickler over a hand-rolled codec:** pickle handles all plain nested Python
  structures natively (no per-type enumeration — an earlier hand-rolled JSON codec silently *missed* the
  `Events.CACHE` event's `torch.dtype`/`torch.device` payload). Anything un-allowlisted now fails **loud
  at decode with the exact refused class name**, easy to extend if legitimately safe.
- **Cost:** ~plain-pickle speed (µs for the tiny tensor-free frame); `find_class` fires only on the
  handful of globals per frame. Negligible vs the ~0.6 ms/hook round-trip.
- **Why not `torch.load(weights_only=True)`:** it had an RCE bypass (CVE-2025-32434, fixed only in 2.6);
  our unpickler is *tighter* — it enables no tensor-rebuild path at all (tensors aren't in the pickle).
- **Capability narrowing (documented):** `.save()` of an arbitrary object / numpy array / framework type
  (e.g. `ModelOutput`) is no longer transmittable from a worker — save a tensor (or basic data) instead.

Coverage: `prototypes/mediator-sandbox/gpu_sandbox/test_isolated_codec_security.py` — fidelity for
VALUE/SWAP/END/**CACHE (dtype+device)**/EXCEPTION/push, and a real `__reduce__` gadget refused at decode
without executing. (CPU is enough; needs torch.) The legacy AF_UNIX socket channels
(`SocketHostChannel`/`ShmSocketHostChannel`) are unused by `isolate_mediators` and still plain-unpickle —
route them through `_safe_loads` before wiring them to an untrusted worker.

---

## 8. Open items / next steps

- **Group-A namespace jail** behind a pluggable `Jailer` (applied post-warm/pre-`ready`) — the
  highest-value, design-preserving hardening; R2 on the fast path.
- **Co-batch invariant** in the (future) HF-merged-batch integration: keep the nnsight Batcher
  per-tenant; add a multi-tenant tripwire rejecting unbounded `batch_group`. (Single-trace=single-Batcher
  already holds it today; an unconditional guard would break legit single-user empty invokes.)
- **GPU memory scrubbing** (zero-on-free) for the shared-GPU residual.
- **gVisor + CUDA-IPC empirical test** — decides whether any R3-strong path keeps GPU sharing.
- **Run the codec + isolated suite** on a torch/GPU box to confirm fidelity + bit-identical regression.
