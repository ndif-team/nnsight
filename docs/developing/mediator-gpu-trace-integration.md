# Transparent GPU-Worker Trace Integration — Design

**Status:** Design (approved) · **Date:** 2026-06-06 · **Branch:** `worktree-mediator-sandbox`
**Builds on:**
- [gpu-sandbox.md](gpu-sandbox.md) — the chosen GPU-worker mechanism (`GPUSandbox.apply`, bounce buffer, seccomp, lifecycle).
- [mediator-isolation-harness-plan.md](mediator-isolation-harness-plan.md) — the earlier CPU-transport prototypes; this doc supersedes the transparent-`model.trace()` milestone described there.
- [mediator-isolation-sandbox.md](mediator-isolation-sandbox.md) — threat model (relaxed to "contain footguns, not adversaries") + AWS deployment.

**Goal:** make `model.trace()` run user interventions in an isolated GPU worker process **transparently** —
identical results to in-process — so a footgun in user code (infinite loop, OOM alloc, bad-index
device assert, host-object poke) is contained to the worker while the shared model server keeps serving.
The standalone `GPUSandbox.apply()` proved the worker *mechanics*; this turns it into the actual
`model.trace()` execution backend.

---

## 1. The architecture: an outer harness, not a rewrite

The Mediator↔model handoff already routes through the `MediatorChannel` seam
(`interleaver.py:774`, extracted earlier). The Mediator talks to the model **only** through that channel,
via the six events (`VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION`). **Those events do not change.** Isolation is
an outer harness: run the Mediator's worker in a separate (GPU-enabled, locked-down) process, and give it
a cross-process channel.

```
HOST (trusted: model + weights + GPU)               WORKER (spawned, GPU-enabled, seccomp'd)
  real Envoy tree + Interleaver + Batcher             path-only Envoy mirror (no weights)
  host-side Mediator                                  worker-side Mediator (same class, other half)
    · owns channel, batch_group, history, handle_*       · runs the REAL user intervention fn
    · registers hooks ON DEMAND                           · eproperty.__get__ → request() → blocks
  CudaIpcChannel host end  ◄── control pipe ──►        CudaIpcChannel worker end
        │                  + shared GPU buffer                  │
  forward fires hook → handle() → narrow →            user code runs on the GPU tensor view,
    D2D into buffer → respond                          `.save()` accumulates locally,
                                                       shipped back at END
```

**One `Mediator` class, instantiated on both sides**, each using its half. The existing
`Mediator.__getstate__`/`__setstate__` (`interleaver.py:1558`) already define exactly what crosses to
construct the worker side — they were built for NDIF remote. The host half owns the Batcher and
`handle_*`; the worker half owns `send`/`request`/`swap`/`push` and runs the intervention.

### What does NOT change
- The six-event protocol and its one-event-in-flight invariant (the "access in forward-pass order"
  contract).
- `Mediator.send`/`respond`/`handle`/`handle_*`, the Batcher narrow/swap, requester↔provider matching.
- `eproperty`/`hooks.py` core logic (the worker's hook call lands on a dead dummy module — see below).
- The in-process path: when isolation is off, `Mediator.start` runs the existing `Thread` exactly as today.

### What is genuinely new (and *why*)
Two places in the in-process code rely on **shared memory** rather than the channel. These — and only
these — are why isolation is more than "spawn + new channel." Both live in the harness layer.

1. **Hook registration touches the real module via shared memory.** In-process, `eproperty.__get__`
   registers a PyTorch hook *on the actual module object* (`hooks.py:307`, `output_hook`). The worker
   process has no module. But the event already carries the requester string
   (`"transformer.h.6.output.i0"`), which fully encodes (path, kind, iteration) — so the **host**
   registers the hook on receipt (the *host-side on-demand hook registration* described in §3).
2. **`.save()` propagates through a shared Python frame.** In-process, trace exit reads the worker's
   frame locals, keeps `id(v) in Globals.saves`, and `push_variables` into the user frame — all
   same-process pointer work (`tracing/base.py:565`). Across processes the worker's frame and
   `Globals.saves` live in the worker, so it runs that filter locally and ships the result over the
   channel; the host does the final `push_variables` (the *worker→host saves transmission* in §4).

---

## 2. Transport — `CudaIpcChannel`

A new `MediatorChannel` implementation, same shape as `transport.py`'s shared-memory socket channel; only
the tensor encode/decode differs.

- **Control plane** (`mp.Pipe`, small pickle frames): `(event, requester)` + an *offset table*
  `[(offset, nbytes, shape, dtype), …]` + a non-tensor skeleton. Reuses `transport.py`'s
  `_split_tensors`/`_merge_tensors` verbatim — the only change is "encode each tensor as a buffer
  offset" instead of "safetensors bytes."
- **Bulk plane** (shared GPU bounce buffer): the sender D2D-copies each tensor into the buffer at
  successive offsets; the receiver views them at those offsets. The buffer is a CUDA tensor created on
  the host and shared to the spawned worker via `torch.multiprocessing` (CUDA IPC), mapped once before
  lockdown.
- **Aliasing rule (correctness, mandatory):** under one-event-in-flight the single buffer is reused in
  strict alternation, so **both sides clone-on-receive** before the next access can overwrite it. This
  covers `x = h[6].output; y = h[7].output; use(x, y)` — `x` must be cloned out of the buffer before the
  request for `y` overwrites it. A D2D clone is HBM-speed (~µs/MB); "zero-copy" always meant *no PCIe,
  no pickle*, not *no copy*. The measured ~0.6 ms/hook already includes these copies.
- **Sizing:** a fixed arena (64 MB default). A value larger than the arena raises a clear error (the
  `ShmArena.write` "exceeds arena" guard). Chunking is a later perf item, out of scope.
- **CPU-only fallback backend:** the existing CPU shared-memory socket channel stays config-selectable
  for CPU-only isolation — free, since it is already a `MediatorChannel`. GPU-IPC is the default.

Two roles, mirroring `transport.py`: `CudaIpcHostChannel` (main thread; only `wait_event` reads the
pipe; `has_event`/`get_event`/`restore_event` are host-local buffer ops) and `CudaIpcWorkerChannel`
(send→wait→get).

---

## 3. Host-side on-demand hook registration

In-process, the worker's `requires_output` registers the hook just-in-time as it single-steps. In
isolation the worker can't (no real module), so the host registers it.

- **Where:** in the host `handle` loop, when a `VALUE`/`SWAP`/`SKIP` arrives for a requester `R` whose
  provider is not yet set up (the existing `handle_value_event` else-branch, `interleaver.py:1203`,
  that today does `history.add`/`restore_event`/`return False`).
- **What:** `ensure_isolated_provider(R)`: parse `R` (`"<path>.<output|input>.i<N>"`) → `iteration = N`,
  `kind`, `path`; resolve `envoy = root_envoy.get(path)` (`envoy.py:586`); call the **existing**
  `output_hook`/`input_hook(host_mediator, envoy._module, f"{path}.{kind}")` (`hooks.py:224`/`154`).
  The host-side mediator is passed, so the hook closure delivers over the channel.
- **Worker side:** the worker's `eproperty.__get__` still runs `self._hook(obj)`. The worker's
  path-only envoy carries a bare `torch.nn.Module()` as `_module`, so `output_hook` registers a hook on
  a **dead dummy** that never fires (harmless; reclaimed when the worker process exits — NOT by
  `cancel`, since the worker never calls it). **Warm-pool caveat:** a warm worker pool reuses processes,
  so these dummy-module hooks (and the worker-local `iteration_tracker`/`Globals.saves`) would accumulate
  across traces and need explicit per-trace cleanup. Keeping `eproperty`/`hooks.py` unchanged was the
  reason to register on a dead dummy rather than guard `_hook` with an `interleaver.isolated` flag;
  revisit if dummy-module accumulation shows up in profiling.
- **Idempotency:** `ensure_isolated_provider` registers a given `R` once per mediator (tracked in a set on the
  host mediator), mirroring the `current_provider`-skip logic in `requires_output`.

For a single forward pass the iteration is always `0`, so `R` is `"<path>.<kind>.i0"`. Multi-token
iteration stamping is handled where the multi-token feature is built (§9).

---

## 4. Worker→host saves transmission

- **Worker:** at trace exit / on `END`, the worker runs the *existing* `base.py` exit filter (it owns
  the frame + `Globals.saves`) to produce the saved dict, then — instead of `push_variables` locally —
  ships that dict over the channel.
- **Host:** receives the dict and `push_variables(user_frame, saved_dict)` into the real user frame
  (the frame that called `model.trace()`, which lives on the host).
- **Encoding:** tensors in the saved dict ride the bounce buffer (offset table); other saved objects
  ride the control pickle, reusing `_split_tensors`.
- **New constraint vs in-process:** a *non-tensor* `.save()` must be serializable to cross the boundary
  (in-process it was just a reference). Tensors (the overwhelming common case) are fine; a
  non-serializable save raises a clear error naming the variable. Documented as an isolation semantic
  difference.

### Isolation semantic: in-place modification requires explicit assignment
In-process, `.output` returns the model's *real* tensor (shared memory), so an in-place edit
(`h[6].output[0][:] = ...`) propagates by mutation with no SWAP event. In the isolated path the worker
receives a **clone** (the clone-on-receive rule, §2), so an in-place edit mutates a worker-local copy and
does **not** reach the host. Use **explicit assignment** (`h[6].output = ...`), which emits a SWAP event
and applies via `batcher.swap`. This is identical to **NDIF remote** semantics (in-place on a serialized
clone never crossed the wire either), so it is a consistency property, not a new wart. Transparent
in-place support would need a protocol read-back that conflicts with the single-buffer clone-on-receive
rule; deferred (revisit with double-buffering if real workloads need it).

---

## 5. Lifecycle parity

- **`cancel()`** terminates the worker process (the `gpu_sandbox` `close()`/`terminate()` pattern)
  instead of dropping a `Thread`. `check_dangling_mediators` and the EarlyStop drain account for
  process workers.
- **Timeout:** a per-call timeout kills a wedged worker (infinite loop in user code). Surfaces a clear
  `TimeoutError`; the host is unaffected; a warm pool would respawn.
- **EXCEPTION:** the worker ships the exception over the channel; the host re-raises it in the user's
  context with the rebuilt traceback. Dynamic nnsight exceptions that can't pickle are degraded to a
  plain exception preserving type name + message before they cross (§11).
- **SKIP / BARRIER / END / EarlyStop:** already cross as events. `END` triggers the saves transmission
  then worker exit + host join. `stop()`'s `push()` ships saves before signalling.

---

## 6. Feature coverage map

Folded into the **support matrix (§8)**, which carries each feature's cross-process mechanism and
current status in one table. The invariant it tracks: when isolation is on and a not-yet-supported
feature is used, the trace **fails cleanly** — a missed-provider error or the per-step timeout (the
lifecycle is the safety net), not a silent deadlock or silent-wrong result. (There is no automatic
"route to in-process" fallback; features are added one at a time.)

---

## 7. Core single-pass interventions — scope, components, acceptance

This is the first slice: a single forward pass (iteration 0), covering read / swap / `.save()` /
multi-invoke / skip / exception / lifecycle / seccomp lockdown — the foundation the rest builds on.

### Scope
Single forward pass; single and multiple invokes; read (`.output`/`.input`/`.inputs`); swap (`=`); tensor
`.save()`; skip; exception; lifecycle (normal END, exception, timeout→kill); tuple/nested block outputs.
`cross_invoker` disabled. Multi-token / barrier / backward / cache not yet handled.

### Components
1. **`CudaIpcChannel`** (`transport.py`, additive): `CudaIpcHostChannel` + `CudaIpcWorkerChannel` +
   GPU-buffer tensor pack/unpack (reusing `_split_tensors`/`_merge_tensors`).
2. **Isolated `Mediator.start`** (`interleaver.py`): a branch that spawns the worker (CUDA → `spawn`),
   ships `self` via source-serialization with the model→path-only-envoy persistent-object map, and a
   worker bootstrap that builds the interleaver stub + path-only envoy mirror + runs the intervention.
3. **Host-side on-demand hook registration** (`interleaver.py`): `ensure_isolated_provider(R)` in `handle`.
4. **Saves transmission + lifecycle** (`interleaver.py` + bootstrap): worker ships `Globals.saves`-
   filtered frame locals at END; host injects; `cancel()` kills the process.
5. **Opt-in** (§12): `CONFIG.APP.ISOLATE_MEDIATORS` + `nnsight.isolate_mediators()` context.

### Acceptance — bit-identical to in-process (`max|Δ|=0` on GPU)
1. read-save; swap; multi-invoke batch-narrow; nested-tuple block output.
2. **A non-standard-named toy model** (`decoder_blocks`, `output_projection`) **and** gpt2 — per the
   testing rules: structure derived at runtime via `envoy.get(path)`, no GPT-2-only assumptions, no
   hardcoded module-name tables.
3. exception in user code → raised in the user's context with the correct traceback.
4. infinite loop in user code → worker killed, host survives, clear `TimeoutError`.
5. safety: the unsafe payloads (filesystem/network/exec/OOM/crash) contained under seccomp lockdown.
6. core suite still green (in-process path untouched).

### Status: DONE (2026-06-06)
Built and verified (TDD; harnesses in `prototypes/mediator-sandbox/gpu_sandbox/`):
- **Code:** `transport.py` (`pack_cuda`/`unpack_cuda` + the two channel ends, clone-on-receive, per-wait
  timeout); `isolation.py` (`isolate_mediators` + `_spawn_worker` + `_pool_worker_main` +
  `ensure_isolated_provider` + `_WorkerInterleaver`/`_WorkerPersistent`); `_sandbox.py` (seccomp
  `lock_down`, relocated from the prototype); `interleaver.py` seam (`_iso` field, isolated
  `Mediator.start` branch, the on-demand hook-registration call in `handle`, saves injection in
  `handle_end_event`, `_iso.close()` in `cancel`).
- **Foundations (probes, all PASS):** buffer codec round-trip + clone-on-receive; CUDA-IPC channel over
  a real spawned process; bootstrap (serialize→deserialize-against-dummies→run→correct `request`);
  saves capture (`info.frame.f_locals` filtered by `Globals.saves`, keyed by user var name).
- **End-to-end (`test_isolated_trace.py`):** isolated read `.output[0].save()` and explicit-assignment
  swap (`h[6].output = … * 2`) **bit-identical to in-process, `max|Δ|=0`** on gpt2/cuda.
- **Acceptance (`test_isolated_acceptance.py`):** non-standard module names (`decoder_blocks`,
  `NNsight(net)`) `max|Δ|=0`; two isolated invokes bit-identical + rows distinct (no cross-leak); user
  exception → `NNsightException` in the user context; infinite loop killed (timeout), host survives.
- **Safety (`test_isolated_lockdown_safety.py`):** read under `lockdown=True` `max|Δ|=0`; `open()`
  blocked + no host file; `socket()/connect()` blocked.
- **Regression:** core in-process suite **155 passed** (in-process path untouched).

**Findings folded into the design:** per-mediator serialization must set `intervention.__source__`
first (the tracer normally does it in *its* `__getstate__`); the worker frame is a `SerializedFrame` so
`push()` lands locals there; saved values inject into the **tracer's** `info.frame` (the user frame),
not the mediator's; `_remoteable_persistent_objects` is remoteable-only (gate with `isinstance`);
locked-down workers must `os._exit(0)` to skip tempfile atexit (avoids a `RecursionError`). `lockdown`
defaults off in `_STATE` (flip to on once the warm pool lands); the bounce-buffer force-kill emits a
benign CudaIPC release warning.

---

## 8. Support matrix (what works under `isolate_mediators()` today, and how)

> **Fast lane (2026-06-13, [fast-lane.md](fast-lane.md)).** `isolate_mediators()` now runs each
> mediator on one of three tiers: a static classifier confirms safe interventions and runs them
> **in-process** (FAST — full model + weights), isolates the unconfirmable remainder in the GPU worker
> (ISOLATE), and rejects introspection escapes (REJECT). This is what lets the **weight-reading interp
> majority** (logit lens, steering, ablation, activation patching, attribution) run under isolation at
> all — the worker holds weightless dummy modules, so those workloads can only run on the fast lane.
> The matrix below describes the **ISOLATE (worker) tier**; a row marked weight-reading/host-module is
> served by the FAST tier. Default-on; `fast_lane=False` forces the worker tier.

| Feature | Cross-process mechanism | Status |
|---|---|---|
| read / swap (`=`) / `.save()` (tensors) / skip / exception | six events over the channel; host-side hook registration; worker→host saves transmission at END | ✅ bit-identical |
| multi-invoke + batch narrowing | per-invoke worker + host mediator; Batcher stays host-side | ✅ bit-identical |
| single-forward `generate(...)` (no iter) | same as trace | ✅ verified |
| seccomp lockdown (fs/net/exec) | `_sandbox.lock_down` after the first job's deserialize, in `_run_one_job` | ✓ a read under `lockdown=True` is `max|Δ|=0`, user-code `open`/`socket`/`exec` are blocked, and a warm pool keeps serving (`test_isolated_lockdown_safety.py` 4/4). The break — the worker locked down BEFORE receiving its first job, so unpickling the job message (its `transformers` lazy submodules load only at unpickle time) hit a seccomp-blocked `open` (an `OSError` the recv loop swallowed → silent death) — is fixed by deferring lockdown to after the (host-authored, trusted) first job is deserialized, before its user code. One-way and installed once: in a warm pool, a later job needing a *new* model's modules still needs `preimport=` (a homogeneous model needs nothing). |
| `iter`/`all`/`next` (multi-token) | step stamped in the requester; host iter-hooks bump the tracker; live `default_all` piggyback (§9) | ✅ bit-identical (`iter[N]`, `iter[:]`, per-step swap) |
| `tracer.barrier()` | worker sends the target count; host accumulates participants + runs the coordination loop (§10) | ✅ |
| `cross_invoker` variable sharing | host variable store; worker pushes data locals, pulls the merged store; transmittable data only (§10) | ✅ |
| warm worker pool (`pool_size=N`, `warm_worker_pool`) | generic workers receive serialized mediators as jobs; clean-END recycle (§14) | ✅ ~21× faster per request once warm |
| `with tensor.backward()` / `.grad` | BACKWARD event: worker seeds `dL/d(delivered clone)`, host continues `torch.autograd.grad` on the real graph, `.grad` by provenance path; **grad-through-swap** iterates the exchange — host returns `dL/d(swap leaf)`, worker backprops it through its swap tape, re-seeds the pre-swap graph (§16) | ✅ read-path AND grad-through-swap bit-identical (scalar loss; single invoke); multi-token backward is a clean-fail (in-process doesn't support it either) |
| `tracer.cache()` (`modules=`, `include_inputs=`) | CACHE event → host registers the real cache hooks; host CacheDict swapped in at END, filled in-place by the forward (§15) | ✅ bit-identical |
| part-2 primitives: `tracer.unembed` / `tracer.steer` / `tracer.patch` / `tracer.ablate` | host-routed weight read (UNEMBED event); replacement-swap injection/transplant/knockout (ride SWAP, no new event) — [fast-lane.md](fast-lane.md) §6 | ✅ bit-identical (isolated and in-process) |
| session cross-trace handoff (`.save()` used in a later trace; `.carry()` / `nnsight.carry(x)`) | inner-trace END writeback to the session frame: saved values re-registered host-side so the session exit-push keeps them; carried (non-saved) values written for the next trace only — [fast-lane.md](fast-lane.md) §6 | ✅ bit-identical (`.carry()` is portable: harmless in-process, load-bearing under isolation) |
| Triton-kernel models (MoE / SSM / `torch.compile`) | host-side forward compiles/runs Triton unrestricted (the worker holds only the intervention) | ✅ — §17 |
| user-code Triton (kernel inside the intervention) | — (compiling a kernel needs `open`/`subprocess`/ptxas, which lockdown blocks by design) | ⛔ under lockdown — §17 |
| `.source` operation-level access (`...attn.split_1.output`) | — (op paths aren't in `model.modules()`) | 🔜 not yet |
| in-place `[:]=` | — (clone-on-receive semantics; use explicit `=`, §4) | ⛔ |
| MPS / `isolate_mediators()` further polish | — | planned |

Not-yet-supported features fail **cleanly** (missed-provider error or the per-step timeout), not as a
silent deadlock — the lifecycle (timeout + `finally: cancel()`) is the safety net until each feature lands.

---

## 9. Multi-token iteration — STATUS: DONE (2026-06-07)

Multi-token iteration (`iter[N]`, `iter[:]`, `next`) isolated == in-process. The iter loop runs in the
worker (sets `mediator.iteration` explicitly per step → correct requester suffixes); the **host** side
gained three pieces:
- **Per-step hook registration:** `ensure_isolated_provider` parses the step `N` from the requester and
  passes `iteration=N` to `output_hook`/`input_hook` (new optional param), so the hook fires on step N —
  not the host mediator's iteration.
- **Host iter-hooks:** `_spawn_worker` calls `register_iter_hooks(host_mediator, real_model)` so
  the host `iteration_tracker` advances per forward.
- **Live host→worker piggyback:** `default_all` (= `generate(max_new_tokens)`) is set *after* the worker
  spawns, so it's piggybacked on each response frame (`CudaIpcHostChannel.meta_provider` →
  `CudaIpcWorkerChannel.on_meta`) and applied to the worker's interleaver stub before its `iter[:]` loop
  computes its bound. This is a general host→worker state channel reused later for the variable store.

**Verified (`test_isolated_multitoken_iter.py`):** `iter[N]` for N∈{0,1,2} `max|Δ|=0`; per-step swap
propagates (`iter[1]`); `iter[:]` accumulating into `nnsight.save(hs)` bit-identical (3 steps).
**Regression:** channel echo PASS (3-tuple response format); core in-process suite (incl.
`test_iter_edge_cases`) 113 passed; isolated single-pass trace still `max|Δ|=0`.

**Findings:** `default_all` is live host state set post-spawn → can't snapshot at spawn (hence the
piggyback). The worker over-requests one step past the end only if `default_all` is missing (now fixed).
In-place `[:]=` per step still requires explicit `=` (§4).

---

## 10. Cross-invoke (barrier + variable sharing) — STATUS: DONE (2026-06-07)

Both isolated == in-process.

- **Barrier (host-side counting):** each invoke runs in its own worker with its own `Barrier` copy, so it
  can't count cross-invoke. `Barrier.__call__` (isolated) sends the TARGET count; the host accumulates
  participant names in `Interleaver._barrier_acc` and, once all arrive, runs the existing coordination
  loop (`handle_barrier_event` iterates the host mediators, `respond()`+`handle()` each over its own
  channel). `Mediator._isolated_worker` (set in `_pool_worker_main`) gates the worker-side behavior.
- **Variable sharing (host store):** worker frames aren't shared across processes, so each worker pushes
  its *data* locals to `Interleaver._xinvoke_store` (a 4th `push` field on the event frame) and pulls the
  merged store back (piggybacked on the response, reusing the multi-token channel). Only **transmittable
  data** (tensors + basic scalars/containers, `_transmittable`) crosses — framework objects
  (`Barrier`/`Envoy`, which hold the model) are skipped (the worker already has them via its closure).
  Tensors travel **CPU-serialized** (D2H on push, H2D on pull): a worker tensor cloned from the CUDA-IPC
  buffer can't be re-shared over IPC by the host ("CUDA tensor received from another process").
  cross_invoker is gated exactly like the host (`len(mediators)>1 and CONFIG.APP.CROSS_INVOKER`).

**Verified (`test_isolated_cross_invoke.py`):** two-invoke `tracer.barrier(2)` produces matching saved
values; the canonical cross-invoke pattern (invoke A captures `h[3].output`, barrier, invoke B sets
`h[3].output = captured`) is bit-identical. **Non-standard names (`test_nonstd.py`):** gpt2 with
`rename={transformer.h→decoder_blocks, lm_head→output_projection}`, read + iter[1] isolated `max|Δ|=0`.
(Note: `Envoy.__getattr__` resolves the alias to the *real* path before the requester is built, so the
wire string is always the real path — this validates renamed models run end-to-end, but does not itself
exercise an alias↔real mismatch in the host hook registration, since none reaches the wire.) Echo
regression PASS (4-tuple event).

**Findings:** the variable-sharing push must filter to transmittable data (else the `Barrier` local pulls
in the whole model) AND move tensors to CPU (CUDA-IPC tensors can't be re-shared by the host). The
`_xinvoke_store` is per-interleaver (reset each trace) so no cross-trace leak today; the warm pool (§14)
rebuilds the interleaver + dummy modules per job and clears `Globals.saves`/`Globals.shared`, so there is
no cross-trace leak between *unrelated* traces under reuse either. (Intentional cross-trace handoff
*within a `model.session()`* — a `.save()`d value used in a later trace, or `.carry()` — is a separate,
supported path: the worker ships those values to the session frame at END; see the §8 support matrix and
[fast-lane.md](fast-lane.md) §6.) Known coverage gaps: no tests yet for
multi-barrier-in-one-trace, 3+ participants, multi-token + barrier, or variable sharing without a barrier;
the store grows monotonically per trace and ships all shared tensors CPU-serialized on every response (a
perf cliff for large cross-invoke tensors).

---

## 11. Backward + caching — characterization (both gaps since closed: cache §15, backward §16)

A gap-characterization harness (`test_isolated_backward_cache_gaps.py`, since retired — canonical
coverage lives in `test_isolated_cache.py` / `test_isolated_backward.py`) confirmed the two gaps and
their difficulty at the time; kept for the reasoning record:

- **`tracer.cache()` (a real build, not a quick shim):** `tracer.cache()` runs in the worker → registers
  cache hooks on dummy modules → never fire → the `.save()`'d CacheDict comes back **empty**. The fix
  needs: (1) a `CACHE` event carrying the spec (module paths + options) so the host registers
  `cache_output_hook`/`cache_input_hook` on the *real* modules; (2) **post-forward injection** — the
  catch is that cache hooks are **persistent** and the CacheDict is populated **by the forward, which runs
  AFTER the mediator's intervention ends** (the intervention only sets up the cache, then ends), so the
  END-time saves transmission runs too early. The populated host cache must be copied into the user
  frame's CacheDict **at trace exit** — specifically in `Interleaver.cancel` BEFORE `remove_hooks` (the
  host cache hooks survive `mediator.cancel`; they're only dropped in `Interleaver.cancel`'s finally,
  after the forward has populated them). Matching each worker CacheDict to its host cache needs a token (a
  counter set on the CacheDict + carried in the CACHE event; order-based matching is fragile and the token
  attr must survive the CacheDict's pickling). A clean build, but ~4 touch points + its own mechanism.
  (Acceptance test `test_isolated_cache.py` written, currently failing as expected.)
- **`with tensor.backward()` (hard — likely a major build or a documented limitation):** the backward
  context runs in the worker on **detached clones** (clone-on-receive strips the autograd graph, which is
  host-side), so there is no graph to differentiate. Making it work requires the backward pass to run
  **host-side** (where the graph is) with **path-based grad providers** (today they are `id(tensor)`,
  process-local) — architecturally similar to the forward's host-side hook registration but for the
  backward session, comparable in size to the core single-pass seam. Until then, backward under isolation
  must fail cleanly / run non-isolated.
- **Robustness bug found + fixed (affects all features):** a *dynamic* `NNsightException` raised inside
  the worker fails to pickle across the EXCEPTION event (`PicklingError: Can't pickle
  nnsight.NNsightException`). Plain exceptions (e.g. a user `ValueError`) cross fine; nnsight-internal
  dynamic exceptions did not. The worker now degrades a non-picklable exception to a plain
  `RuntimeError(type name + message)` before the EXCEPTION event.

---

## 12. Opt-in surface
`CONFIG.APP.ISOLATE_MEDIATORS` (flag) + `with nnsight.isolate_mediators(): ...` (context). `Mediator.start`
checks it to select the isolated branch. The channel backend (GPU-IPC default, CPU shared-memory
fallback) is config-selectable. Server deployments set the flag globally.

---

## 13. Risks
- **Per-hook latency** — the accepted cost; ~0.6 ms measured, size-independent (vs CPU pickle 1→285 ms).
- **Codec correctness for nested/tuple values** — needs the `applyn`-aware pack tested against
  tuple-output blocks (covered by the nested-tuple acceptance case).
- **Lifecycle/deadlock** — one-event-in-flight must hold over the pipe exactly as over the queue; the
  clone-on-receive rule must not be skipped, or held-across-access tensors corrupt silently.
- **Path-only envoy fidelity** — the worker mirror must resolve every path the user writes; built from
  the serialized tree, validated by the non-standard-named-model acceptance test.

---

## 14. Warm worker pool — DONE (2026-06-07)

**Why.** Spawning a worker per request is the dominant isolation cost: **~4.5 s** end-to-end on gpt2/A100
(~12 ms in-process, ~370×), of which **~4.2 s** is cold `import torch` (1.3 s) + `import nnsight` (2.3 s) +
CUDA context init (0.4 s) — measured (`perf_spawn_cost.py`), **model-independent** (weights aren't shipped),
a flat per-request tax. Host-side mediator serialization is only ~3 ms. A warm pool amortizes the spawn.

**The key change: the worker is generic, not mediator-bound.** Previously `_worker_main(payload, ...)`
received its mediator as a spawn-time argument. Now `_pool_worker_main(conn, buf, base_opts)` warms CUDA +
imports + `_ensure_mounted` **once**, optionally locks down, sends a one-time `"ready"` ack, then loops:
`conn.recv()` → on `("job", payload, extras, opts)` clear `Globals.saves`, deserialize a fresh mediator
against fresh dummies (`_run_one_job`), run it, loop; on `"stop"` `os._exit(0)`. The CUDA context, warmed
kernels, bounce buffer, and `CudaIpcWorkerChannel` persist across jobs; only the ~3 ms payload changes per
request. This **unifies** the cold and pooled paths — the worker always runs the loop; the host decides
recycle-vs-kill. The one-time `"ready"` is consumed by the spawner before the channel reads protocol frames
(they share the pipe).

**Host side.** A process-global `_WorkerPool` (thread-safe) persists across traces. `acquire_isolated_worker`
serializes the mediator (`_build_job`), pulls an idle worker (or lazily grows to the `pool_size` cap, or a
cold one-shot worker past the cap so a trace never blocks on the budget), ships the job, and re-points the
host channel's `meta_provider`/`on_push` to *this* mediator's interleaver. `Mediator.cancel` calls
`release_isolated_worker`.

**Recycle-safety rule.** Recycle **only a cleanly-ended worker**: `handle_end_event` sets `_iso.clean=True`
when a worker's END is consumed; `release` recycles iff `clean and poolable and alive and not dirty`. A
worker drained mid-protocol with a `Cancelation` (`dirty` — the pipe is now unbalanced), a timeout/death
(`clean` never set — it's *spinning*, not idle), or a cold one-shot worker is **retired** (killed) and the
pool re-warms lazily. Recycle resets the worker's host channel (`CudaIpcHostChannel.reset`) and per-job
hook-registration state; the worker rebuilds its interleaver + dummy modules per job and clears
`Globals.saves`, so there is no cross-trace state leak.

**Opt-in.** `isolate_mediators(..., pool_size=N)` routes through the pool (`pool_size=0`, the default, is
the unchanged cold-spawn path). `warm_worker_pool(N, ...)` pre-warms at startup (blocks until N ack ready);
`shutdown_worker_pool()` tears it down. Workers are pooled **per (device, arena_bytes, gpu_mem_fraction,
lockdown, preimport) signature** (the `IsoOptions.pool_key` 5-tuple; `preimport` and `lockdown` are
warm-time, so they partition the pool); a worker is reused only for a matching signature, so a process hosting models on
different GPUs gets a per-device sub-pool (NOT a shared pool whose bounce buffer is fixed to the first
model's device — that would copy into the wrong-device buffer). Per-trace options (`default_all`,
`cross_invoker`, `timeout`) ride each job.

**Pool sizing is a GPU-memory budget.** The natural ceiling is the **batch size** (one worker per mediator,
one mediator per invoke, all concurrent vs one forward pass → concurrent workers = #invokes ≤ batch size).
Each warm worker costs **~0.55 GiB GPU per GPU it touches** (CUDA context + cuBLAS kernels; measured,
model-weight-independent, linear in worker count) — and **MPS does not reduce this** (Ampere MPS shares the
*scheduler*, not context memory; measured identical under MPS). So at batch-16/single-GPU ≈ 8.7 GB (11% of
an 80 GB A100 but ~55% of a 16 GB T4) — the cap must be deliberate, with the cold-spawn fallback past it.
(`probe_pool_gpu_footprint.py`.) Note this ~0.55 GiB is the fixed CUDA-context cost and is distinct from
`set_per_process_memory_fraction`: that knob caps the worker's **allocator pool** (so a runaway allocation
can't exhaust the device, the 20 GB-symptom footgun), not the context, so it does not reduce the per-worker
0.55 GiB.

**Lockdown + pool.** Seccomp lockdown is installed once, in `_run_one_job`, after the **first** job's
(host-authored, trusted) payload is deserialized and before its user code runs — so deserialization's own
imports (e.g. transformers' lazy modeling submodules, loaded only at unpickle time) succeed. It is one-way:
in a warm pool, later jobs run under the first job's lockdown, so a later job whose deserialize needs a
*new* import fails — true only across *different* models (a homogeneous model is already imported); pre-load
the deployment's model set via `preimport=` to serve heterogeneous models under lockdown. The cold path
(`pool_size=0`) is the same code, so it behaves identically (deserialize the one job, then lock down).
Lockdown defaults off.

**Hardening (independent review, 2026-06-08).** Both passes confirmed no Critical issue — the cross-request
*data* invariant holds (per-job fresh interleaver/dummies/`Globals.saves`; host channel reset before
re-bind). Fixes applied: (1) `acquire` skips/forgets workers that **died while idle** and re-spawns, instead
of handing out a dead worker; (2) the pool is **keyed per device-signature** (above); (3) an **EXCEPTION-
ended worker is recycled** (it's alive + the pipe is balanced), not retired — so erroring traces keep the
pool benefit; (4) a recycled worker's first event uses `timeout + margin`, not the cold 180 s, preserving
hang-containment; (5) the grow slot is **reserved under the lock** so concurrent acquires can't exceed the
cap; (6) `close()` releases the pipe fd + GPU buffer; (7) a `_shutting_down` flag stops a shutdown/release
race from orphaning a worker.

**Verified (`test_isolated_pool.py`, gpt2/A100):** reuse bit-identical (`max|Δ|=0`) at **~21× faster** once
warm (4.57 s cold → 0.22 s warm) with worker PIDs reused (no fresh spawn); a 3-invoke trace draws 3 distinct
pooled workers all bit-identical; a timed-out (infinite-loop) worker is retired and the pool re-warms with
the next trace bit-identical; a killed idle worker is skipped + replaced on the next acquire; a non-standard-
named model works through the pool. Cold path (`pool_size=0`) stays bit-identical across
read/swap/save/multi/exception/hang/multitoken/cross-invoke/barrier/nonstd.

---

## 15. `tracer.cache()` — DONE (2026-06-08)

**The gap.** `tracer.cache()` registers **persistent** hooks (`mediator_idx=inf`) that fill a `.save()`'d
`CacheDict` *during the forward*. In the worker those hooks land on the **dummy** modules and never fire, so
the user got an empty `CacheDict` (the `.save()`'d placeholder shipped back unfilled).

**The fix — a `CACHE` event + host-side registration, with the post-forward injection collapsed.**
- **Worker `cache()` (isolated branch in `tracer.py`):** instead of registering dummy hooks, ship the spec
  `(token, module-paths, device, dtype, detach, include_output, include_inputs, rename, alias)` via a new
  `Events.CACHE` request (`mediator.send`). Return a **token-tagged** placeholder `CacheDict` the user binds
  and `.save()`s — that is what carries the user's variable name across the boundary.
- **Host `handle_cache_event`:** resolve the paths to the **real** envoys, register the real
  `cache_output_hook`/`cache_input_hook` into a host `Cache` keyed by the token (on `Mediator._iso_caches`),
  `set_user_cache`, ack. Hooks live on the host mediator's `hooks`, removed at teardown by `remove_hooks` —
  exactly like in-process.
- **The timing insight (no separate teardown step).** `handle_cache_event` acks and returns `True`, so the
  host loop processes `CACHE` then `END` consecutively at `Mediator.start` — *before* the forward. So
  `handle_end_event` swaps the **host** `CacheDict` reference in for the worker's empty placeholder (matched
  by the token on the saved value) when it injects the saves. The forward then fills *that same object*
  in-place. The user's variable **is** the forward-filled host cache; the doc's earlier "token-matched
  post-forward injection" collapses to "swap in the host CacheDict at END; the forward fills it." This also
  preserves in-process semantics, including "a cache defined after a module is called misses it" (the host
  hooks register only when the `CACHE` event arrives, just like the in-process registration point).

**Touch points:** `Events.CACHE`; `tracer.cache()` isolated branch; `handle()` dispatch + `handle_cache_event`;
the `handle_end_event` token-swap (gated on `_iso_caches`, so non-cache traces are untouched); `_iso_caches`
on `Mediator`.

**Verified (`test_isolated_cache.py`, gpt2/A100):** single module, multi-module (3 keys), and
`include_inputs=True` all bit-identical (`max|Δ|=0`, keys match in-process). Cold path + the full isolated
regression (trace/acceptance/multitoken/cross-invoke/pool) unchanged.

**Not covered:** a cache placeholder nested inside a container save (`got = [t.cache().save()]`) — the swap
matches a top-level saved `CacheDict`; nesting would need a recursive walk. `cross_invoker` + cache is
untested. `modules=None` (cache *all* modules) registers a hook per module on the host — correct but heavy.

---

## 16. `with tensor.backward()` — read-path DONE (2026-06-10), grad-through-swap DONE (2026-06-23)

**The gap (§11).** Clone-on-receive strips `grad_fn` — the worker's delivered activations are detached
clones, the autograd graph lives only on the host, and `.grad` providers were keyed by `id(tensor)`
(process-local). So a backward block in the worker had nothing to differentiate.

**The fix — split the chain rule at the process boundary** (distributed-autograd-style stitch), keyed by
**requester string** (module path + kind + step), not `id(tensor)`:

- **Worker, forward time:** when the trace contains a backward block (detected from the intervention
  source), every delivered activation clone is tagged — `requires_grad_(True)` + an `id(clone)` →
  requester-string provenance map — via a wrapped `mediator.request` (`_tag_delivered`,
  `isolation.py`). Worker-side ops on delivered values therefore build the worker's local half of the
  graph (delivered leaves → loss).
- **Worker, backward time:** `BackwardsTracer.execute` detects the isolated context
  (`worker_backward_context()`) and runs `_execute_isolated` (`tracing/backwards.py`): it computes the
  worker half of the chain rule — `dL/d(delivered clone)` for each tagged leaf the loss depends on
  (`torch.autograd.grad`, `allow_unused`) — and ships the seed dict `{requester: grad}` via a new
  `Events.BACKWARD`.
- **Host:** `handle_value_event` retains the REAL on-graph activation per requester (gated on
  `_iso_backward` so non-backward traces pay nothing); `handle_backward_event` continues the chain rule
  on the host's real graph (`torch.autograd.grad` seeded by the worker grads, targets = all retained
  activations, `allow_unused`, `retain_graph`) and returns `{requester: dL/d(activation)}`. No user code
  runs on the host.
- **Worker, block body:** runs locally; each `.grad` read is served from the returned dict by the
  tensor's provenance path (a patched `Tensor.grad` property). `.grad` on a user-derived tensor (no
  provenance) raises a clear error; the `.grad` setter raises `NotImplementedError`.

**Stitch correctness (no double-counting):** the worker seeds are partials treating delivered leaves as
independent; the host's autograd adds the indirect inter-layer contributions. Verified empirically — a
loss using the same read activation twice is bit-identical to in-process.

**Verified (`test_isolated_backward.py`, gpt2/A100):** grad of `ln_f.output` `max|Δ|=0` vs in-process;
renamed model (`final_norm`/`output_projection`) `max|Δ|=0`; user-derived-tensor `.grad` → clear error.
Independent review (7 finder angles, dedup + verify): **no silent-wrong in the in-scope path** (single
invoke, on-path tensor-output target, scalar loss, no swaps).

**Gradient-through-swap — DONE (2026-06-23).** An isolated SWAP installs a worker-computed value as a host
*leaf* (clone-on-receive strips `grad_fn`), so the host backward used to dead-end at the swap — while
in-process gradients flow through swaps. Now the seam is stitched by iterating the existing `Events.BACKWARD`
exchange to a fixpoint. Host: `handle_swap_event` (under `_iso_backward`) makes the swap leaf
`requires_grad_(True)` and retains it (`_iso_grad_swaps`), so the downstream forward tracks it and it is a
backward target; `handle_backward_event` adds swap leaves to its targets and returns `dL/d(swap leaf)` under
a reserved key. Worker: `WorkerMediator.swap` keeps the worker-tape swap value (with `grad_fn`); the backward
block loops — send seeds, receive `dL/d(swap leaf)`, backprop it through the swap tape to `dL/d(delivered
clone)`, re-seed the pre-swap graph, repeat — accumulating each read's gradient across rounds (a clone
reached both directly and through a swap sums both paths). With no swaps the loop is the original single
exchange. Chained swaps converge in N rounds. **Verified (`test_isolated_grad_through_swap.py`, gpt2 +
renamed):** grad through `h*2`, `h+vec`, `tracer.steer`, and TWO chained swaps, plus the renamed model, all
isolated-vs-in-process `max|Δ|=0`.

**Limits / open:**
- Scalar loss only — `loss.backward(gradient=...)` is not honored (scalar-only error).
- Batched traces error with a cryptic shape mismatch (host retains the full-batch tensor, worker seeds
  the narrowed clone) — needs a clear error or narrowed retention.
- Multi-token backward: **not supported in-process either** (characterized 2026-06-10,
  `test_isolated_multitoken_backward.py`) — `generate()` runs the forward without gradient tracking, so
  the first `.grad` read fails in-process ("cannot register a hook on a tensor that doesn't require
  gradient"); no silent-wrong is possible (there is no graph at all, so the earlier "per-step retention
  overwrite" concern is moot). The isolated path fails at the same user line; the host signals the
  no-graph case (`handle_backward_event` returns a marker when nothing retained requires grad) so the
  worker's error names the real cause instead of "off the backward path".
- Efficiency deferred: the host computes + ships grads for ALL retained reads (`retain_graph` always
  on); a large read set can overflow the 64 MB arena.
- The `".backward("` source-substring detection can false-positive (e.g. the string in a comment),
  which only costs needless tagging — tightening is planned alongside the gate consolidation.
---

## 17. Triton-kernel models — the deployment motivation (and why this beats the in-process whitelist)

**Why this is urgent.** Frontier GPU model execution increasingly runs **Triton JIT kernels**, by three
independent routes: (1) architectures whose core op has no fast eager form ship Triton kernels — fused MoE
(vLLM `fused_moe`: Mixtral, DeepSeek-V2/V3/R1, Qwen-MoE, Llama-4, …) and SSM selective-scan (HF `mamba2`
imports `mamba_ssm.ops.triton.selective_state_update`; Jamba/Bamba/FalconMamba/Zamba); (2) HF's `kernels`
library + `kernelize(model, mode=inference)` pulls Triton kernels from the Hub as drop-in
norm/activation/attention replacements (Liger fuses RMSNorm/RoPE/SwiGLU/CE); (3) `torch.compile` →
TorchInductor, whose GPU codegen target *is* Triton. The triton-free path is essentially limited to a plain
dense transformer in eager mode with SDPA / precompiled-CUDA FlashAttention (which is an AOT library call,
not runtime codegen).

**Why the in-process whitelist can't serve them.** NDIF's sandbox
(`ndif:src/ndif/services/ray/nn/security/`) is a Python import whitelist that wraps the ENTIRE
`tracer.execute(model)` — the forward runs *inside* the Protector (unavoidable in-process: user
interventions interleave with the forward in one call stack). Triton's first-use compilation needs
`subprocess`+ptxas, `tempfile`, `os`, `importlib`, `open()` — none whitelisted, several hard-blocked by the
audit hook. So the model's own Triton kernels, firing during the sandboxed forward, are denied. A
pre-warmed kernel cache doesn't save it: per-request shapes recompile, and even a cache *hit* still needs
`os`/`open`/`importlib` to load the cached `.so`.

**Why this backend fixes it — the trust boundary aligns with the capability boundary.** The forward runs on
the TRUSTED host (unrestricted → Triton compiles normally); only UNTRUSTED user intervention code runs in
the worker. The split lands on the seam the whitelist couldn't find in-process: the thing needing dangerous
capability (runtime codegen) is the *model* (trusted); the thing we distrust is *user code*. The worker's
lockdown level is therefore **orthogonal** to whether Triton models run — worker security can be cranked to
the max without touching model Triton.

**Strictly better than upstream on the module-restriction axis.** Against the right baseline (upstream
*sandboxed* NDIF, not un-sandboxed nnsight):

| | Upstream whitelist | This backend (lockdown on) |
|---|---|---|
| user-code imports | restricted (allowlist) | restricted (seccomp import-freeze) — match-or-stricter |
| model serving stack | **also restricted** → Triton broken | **unrestricted** (host-side) → Triton works |
| user escape past Python layer | reaches host / tenants (whitelist README concedes C-level escapes; suite: 10/10 escapes succeed) | contained in a refless, killable process |

The seccomp **import-freeze** (under lockdown, only warm-time-loaded modules are importable, since
import == `open()`, which seccomp then blocks) is therefore *not a regression* — upstream restricts user
imports at least as hard. The one parity item is *which* modules a legit user may reach; close it with
`isolate_mediators(..., preimport=(...))` (and the matching `warm_worker_pool(..., preimport=(...))`), which
loads the deployment's allowed-module set at warm time, before lockdown. User-code Triton is **doubly
blocked** (import wall if not pre-warmed + compile wall: ptxas execve / cache open) — intended containment,
documented as unsupported.

**Verified properties (code, not just design):**
- *Timeout directionality is correct for slow host compiles.* The worker waits on the host with **no
  timeout** (`CudaIpcWorkerChannel.wait_response` — a blocking `recv`); the timeout lives only on the host
  (`CudaIpcHostChannel.wait_event`) and measures worker think-time. Host-side Triton compilation happens
  during the host's *forward execution* — never while the host is in `wait_event` — so a multi-second cold
  MoE autotune never false-trips the worker's hang-detector, in either direction.
- *Lockdown ordering / cold-vs-pool.* `lock_down()` runs once in `_run_one_job`, after the first job's
  payload is deserialized and before its user code runs — in the unified `_pool_worker_main` (the only
  worker entrypoint, cold via `poolable=False`, pooled via `poolable=True`). So **both** paths deserialize
  their first job before lockdown; the cold-vs-pool difference is recycle-vs-retire. The first `conn.recv`
  (which unpickles the job message) runs unlocked, so a fresh worker's first job needs no `preimport=`; a
  warm pool's *later* jobs run under the first job's lockdown (so a different model would need `preimport=`).

**Coverage:** `prototypes/mediator-sandbox/gpu_sandbox/test_isolated_triton_model.py` — `host_compiles`
(isolated + `lockdown=True` Triton-kernel model bit-identical to in-process, with a cold `TRITON_CACHE_DIR`
shown to populate during the isolated run) and `user_contained` (worker-side Triton blocked under
lockdown, exercising `preimport=`). Requires a GPU + a Triton install.
