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
- **What:** `ensure_provider(R)` — parse `R` (`"<path>.<output|input>.i<N>"`) → `iteration = N`,
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
- **Idempotency:** `ensure_provider` registers a given `R` once per mediator (tracked in a set on the
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

## 6. Feature coverage map (so nothing is silently dropped)

| Feature | Cross-process mechanism | Status |
|---|---|---|
| read / swap / skip / exception | events over channel + host-side hook registration | done |
| `.save()` (tensors) | worker→host saves transmission | done |
| multi-invoke + batch narrowing | per-worker host mediator; Batcher host-side | done |
| `iter`/`all`/`next` (multi-token) | iteration step stamped host-side; host iter-hooks bump the tracker; worker sets its step explicitly | done |
| `tracer.barrier()` | host-side participant counting + the existing `handle_barrier_event` coordination loop | done |
| `cross_invoker` variable sharing | host-mediated variable store (worker pushes data locals, pulls the merged store) | done |
| `with tensor.backward()` / `.grad` | needs host-side backward execution (the autograd graph is host-side) | planned |
| `tracer.cache()` | host-side cache-hook registration + post-forward injection of the populated CacheDict | planned |
| warm worker pool / MPS / `isolate_mediators()` polish | — | planned |

When isolation is on and a not-yet-supported feature is used, the trace **fails cleanly** — a
missed-provider error or the per-step timeout (the lifecycle is the safety net), not a silent deadlock or
silent-wrong result. (There is no automatic "route to in-process" fallback; features are added one at a
time. See the support matrix in §8 for what works today.)

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
3. **Host-side on-demand hook registration** (`interleaver.py`): `ensure_provider(R)` in `handle`.
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
  timeout); `isolation.py` (`isolate_mediators` + `spawn_isolated_worker` + `_worker_main` +
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

## 8. Support matrix (what works under `isolate_mediators()` today)

| Feature | Status |
|---|---|
| read / swap (`=`) / `.save()` (tensors) / skip / exception / multi-invoke | ✅ bit-identical |
| single-forward `generate(...)` (no iter) | ✅ verified |
| seccomp lockdown (fs/net/exec) | ✅ |
| `iter`/`all`/`next` (multi-token) | ✅ bit-identical (`iter[N]`, `iter[:]`, per-step swap) |
| `tracer.barrier()` | ✅ host-side participant counting |
| `cross_invoker` variable sharing | ✅ host variable store; transmittable data vars only — see §10 |
| `with tensor.backward()` / `.grad` | 🔜 hard: the autograd graph is host-side, the worker has detached clones; needs the backward pass to run host-side with path-based grad providers (a major build) |
| `tracer.cache()` | 🔜 tractable: returns an empty CacheDict today (hooks fire on dummy modules); needs host-side cache-hook registration + shipping the populated CacheDict back |
| `.source` operation-level access (`...attn.split_1.output`) | 🔜 not yet (op paths aren't in `model.modules()`) |
| in-place `[:]=` | ⛔ use explicit `=` (clone semantics, §4) |

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
- **Host iter-hooks:** `spawn_isolated_worker` calls `register_iter_hooks(host_mediator, real_model)` so
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
  channel). `Mediator._isolated_worker` (set in `_worker_main`) gates the worker-side behavior.
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
`_xinvoke_store` is per-interleaver (reset each trace) so no cross-trace leak today; a warm pool must
clear it + the dummy-module hooks + `Globals.saves` per trace. Known coverage gaps: no tests yet for
multi-barrier-in-one-trace, 3+ participants, multi-token + barrier, or variable sharing without a barrier;
the store grows monotonically per trace and ships all shared tensors CPU-serialized on every response (a
perf cliff for large cross-invoke tensors).

---

## 11. Backward + caching — CHARACTERIZED (not yet built)

`test_isolated_backward_cache_gaps.py` confirms the two gaps and their difficulty:

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
