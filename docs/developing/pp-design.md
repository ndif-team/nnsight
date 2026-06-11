---
title: Pipeline Parallelism — Design Spec (as built)
one_liner: The detailed engineer-facing design of nnsight's vLLM pipeline-parallel path — how a single-GPU-style trace runs transparently across PP stages via Envoy short-circuit, LazyRemoteTensor, a run-ahead worker, a gloo pull listener, and a finalize drain barrier.
tags: [internals, dev, vllm, pp, design]
related: [docs/developing/pp-pipeline-parallelism.md, docs/developing/vllm-integration.md, docs/developing/pp-stress-findings.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/modeling/vllm/pp.py, src/nnsight/modeling/vllm/pp_envoy.py, src/nnsight/modeling/vllm/lazy_remote_tensor.py, src/nnsight/modeling/vllm/pp_listener.py, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py, src/nnsight/intervention/interleaver.py]
---

# Pipeline Parallelism — Design Spec (as built)

> **What this is.** The detailed, engineer-facing design of the nnsight vLLM
> pipeline-parallel (PP) path, kept current with the code on `pp-on-dev`. For the
> gentle, picture-first introduction (why PP breaks the normal intervention model and
> the shape of the fix), read the walkthrough first:
> [pp-pipeline-parallelism.md](pp-pipeline-parallelism.md). This page assumes that
> framing and goes component-by-component, as built, with file references.
>
> Scope: **PP ≥ 2** on the vLLM V1 path (sync `LLM` and async `AsyncLLM`), single-node
> multiproc and Ray (incl. 2-node containers). Composes with TP. The whole surface lives
> in `src/nnsight/modeling/vllm/`; `nnsight.intervention` stays PP-unaware except for the
> one buffer-clone sidecar in `interleaver.py`.

## 1. Goal & the transparency contract

The user writes single-GPU-style intervention code; the system runs it across stages with
no `if pp_rank == X` guards and no awareness of stage boundaries:

```python
model = VLLM("meta-llama/Llama-3.1-70B", tensor_parallel_size=4, pipeline_parallel_size=2)

with model.trace("Hello") as tracer:
    h5 = model.model.layers[5].output[0]            # produced on stage 0
    model.model.layers[60].mlp.output = h5 * 2      # consumed+written on stage 1 (cross-stage!)
    logits = model.logits.save()                    # produced on stage 1 (last stage)

with model.trace("Hello", max_tokens=8) as tracer:  # multi-token, same contract
    outs = list().save()
    for _ in tracer.iter[:8]:
        outs.append(model.logits)
```

The middle line of the first trace expresses a **cross-stage data dependency** in one
line: read a hidden state owned by stage 0, scale it, write it into a module owned by
stage 1. Making that work transparently — including across many decode tokens — is the
whole problem.

> **Write idiom.** Cross-stage (and same-stage) writes use **replacement**
> (`module.output = value`). In-place mutation of a vLLM inference tensor
> (`module.output[0][:] = …`) is intentionally unsupported (it raises
> `RuntimeError: Inplace update to inference tensor …`); see
> [pp-stress-findings.md](pp-stress-findings.md) §N1.

## 2. How vLLM PP works (the substrate)

**Model structure.** With PP=2 on an N-layer model, vLLM's `make_layers()` builds **all N
layer slots on every rank**, but only `start_layer:end_layer` are real `nn.Module`s; the
rest are `PPMissingLayer` (a subclass of `nn.Identity`) that the forward never calls.
Non-layer modules split too: `embed_tokens` is real only on the first stage; `norm` /
`lm_head` / logits / sampling only on the last.

```
Stage 0 (PP rank 0)                       Stage 1 (PP rank 1)
model.model.layers = [                    model.model.layers = [
  [0..k):  Real                             [0..k):  PPMissingLayer (stub)
  [k..N):  PPMissingLayer (stub)            [k..N):  Real
]                                         ]
embed_tokens = Real, norm = Missing       embed_tokens = Missing, norm/lm_head = Real
```

**Execution.** All ranks enter `execute_model()` together via vLLM's `collective_rpc`.
Within a forward, stages run sequentially: stage 0 computes its layers and sends an
`IntermediateTensors` blob to stage 1, which computes the rest and the head.
`execute_model()` runs **once per decode token**; multi-token generation = many calls.

**nnsight's contract on top.** The user's intervention body is compiled and run on a
**worker thread** (the *mediator*; see
[threading-and-mediators](../concepts/threading-and-mediators.md)). The **same** mediator
is deserialized and run on **every rank**. Accessing a module normally **posts a request
and blocks** until the forward fires that module's hook. The *interleaver* opens/closes
once per `execute_model()` and gates hook dispatch.

**Why this collides under PP.** On stage 1, the very first line `model.model.layers[5]`
touches a `PPMissingLayer` whose hook never fires → the mediator blocks on line 1 forever.
The naive "push all intervention state to the owning rank" fix is impossible: the mediator
is a **live, stateful thread** (frame locals, an accumulating saved list, iter counters)
that can't be serialized mid-flight, each stage needs its *own* past values across tokens,
and most interventions have no cross-stage dependency at all.

## 3. The design in one paragraph

Detect "this module lives on another stage" **at the access site, before blocking**, and
hand back a `LazyRemoteTensor` placeholder instantly. Writes/saves to it are no-ops (the
real write lands on the owning rank, which runs the same line during its own forward);
only a genuine **consume** (`lazy * 2`, indexing-then-reduce, iteration) **materializes**
it — a pull over a dedicated gloo group from the owning rank's buffer. Because remote
accesses never block, each rank's worker **runs ahead** of its forward, skating past every
remote module and parking exactly at the first module it owns. A small **readiness gate**
synchronizes the worker with the forward; a **drain barrier** at request finalize keeps the
serving buffer alive until every rank's worker has finished pulling.

## 4. Components, as built

### 4.1 `PPModuleMap` — owning-rank resolution (`pp.py`)

Ownership is **derived, not guessed from naming conventions**. The load-time meta
exchange (§4.10's sibling, `_exchange_pp_module_meta`) allgathers each rank's real
(non-`PPMissingLayer`) `named_modules()` names; a module reported by exactly one stage is
owned by it (a module reported by several — containers, build-on-every-rank modules — is
ambiguous and dropped). The runner installs the result via `set_derived_owners`, first
injecting the only non-derivable facts: the build-on-every-rank, fire-on-last modules
(vLLM's `logits_processor` and nnsight's own `logits`/`samples` root wrappers) are
claimed for the **last** stage, since sampling runs there. All ownership knowledge lives
at that one install site; non-standard names (Falcon `word_embeddings`, OPT
`final_layer_norm`, GPT-NeoX `embed_out`) resolve with no name tables.

`get_owning_rank(module_path)` strips a trailing eproperty key, canonicalizes against
the known envoy root (every envoy path is the root component — `nnsight_model.path` —
plus the raw `named_modules()` name), and walks up to the nearest owned ancestor
(`model.model.layers.5.mlp` → `model.layers.5`). Results are memoized per path
(ownership is constant for the model's life). Unknown → `None` (treated as local; a
genuine cross-stage *consume* of an unresolvable path raises a descriptive error at pull
time). Before `set_derived_owners` runs, every path resolves to `None` — nothing traces
that early.

`is_pp_missing(module)` detects the stub by class name (`type(m).__name__ ==
"PPMissingLayer"`) so no hard import of a vLLM-internal class. `resolve_meta(meta_map,
path, root)` looks up the per-module dtype hint: exact match, else strip the single
known root component — never "strip until something matches", which could silently hit
a wrong entry.

### 4.2 `PPEnvoy` / `pp_eproperty` — the short-circuit (`pp_envoy.py`)

`VLLM` is wired with `envoys=PPEnvoy`, so every module's `.output`/`.input`/`.inputs` is a
`pp_eproperty`. At the top of `__get__`/`__set__` it calls `_is_pp_missing(obj, key)`:

- True if the underlying module is a `PPMissingLayer`, **or** `pp_module_map.get_owning_rank(f"{obj.path}.{key}")` ≠ this rank (handles the always-present-on-every-rank `logits`/`samples`/`logits_processor` case).

On a hit:
- **`__get__`** returns `_pp_lazy_access(obj, key)` — a `LazyRemoteTensor`. It does **not**
  call `_hook(obj)` (no forward hook on a module that won't run) or
  `interleaver.current.request(...)` (no blocking wait).
- **`__set__`** is a **no-op** (the real swap happens on the owning rank running the same
  line during its forward).

Both paths run **regardless of `interleaving`** (not just while a forward is live). This is
load-bearing: the worker runs ahead and a downstream access (e.g. `model.logits` after a
cross-stage write) can execute on the mediator thread *after* this rank's forward released
and the interleaver tore down. Gating the short-circuit on `interleaving` (the old
behavior) let that case fall through to `super().__get__`, which raises "Cannot access …
outside of interleaving."

`_pp_lazy_access` (the consumer side of a pull):
1. Bumps the per-`(path.key)` iteration tracker and forms the **provider string**
   `f"{module_key}.i{iteration}"` so repeated accesses across tokens get distinct keys.
2. Resolves the owner from the **full `module_key = f"{path}.{key}"`** — *not* `path` alone.
   Root-level epropertys (`logits`/`samples`) have `obj.path == "model"` (the name lives in
   `key`); `get_owning_rank("model")` is `None` → a pull with `source_rank=None` that blocks
   forever. Resolving from `module_key` matches `_is_pp_missing`'s own lookup. (This was the
   "logits-consume-in-iter hang"; see [pp-stress-findings.md](pp-stress-findings.md) §P1.)
3. Builds the `LazyRemoteTensor(source_rank, provider_string, dtype)` and wires its
   `_pull_fn` to `listener.pull_from_remote(src, prov, req_id=mediator.pp_req_id)`. The
   pull is sized entirely from the producer's reply (shape and true dtype ride the wire):
   the run-ahead worker builds the lazy before the matching forward is scheduled, so no
   consumer-side token-count capture can reliably match the produced value's leading dim
   (a wrong size either under-allocates the recv buffer — gloo abort — or over-allocates
   it — a silently wrong-shaped tensor).

`_pp_signal_remote(obj, key)` runs on every cross-stage access and classifies it against the
worker's per-step `(leading-remote)(local)(trailing-remote)` lifecycle, recorded on the
mediator's `pp_progress` (`PPWorkerProgress`, §4.6):
- **Downstream** owner (later stage) → *trailing remote*: no more local hooks this step, so
  mark `pp_progress.past_local = True` (the readiness gate stops waiting for it) and, once
  (`pp_progress.gone_remote` guard), `go_remote()` to release this rank's forward so it can
  complete and perform the inter-stage send the downstream value depends on.
- **Upstream** owner (earlier stage) → *leading remote*: the pull resolves on the producing
  rank, so do **not** mark past-local (a local access may still follow); only release a
  blocked value-injection `respond` if one is pending.
`past_local` is marked **regardless of `interleaving`** (the worker can reach the access
in the gap between forwards); only `go_remote` (which posts into the live event protocol) is
gated on a live forward.

### 4.3 `LazyRemoteTensor` (`lazy_remote_tensor.py`)

A metadata-only proxy (`source_rank`, `provider_string`, `dtype`) with `_real=None` and a
`_pull_fn`. Materializes once, lazily, caching `_real`.

| Operation | Behavior | Crosses the wire? |
|---|---|---|
| `lazy[:] = X`, `lazy[0][:] = X` | `__setitem__` → no-op | No |
| `lazy.save()` | returns self; merged away from the owning rank's real save | No |
| `lazy.shape` / `.dtype` / `.device` | metadata (`.shape`/`.device` do materialize) | mostly No |
| `lazy[i]` | **child** lazy whose deferred pull does `parent._materialize()[i]` | No (until consumed) |
| `lazy * 2`, `torch.cat([lazy, x])`, `lazy.float()`, dunder ops | `__torch_function__`/dunder → materialize → pull | **Yes** |
| `for row in lazy`, `tuple(lazy)`, `len(lazy)` | `__iter__`/`__len__` → materialize | **Yes** |

Two subtleties:
- **`__getitem__` returns a child lazy**, not `self`. `layers[i].output[0]` is `lazy[0]`,
  a child tracking element 0; consuming it pulls the parent **once** (cached) and indexes.
  This is why a decoder layer's `(hidden, residual)` tuple element reads correctly across
  stages (see §6 and [pp-stress-findings.md](pp-stress-findings.md) §P2). Earlier, when
  `__getitem__`/`__iter__` returned `self`, `tuple(lazy)` spun forever on the non-owning
  rank — the "cross-stage write hang."
- **`__getstate__` nulls `_pull_fn`** so a lazy is picklable. A raw-saved, never-consumed
  lazy ships back to the driver as a proxy with `_real=None` (no pull); only a real consume
  on the worker pulls. `strip_lazy` replaces any still-unmaterialized lazy in the saves with
  the `NOT_ON_THIS_RANK` sentinel before the merge (§4.8).

### 4.4 `pp_hook_buffer` + the clone (`interleaver.py`)

When a local hook produces a value the mediator requested, `Mediator.handle_value_event`
(the `respond` path) stores a **clone** into the rank's `pp_hook_buffer`, keyed by the
composite `(provider_string, req_id)`, then hands it to
`listener.dispatch_parked(key, value)` so any parked cross-rank pull unblocks:

```python
# interleaver.py (the PP sidecar, under the buffer condition + torch.inference_mode())
stored = _deep_clone(value)                  # see below
self.interleaver.pp_hook_buffer[key] = stored
listener.dispatch_parked(key, stored)
```

- **Why clone:** the raw hook tensor is part of the live forward graph; vLLM (eager,
  aggressive buffer reuse in fused add-RMSNorm) overwrites it in later layers. The buffer
  needs a value that survives independently.
- **Why `_deep_clone`, not `value.clone()`:** a decoder layer's output is a `(hidden,
  residual)` **tuple**; cloning only a bare tensor left the tuple's tensors aliased to live
  storage that later layers mutated → stale cross-stage reads of internal (non-boundary)
  layers, with correct boundary/MLP reads (which happen to clone). `_deep_clone` recurses
  into tuples/lists/dicts. (See [pp-stress-findings.md](pp-stress-findings.md) §P2.)
- **Speculative & local:** the clone fires on **every** hooked access on the owning rank
  whether or not anyone pulls; it never leaves the rank. The listener serves from it on
  demand. Measured cost ≈ 2–3 ms/layer, overlapping compute — not a bottleneck.
- **Composite key** `(provider, req_id)` keeps concurrent requests in the same forward from
  delivering each other's slices.

### 4.5 `PPListener` — the pull protocol (`pp_listener.py`)

One persistent daemon thread per rank, on a **dedicated gloo group** (`pp_pull_group`,
separate from vLLM's NCCL PP group so the listener's `recv` never collides with inter-stage
transfer). gloo routes strictly by `(peer, tag)`:

- `TAG_REQUEST = 0` — the single, fixed-size, self-identifying pull request.
- `TAG_RESPONSE_BASE + (n % 2²⁰)` — a **distinct response tag per in-flight pull**, so
  concurrent consumers never receive each other's reply.
- `TAG_DRAIN` — one above the whole response range; used only by the finalize barrier
  (§4.9), invisible to serving.

**Serve loop** (`_listen_loop`) — never blocks on producing a value:
```
recv one request on TAG_REQUEST  (group_src wildcard for PP>2; the request carries the requester + response tag)
decode -> lookup_key = (provider, req_id)
under the buffer condition:
    if key in buffer:  reply_pool.submit(_serve_reply, req, value)     # serve now
    else:              park the request under key                       # served later by dispatch_parked
```
Parking (rather than blocking the recv) is essential: a blocking serve here would stop the
loop posting the next `recv`, freezing every other rank's request `send` at the rendezvous —
the multinode head-of-line deadlock. `dispatch_parked(key, value)` (called by the producer
right after it writes the buffer) hands any waiters to the reply pool.

**Reply** (`_serve_reply`, on a thread pool): one **self-describing** format — a fixed
shape header (tensor count, the value's *true* dtype as a wire-codec code, per-tensor
shapes) then the flat data, both on the per-pull tag. The producer is the only side that
knows the value's shape and dtype under run-ahead; stamping the true dtype is what keeps
integer-valued outputs correct (sampled token ids are int32, not the model's bf16 compute
dtype — a weight-derived guess under-sizes the recv buffer and gloo aborts). The whole
reply is **prepared before any send**: a serialization failure (non-tensor value,
mixed-dtype tuple, header overflow, un-encodable dtype) becomes an **error reply**
(sentinel header + message) that makes the blocked consumer raise descriptively — never a
partial send that desyncs its recv. A per-op gloo recv timeout cannot be the backstop:
probed, it closes the whole peer pair on expiry. `clear_buffer` error-replies any pull
still parked for a finalized request (its value will never be produced).

**Consumer** (`pull_from_remote`): allocate a private response tag, `send` one request on
`TAG_REQUEST`, then `recv` the header and data on that tag. No lock — each mediator thread
runs its own pull concurrently; the per-pull tag keeps replies from colliding.

Measured pull cost ≈ 3.4 ms (local/SHM) / ~7 ms (cross-container TCP) per pull — cheap.

### 4.6 The run-ahead worker & the readiness gate (`GPUModelRunner.py`)

The mediator worker **runs ahead of the forward** — the forward waits for the worker, never
the reverse. So a one-shot hook is always registered before the forward reaches its module
(otherwise the monotonic iteration tracker advances past it and the hook never fires — a
hang). `_update_states` schedules this step's mediators (`process_batch_groups`) and, with
PP enabled, calls the **one PP sync point**, `_pp_wait_for_mediators`, before hooks fire.

The worker-progress state the gate reads lives in one object,
`mediator.pp_progress` (`PPWorkerProgress`, defined in `intervention/interleaver.py`):
the per-step latches (`past_local`, `gone_remote`), the gate-only counters
(`scheduled_count`, `max_iteration`), and `worker_iteration`. `reset_iteration()`
encapsulates the publish-order invariant — clear the latches *before* publishing the new
iteration number, so a gate thread that sees `worker_iteration == k` can never read a
stale `past_local` from the previous step.

For the forward of iteration `k = pp_progress.scheduled_count - 1`, a mediator is "ahead"
when (`pp_progress.is_ahead_of(k, alive=…, parked=event_queue.has_value)`):
- it's no longer `alive`, or
- `k > max_iteration` (a single-shot `model.trace` that never reaches `k`), or
- `worker_iteration > k` (the worker already ran past `k`, so iteration-`k` hooks were
  registered), or
- `worker_iteration == k` **and** (parked at a local module — **or** `past_local` — it
  determined it has no local part this step).

The gate spins (`time.sleep(1e-4)`) until every mediator is ahead, bounded by a 30 s
deadline that raises loudly instead of hanging (env-overridable via
`NNSIGHT_PP_GATE_TIMEOUT` for genuinely slow links).

### 4.7 Save collection & merge (`collect_nnsight`, `merge_saved`)

`collect_nnsight(req_ids, finished_req_ids)` is invoked on **every** rank via
`collective_rpc` (sync engine `step()` and async `_stream()`, both gated on
`output.finished`), so all ranks see identical `finished_req_ids`. **Every** rank —
TP siblings included — runs the finalize teardown (its mediators, hooks, worker threads,
and buffer entries are real and would otherwise leak); only **TP-rank-0 of each PP
stage** *ships* saves (TP siblings carry replicated mediator state and would duplicate).
A pure streaming collect with nothing to finalize still short-circuits on TP siblings.

Each rank produces a partial save tree; non-owning slots are the `NOT_ON_THIS_RANK`
sentinel (`strip_lazy` converts unmaterialized lazies to it). The engine merges position-
wise (`collect.merge_collected_saves`, the single implementation shared by the sync
engine, async backend, and serve handler) with `merge_saved(a, b)`: prefer the
non-`NOT_ON_THIS_RANK` leaf at each slot; recurse through equal-length lists/tuples and
same-key dicts; on a real/real clash or structural mismatch, **`b` wins** (preserving
"later-rank-wins" scalar semantics, degrading safely).

This is **why cross-stage reads stay correct even if a consumer's pull is abandoned**: both
ranks save all values, the producer rank computes the remote layers locally and correctly,
and the merge takes the producer's real leaf over the consumer's sentinel.

### 4.8 Request finalize & the drain barrier (`collect_nnsight`, `PPListener.drain_barrier`)

When a request finishes, the finalize block runs (all ranks, identical `finished_req_ids`):

```python
interleaver.stop_iteration()                 # wake iter-gated workers, set _generation_done
for mediator in matched: mediator.worker.join(timeout=5.0)   # drain this rank's worker
if self.pp_listener is not None:
    self.pp_listener.drain_barrier()          # <-- hold until ALL ranks drained
... collect_saves ...
self.pp_listener.clear_buffer(finished_keys)  # scoped clear of THIS request's keys only
```

`drain_barrier()` is an all-PP-ranks barrier on `TAG_DRAIN` (rank-ordered pairwise p2p:
lower rank sends first — deadlock-free; no-op without a peer). It sits **after the worker
join and before the buffer clear**. Each rank reaches it only after joining its own
(drained) worker, and **keeps serving peers on its listener thread while it waits** (the
buffer is still alive — clear is after the barrier), so a peer's still-in-flight pulls are
satisfied from the live buffer instead of blocking on a torn-down peer.

**Why it's needed.** Without it, under vLLM async scheduling the producer rank — whose own
worker only read local layers and so joined instantly — cleared its buffer before the
consumer rank finished pulling; the consumer's late pulls then blocked in `dist.recv` until
the 5 s join timeout, a fixed ~5 s stall on reading many cross-stage layers in one forward
(logit-lens / save-every-layer). Correctness was unaffected (the merge backfills), so it was
a pure latency stall. The barrier closes the race; the `join` now completes promptly and
remains only as a safety net. **Deadlock-free** because `collect_nnsight` runs via
`collective_rpc` with identical `finished_req_ids` and only TP-rank-0-of-each-stage runs the
block — exactly the pull group's members — and serving lives on a separate thread, so
bidirectional pulls complete even with both ranks in the barrier. Worst case degrades to the
prior 5 s join, never worse.

`clear_buffer` is **scoped** to the finished request's composite keys — a blanket clear
would wipe concurrent in-flight requests' slices.

### 4.9 PP + TP

The pull group is built **per TP slice**: for `tp_offset in range(tp_size)`,
`pp_ranks_for_tp = [pp_rank*tp_size + tp_offset for pp_rank in range(pp_world_size)]`. So PP
rank `r`'s pull group is the column of constant TP offset (one rank per stage). The
listener's `local_rank` is `get_pp_group().rank_in_group` (the stage index). Because
`collect_nnsight` runs the finalize block only on TP-rank-0, the drain barrier's participants
are exactly that group's members. Validated at PP=2×TP=2 on Docker Ray.

### 4.10 PP-aware deserialization (`_pp_aware_load`)

Mediators serialized on the client reference the full meta model. On a PP worker, modules on
other stages are `PPMissingLayer` stubs without children, so `_pp_aware_load` falls back to
the nearest `PPMissingLayer` ancestor for missing paths (e.g. `model.h.6.ln_1` →
`model.h.6` stub). The grafting of stub children onto the worker Envoy tree at load
(`_graft_pp_missing_envoys`) is what lets `model.model.layers[5].attn.output` resolve on a
stage that doesn't own layer 5.

## 5. End-to-end timelines

**Single forward** (`layers[60].mlp.output = layers[5].output[0] * 2; model.logits.save()`):

| Step | Stage 0 (owns L5) | Stage 1 (owns L60, logits) |
|---|---|---|
| `layers[5].output[0]` | real → block → hook fires → tensor; **clone into buffer** | PPMissing → child `LazyRemoteTensor` (instant) |
| `… * 2` | local tensor arithmetic | materialize → **pull L5 from stage 0's buffer**, then `* 2` |
| write `layers[60].mlp.output` | PPMissing → no-op | real → replacement takes effect in the forward |
| `model.logits.save()` | PPMissing → lazy → save no-op (→ `NOT_ON_THIS_RANK`) | real → hook fires → real save |
| finalize | drain barrier · contributes sentinels | drain barrier · contributes real `logits` |
| merge | `merge_saved` keeps stage-1's real `logits` over stage-0's sentinel | |

**Multi-token** (`for _ in tracer.iter[:8]: outs.append(model.logits)`): both workers are
long-lived threads; their frame locals (`outs`, loop counters) persist per rank across
tokens. Each step the readiness gate releases the forward once the worker is ahead; stage 1
pulls a fresh `model.logits`/upstream value lazily per token. Nothing is serialized or
shipped between ranks except the values actually consumed.

**Bidirectional** (read *every* layer, e.g. logit-lens): each rank both pulls the other
stage's layers and serves its own. The finalize drain barrier keeps both buffers alive until
both workers finish pulling.

## 6. Correctness properties

- **Non-determinism is safe.** Both ranks run the same body independently; `torch.randn` etc.
  differ per rank. Writes only take effect on the owning rank (lazy writes are no-ops);
  saves come from the owning rank (lazy saves merge away); reads materialize from the owning
  rank's buffer. So the surfaced result is deterministic per owner.
- **Tuple-element reads** are correct because `__getitem__` returns a child lazy that pulls
  the (deep-cloned) parent once and indexes it — not the residual stream.
- **Abandoned pulls don't corrupt results** — the cross-rank `merge_saved` fills each slot
  from the producer's real local save over any consumer sentinel.

## 7. Performance characteristics

Measured on Qwen2.5-7B, PP=2 (full numbers: the fix commit and
[pp-stress-findings.md](pp-stress-findings.md)):
- **Plain generation is a PP *win*** under `enforce_eager` (single-stream decode is
  launch/schedule-bound): PP=2 ~25–33 % faster than PP=1; nnsight tracks vanilla vLLM.
- **Neither clone nor pull is a fundamental bottleneck** — clone ≈ 2–3 ms/layer (overlaps
  compute), pull ≈ 3.4 ms local / ~7 ms TCP. Reading+consuming all layers ≈ 1.7× PP=1.
- The historical ~5 s save-every-layer stall (multiproc/async scheduling) is fixed by the
  finalize drain barrier (§4.8).
- 2-node (Ray, cross-stage over TCP) plain generation is ~2× slower than single-node
  multiproc (per-token inter-stage transfer + async scheduling off there), but exhibits no
  cross-stage-read stall.

## 8. Limitations & known issues

- **In-place writes** on vLLM inference tensors are unsupported by design — use replacement
  (§1, [pp-stress-findings.md](pp-stress-findings.md) §N1).
- **Buffer growth.** `pp_hook_buffer` accumulates `(provider, req_id, iN)` clones across a
  request's tokens; cleared (scoped) at finish. Fine for decode (small tensors); very long
  generations with many hooked modules may eventually want eviction.
- **≥3-node Ray** PP fails at kv-cache init — a **stock vLLM** bug (`global_rank` not synced
  in `adjust_rank`; upstream #41287 / PR #41298), not nnsight; ≤2-node and single-node
  multiproc are unaffected. See [pp-multinode-ray-init-bug.md](pp-multinode-ray-init-bug.md).
- **`tracer.barrier()` / cross-invoke** is an advanced primitive not specific to PP; its
  vLLM limitation is tracked separately.

## 9. Map of the code

| Concern | File |
|---|---|
| Owning-rank map, stub detection, meta lookup | `pp.py` |
| Envoy short-circuit, lazy build, remote signal, iteration tracking | `pp_envoy.py` |
| The lazy proxy (materialize, child-lazy, iter, pickling) | `lazy_remote_tensor.py`; merge in same file |
| Buffer clone sidecar (`_deep_clone`), `dispatch_parked` wiring | `intervention/interleaver.py` |
| Listener thread, pull protocol, `drain_barrier` | `pp_listener.py` |
| PP setup, pull-group construction, readiness gate, `collect_nnsight` finalize/clear | `model_runners/GPUModelRunner.py` |

## See also

- [pp-pipeline-parallelism.md](pp-pipeline-parallelism.md) — the illustrated walkthrough (why PP breaks the model; the same design with figures).
- [vLLM Integration Internals](vllm-integration.md) — the broader vLLM↔nnsight machinery.
- [pp-stress-findings.md](pp-stress-findings.md) — intervention-pattern correctness/perf findings (P1/P2/N1) and the cross-stage-read stall fix.
- [Threading and Mediators](../concepts/threading-and-mediators.md) · [Interleaver and Hooks](../concepts/interleaver-and-hooks.md) — the single-GPU model PP extends.
