# The in-process fast lane — design

**Status:** Implemented (first slice) · **Date:** 2026-06-13 · **Branch:** `worktree-mediator-sandbox`
**Builds on:** [mediator-gpu-trace-integration.md](mediator-gpu-trace-integration.md) (the GPU-worker isolation harness this adds a tier to).

## 1. Why — isolation can't run the interp majority

The GPU-worker isolation contains footguns by running each user intervention in a spawned process
that holds a **path-only, weightless mirror** of the model: `isolation._WorkerPersistent` synthesizes a
bare `nn.Module()` (only a `__path__`, no parameters, no `forward`) for every `Module:<path>`. That is
fine for reading/swapping/saving *delivered activations*, but the actual interpretability workloads —
checked against the `interp-serve-bench` taxonomy — do more than that **inside the trace body**:

```python
normed = model.transformer.ln_f(hidden)          # call the host's real final-norm module
logits = F.linear(normed, model.lm_head.weight)  # read the host's real unembed WEIGHTS
```

Every logit-lens, steering, ablation, activation-patching, and attribution cell reads host weights
and/or calls host modules. In the worker, `head.weight` → `AttributeError` and `head(x)` →
`NotImplementedError` on the weightless dummy. So **as it stood, isolation could not run any of the
weight-reading interp majority** — the workflow's compatibility pass found this the single largest gap
(9 of the cataloged workloads, all blocked on the same weight/host-module surface).

The fix is not to ship weights into the worker (a per-trace cost, and a partial answer). It is to run
the confirmed-safe interventions **in-process**, where the real model and its weights already live, and
isolate only the code that genuinely needs containing. That is the fast lane.

## 2. The three tiers

`isolate_mediators()` now classifies each mediator once, at fork in `Mediator.start`:

| tier | when | how it runs |
|---|---|---|
| **FAST** | the effective code is all whitelisted ops / host-model access / nnsight primitives | in-process daemon thread (`_iso=None`) — full model + weights, no worker, no per-hook channel; a watchdog bounds runaway loops |
| **ISOLATE** | anything unconfirmable (unresolved global call, import, `while`, unrecoverable closure, unknown node) | the existing GPU worker |
| **REJECT** | an introspection escape (`__globals__` / `getattr` / `eval` / …) | raise `FastLaneRejected` |

The conservative default is **ISOLATE**: absence of proof is not proof of safety; only explicitly
whitelisted code reaches FAST. Default behavior is preserved — isolation **off** never consults the
gate; isolation **on** only ever moves a mediator *isolate → in-process*.

## 3. Detect & confirm — the classifier (`fastlane.py`)

A **fail-closed, default-deny** static walk over the **effective code**: the trace body PLUS every user
closure it calls, resolved through the capturing frame, the function's `__globals__`, and its closure
cells. This closure resolution is load-bearing — the harness wraps real compute in `build()` /
`capture()` / `patch()` closures, so a walk of the `with` block alone sees only `build().save()`, an
opaque call. Confirming on the with-block while the footgun lives in an unresolved closure would be the
exact false-safe the gate must never produce; an unrecoverable closure fails closed to ISOLATE.

The walk (the rules, in `_Walker`):

- **Node allowlist.** Permit assignment / expr / return / `if` / `for` / `with` / `raise` / `pass` and
  pure expression nodes. `while` → ISOLATE (cannot be statically bounded). `import` → ISOLATE (ambient
  authority). `try` → ISOLATE. `global`/`nonlocal` → REJECT. Unknown node type → ISOLATE.
- **Call targets.** Each call must resolve to: a pure-compute op (`torch` / `torch.nn.functional` /
  `math` / `operator` / `numpy`), an nnsight primitive, a host `Envoy`/`nn.Module`/`Tensor` (calling it
  runs the real model — the fast lane's whole point), a recursively confirmed user function, a safe
  builtin, or a local/parameter name (a host object bound at a walked call site). An unresolved global
  call → ISOLATE (unknown authority); `torch.load`/`save`/`hub`/`jit` → ISOLATE (fs/net/JIT).
- **Introspection ban (REJECT).** `getattr`/`setattr`/`eval`/`exec`/`compile`/`__import__`/`globals`/…,
  any dunder attribute (`x.__class__`), and a dunder/private subscript key (`d["__builtins__"]`).
- **Host-state writes (ISOLATE).** An attribute store other than the nnsight boundary writes
  (`.output`/`.input`/`.grad`) mutates state visible to sibling mediators. In-place
  (`hidden[:] = …`) and replacement boundary writes are *allowed* (and in-place is **safer** here than
  under isolation, where clone-on-receive silently no-ops it).
- **Backward detection** is closure-aware (`with x.backward():` found through closures), replacing the
  old `.backward(` source substring that was blind to a backward hidden in a closure — this flag now
  feeds the *isolated* job's gradient-retention too.

Verdicts are cached by intervention code-object identity, so re-running the same trace pays the walk
once. The classifier is GPU-free and unit-tested (`test_fastlane_classifier.py`) on the real workload
shapes and on **renamed** module structures (`decoder_blocks`/`final_norm`/`output_projection`) so
nothing is keyed to GPT-2 naming.

### Threat model (the contract)

This is **not** an adversarial sandbox — a determined author can defeat any in-process restriction
(the pysandbox negative result). Under the harness's relaxed "contain footguns, not adversaries" model
the gate confirms the effective code: introduces no ambient authority (no import, no introspection, no
unresolved global call), has no unbounded loop, writes no host state, and is composed only of
whitelisted ops / host access / confirmed user code. It is cordoned to `trust="local"` provenance and a
`CONFIG.APP.FAST_LANE` flag; anything deserialized/remote, or with the flag off, isolates wholesale.
OOM and device-side asserts in pure tensor math are knowingly traded to the in-process tier (the same
risk a user running without isolation already accepts); a deployment that cannot tolerate them sets
`fast_lane=False`.

## 4. The watchdog

The static gate bans `while` and confirms loops are over bounded iterables, but a *huge* bounded
`range(10**12)` passes statically yet would hang the host — the loop-containment that turning isolation
on implies. A best-effort wall-clock `Watchdog` injects a `FastLaneTimeout` into the fast-lane thread at
its next bytecode if it overruns `fast_lane_timeout`. Because the intervention body is already wrapped
in `try/except → mediator.exception` (invoker.compile), the injected exception routes through the normal
path and the host re-raises it cleanly — no channel hang. It cannot preempt a wedged native/CUDA call
(only the worker-process kill can); the bounded-loop static rule is the primary defense, this is the
backstop. Armed after thread start, disarmed in the thread's `finally` and again at `cancel`.

## 5. Prior art it is built on

| system | tier | borrowed idea |
|---|---|---|
| Cloudflare Workers (V8 isolates) | fast-in-process | many tenants in one process; safety from a capability-restricted runtime, not a process boundary — the fast lane's model |
| RestrictedPython (Zope) | static-confirm | a `RestrictingNodeTransformer`-style default-deny AST pass as the gate |
| SES / Hardened JS (Endo) | fast-in-process | confirmed code runs against an explicitly granted namespace; no ambient authority |
| torch.fx / JAX tracing | static-confirm | a leaf/atomic-op allowlist; "prove the body is a pure function over provided tensors" — fail, don't guess |
| gVisor | hybrid | cost is the boundary *crossing*, not the work — minimize crossings (the fast lane removes them entirely) |
| AWS Lambda + Firecracker / SnapStart | heavy-isolated | what the existing GPU worker *is* — the slow lane you keep for unconfirmable code; warm-restore to amortize spawn (the warm pool) |
| pysandbox (negative result) | — | the contract: in-process restriction is a footgun selector, never an adversarial boundary → the `trust="local"` cordon |

The unifying pattern across all of them: **confirm once up front, then run free** behind a small,
explicit, enumerable fallback to the heavy tier.

## 6. Part 2 — declarative primitives

The fast lane already runs the weight-reading workloads in-process, so they need no new APIs *for the
fast lane*. The declarative primitives' value is (a) making the workloads runnable on the **isolated**
tier too, and (b) collapsing common raw-compute patterns into named calls the gate whitelists trivially:

| primitive | signature | taxonomy primitive | role | status |
|---|---|---|---|---|
| `tracer.unembed` | `(residual, norm, head, formulation="weight") → logits` | host-weight read + module call | the projection every logit-lens / steering-direction / attribution metric does | **built** |
| `tracer.steer` | `(envoy, direction, alpha=1.0)` | boundary write (injection) | always a replacement swap — fixes in-place's silent no-op under isolation | **built** |
| `tracer.patch` | `(envoy, value)` | boundary write (transplant) | whole-tuple replacement | **built** |
| `tracer.ablate` | `(envoy, mode="zero")` | boundary write (injection) | zero/mean knockout | **built** |
| `tracer.capture` | `(value) → handle` | read + run↔run transfer | cross-trace handoff; non-transmittable → clean fail, not silent drop | designed |

Most mirror the existing `tracer.cache()` shape: in-process they resolve the real envoys and run
directly (the fast-lane execution); isolated they ship a spec via a new event whose host handler runs the
real op host-side. The exception is `tracer.steer` (and the boundary-write `patch`/`ablate`): steering
touches **no host weights** — only the *delivered* activation — so it needs no host round-trip and no new
event. It rides the existing `Events.SWAP`: a *replacement* write (assign `envoy.output`) ships the
steered value back, which is what makes it cross the boundary where an in-place `[:] =` silently no-ops.

### `tracer.unembed` — host-routed readout (built, 2026-06-14)

The first primitive, and the one that closes the standing weight-read blocker on the **isolated** tier.
`tracer.unembed(residual, norm, head)` projects a residual through the final norm + unembed:

- **In-process / fast lane** (`mediator._isolated_worker` is False): runs the real modules directly —
  `F.linear(norm(residual), head.weight)` (or `head(norm(residual))` with `formulation="module"`).
  This is the same compute the workloads write by hand; `tracer.unembed` just names it.
- **Isolated worker** (weightless dummies): ships the residual VALUE plus the module **paths**
  (`{norm_path, head_path, formulation}`) via a new `Events.UNEMBED` request. The host's
  `handle_unembed_event` resolves the real envoys (`path_to_envoy`), runs the real norm + unembed on the
  **real weights**, and ships back only the logits over the bounce buffer (clone-on-receive both ways,
  like any VALUE/BACKWARD round-trip). **Weights never cross the boundary** — so the readout works on
  the isolated tier without binding the generic warm worker to a model or placing host weight memory in
  the less-trusted worker (the two costs that ruled out shipping/sharing weights — see §1, §7). Paths
  resolve through renames host-side (the wire path is always the real path), so renamed models work.

Touch points: `Events.UNEMBED`; `handle()` dispatch + `handle_unembed_event` (interleaver.py); the
`tracer.unembed` method with the isolated/in-process branch (tracer.py). Shaped exactly like
`Events.CACHE` / `handle_cache_event`.

**Verified (`test_isolated_unembed.py`, all under forced isolation `fast_lane=False`, gpt2 + renamed
model):** single-layer, 3-layer-interleaved, `formulation="module"`, `norm=None`, and renamed-model
readouts all isolated-vs-in-process `max|Δ|=0`; `tracer.unembed` equals the manual
`F.linear(norm(x), head.weight)` it replaces.

### `tracer.steer` — replacement-swap injection (built, 2026-06-15)

`tracer.steer(envoy, direction, alpha=1.0)` adds `alpha * direction` to a module's output residual —
the activation-steering / injection every steering cell does. It is the simplest of the part-2
primitives and the structural opposite of `unembed`: it touches **no host weights**, only the delivered
activation, so it needs **no new event and no isolated/in-process branch**. The method just performs a
*replacement* boundary write:

```python
out = envoy.output
hidden = out[0] if isinstance(out, tuple) else out
steered = hidden + alpha * direction.to(dtype=hidden.dtype, device=hidden.device)
envoy.output = (steered, *out[1:]) if isinstance(out, tuple) else steered
```

The eproperty setter routes that assignment through `mediator.swap` → `Events.SWAP` on **either** tier:
in-process it swaps into the batcher directly; isolated, the worker ships the steered value back over the
existing SWAP path (`pack_cuda` walks the tuple, carrying a `None` tail through untouched). The same code
is therefore correct in-process, on the fast lane, and in the isolated worker.

The point is the **replacement** swap. The hand-written additive form is in-place
(`block.output[:, -1, :] += direction` — the canonical nnsight steering); under isolation that mutates
only the worker's *delivered clone*, no SWAP fires, the host's real activation is untouched, and the
steering silently no-ops (the save of the steered residual even looks right — only the downstream forward
reveals nothing changed). `tracer.steer` makes the steering cross the boundary by construction. Tuple
outputs (most attention modules — `(tensor, None)` here) are replaced whole, steering element `[0]`.

Touch points: just the `tracer.steer` method (tracer.py). No event, no host handler — it reuses
`Events.SWAP` and the eproperty setter.

**Verified (`test_isolated_steer.py`, gpt2 + renamed model):** steering one block, an attention tuple
output, and three blocks at once are all isolated-vs-in-process `max|Δ|=0` and propagate through later
layers; `tracer.steer` equals the manual untuple + whole-tuple replacement. The crux case proves the
motivation under forced isolation: the in-place form leaves the downstream residual == the unsteered
baseline (silent no-op) while `tracer.steer` changes it (steering took effect) to exactly the in-process
result.

### `tracer.patch` — replacement-swap transplant (built, 2026-06-20)

`tracer.patch(envoy, value)` replaces a module's output residual with a precomputed `value` — the
activation-patching / resampling transplant every patching cell does. It is the structural twin of
`tracer.steer`: a boundary write that touches **no host weights** (only the *delivered* activation is
replaced), so it needs **no new event and no isolated/in-process branch** and rides the existing
`Events.SWAP`. The only difference from `steer` is the source of the new value — `steer` computes it from
the delivered activation (`hidden + alpha*direction`), `patch` takes it from the caller:

```python
out = envoy.output
hidden = out[0] if isinstance(out, tuple) else out
value = value.to(dtype=hidden.dtype, device=hidden.device)
envoy.output = (value, *out[1:]) if isinstance(out, tuple) else value
```

The point, as with `steer`, is the **replacement** swap. The hand-written form is in-place
(`block.output[0][:] = clean_act`); under isolation that mutates only the worker's *delivered clone*, no
SWAP fires, and the host's real activation is untouched — the transplant silently no-ops. `tracer.patch`
ships the value back over SWAP, so it crosses the boundary by construction. The value is cast to the
residual's dtype/device, so a value precomputed **on CPU** — the isolation-relevant case, where the
clean/source activation is captured in a prior run outside the trace — transplants cleanly. `value`
replaces the residual whole (element `[0]` for tuple outputs); partial patches construct a full-shape
value (clone + edit a slice), the standard nnsight idiom.

Touch points: just the `tracer.patch` method (tracer.py). No event, no host handler — it reuses
`Events.SWAP` and the eproperty setter.

**Verified (`test_isolated_patch.py`, gpt2 + renamed model):** transplanting into one block, an attention
tuple output, and three blocks at once are all isolated-vs-in-process `max|Δ|=0` and propagate through
later layers; `tracer.patch` equals the manual untuple + whole-tuple replacement (with the same cast). The
crux case proves the motivation under forced isolation: the in-place form leaves the downstream residual
== the unpatched baseline (silent no-op) while `tracer.patch` changes it (transplant took effect) to
exactly the in-process result.

### `tracer.ablate` — replacement-swap knockout (built, 2026-06-20)

`tracer.ablate(envoy, mode="zero")` replaces a module's output residual with a baseline — the lesion-study
knockout every ablation cell does. Same shape as `patch`/`steer`: a boundary write riding `Events.SWAP`,
no new event, no isolated/in-process branch. Two self-contained modes:

```python
out = envoy.output
hidden = out[0] if isinstance(out, tuple) else out
if mode == "zero":
    ablated = torch.zeros_like(hidden)
elif mode == "mean":                                         # within-sequence mean
    ablated = hidden.mean(dim=-2, keepdim=True).expand_as(hidden).contiguous()
else:
    raise ValueError(...)                                    # no silent wrong-ablation
envoy.output = (ablated, *out[1:]) if isinstance(out, tuple) else ablated
```

`mode="mean"` is the **within-sequence** mean (each position → the per-example mean over the token
dimension), the only mean derivable from a single forward. The reduction decision that §6 flagged resolves
here: **reference-distribution** mean ablation (the mean activation over a *dataset*, per
`docs/patterns/ablation.md`) is not a single-forward quantity, so it is precomputed and transplanted via
`tracer.patch(envoy, mean_act)` — not this mode. Keeping the two distinct avoids a silent-semantics trap
(a user expecting dataset-mean getting sequence-mean). An unknown mode raises `ValueError` rather than
silently picking a baseline. Under isolation the worker computes the mean from its delivered clone (==
the host's real activation) and ships the result back over SWAP, so isolated == in-process bit-identically.

Touch points: just the `tracer.ablate` method (tracer.py). No event, no host handler.

**Verified (`test_isolated_ablate.py`, gpt2 + renamed model):** zero- and mean-ablating a block, an
attention tuple output, and the renamed model are all isolated-vs-in-process `max|Δ|=0` and change the
downstream forward vs the un-ablated baseline (the knockout took effect across the boundary);
`tracer.ablate` equals the manual `zeros_like` / mean-over-seq replacement; an unknown mode raises
`ValueError`. The crux case proves the motivation under forced isolation: the in-place zero leaves the
downstream residual == the un-ablated baseline (silent no-op) while `tracer.ablate` changes it to exactly
the in-process result.

## 7. What was deliberately deferred

- **The process-global `sys.addaudithook` backstop.** Its own failure mode (a leaked thread-local flag
  arms it during the model's *own* forward → server-wide outage) makes it net-negative when the static
  default-deny gate already makes imports / `open` / `exec` / `socket` statically impossible in
  fast-laned code. Documented as future hardening; the static gate is the confirmation.
- **A frozen-namespace `Compartment`** (SES-style) for fast-lane execution. The first slice relies on
  the static pass + the `trust` cordon; namespace shadowing is a later refinement.
- **The last primitive** (§6) — `capture`. `unembed`, `steer`, `patch`, and `ablate` are built; the
  boundary-write trio (`steer`/`patch`/`ablate`) all ride `Events.SWAP` with no new handler. `capture`
  remains because it needs a run↔run handoff (not a single boundary write) and would collide with the
  existing `Tracer.capture(frame)` AST method — both a new mechanism and a naming decision.

## 8. Verification

- **Classifier units** (`test_fastlane_classifier.py`, GPU-free) — 17/17: logit-lens / steering /
  patching / attribution shapes and **renamed** structures classify FAST; imports / `while` /
  unresolved-call / `open` → ISOLATE; introspection → REJECT; the `differentiate`/`in_place`/
  `touches_host_weights` flags are set correctly.
- **Fast-lane end-to-end** (`test_fast_lane.py`, gpt2 + a renamed model) — 6/6: the weight-reading
  logit lens is bit-identical on the fast lane (`max|Δ|=0`) **and raises under forced isolation**
  (`fast_lane=False`) — proving the fast lane is the enabling tier; in-place steering bit-identical;
  renamed-model lens bit-identical; a footgun routes off the fast lane and the host survives; an
  introspection escape is rejected; a runaway loop is killed by the watchdog and the host survives.
- **Isolated steer** (`test_isolated_steer.py`, gpt2 + a renamed model) — 6/6: steering a block, an
  attention tuple output, and three blocks at once are isolated-vs-in-process `max|Δ|=0`; `tracer.steer`
  equals the manual whole-tuple replacement; and the crux — under forced isolation the in-place form is a
  no-op (downstream == unsteered baseline) while `tracer.steer` takes effect and matches in-process.
- **Isolated patch** (`test_isolated_patch.py`, gpt2 + a renamed model) — 6/6: transplanting into a block,
  an attention tuple output, and three blocks at once are isolated-vs-in-process `max|Δ|=0`; `tracer.patch`
  equals the manual whole-tuple replacement (with the dtype/device cast); and the crux — the in-place
  transplant is a no-op under isolation while `tracer.patch` takes effect and matches in-process.
- **Isolated ablate** (`test_isolated_ablate.py`, gpt2 + a renamed model) — 7/7: zero/mean knockout of a
  block, an attention tuple output, and the renamed model are isolated-vs-in-process `max|Δ|=0` and change
  the downstream forward vs the un-ablated baseline; `tracer.ablate` equals the manual `zeros_like` /
  mean-over-seq replacement; an unknown mode raises `ValueError`; and the crux — the in-place zero is a
  no-op under isolation while `tracer.ablate` takes effect and matches in-process.
- **Existing isolated WORKER path** — 9/9 still bit-identical, pinned with `fast_lane=False` so they
  keep exercising the worker (otherwise the simple read/swap/save cells would now fast-lane).
- **In-process core** — 51 passed (the default in-process path is untouched).
