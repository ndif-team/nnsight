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
| `tracer.steer` | `(envoy, direction, alpha)` | boundary write (injection) | always a replacement swap — fixes in-place's silent no-op under isolation | designed |
| `tracer.patch` | `(envoy, value)` | boundary write (transplant) | whole-tuple replacement | designed |
| `tracer.ablate` | `(envoy, mode)` | boundary write (injection) | zero/mean knockout | designed |
| `tracer.capture` | `(value) → handle` | read + run↔run transfer | cross-trace handoff; non-transmittable → clean fail, not silent drop | designed |

Each mirrors the existing `tracer.cache()` shape: in-process it resolves the real envoys and runs
directly (the fast-lane execution); isolated it ships a spec via a new event whose host handler runs the
real op host-side.

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

## 7. What was deliberately deferred

- **The process-global `sys.addaudithook` backstop.** Its own failure mode (a leaked thread-local flag
  arms it during the model's *own* forward → server-wide outage) makes it net-negative when the static
  default-deny gate already makes imports / `open` / `exec` / `socket` statically impossible in
  fast-laned code. Documented as future hardening; the static gate is the confirmation.
- **A frozen-namespace `Compartment`** (SES-style) for fast-lane execution. The first slice relies on
  the static pass + the `trust` cordon; namespace shadowing is a later refinement.
- **The remaining primitives' isolated event handlers** (§6) — `steer`/`patch`/`ablate`/`capture`;
  `unembed` is built.

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
- **Existing isolated WORKER path** — 9/9 still bit-identical, pinned with `fast_lane=False` so they
  keep exercising the worker (otherwise the simple read/swap/save cells would now fast-lane).
- **In-process core** — 51 passed (the default in-process path is untouched).
