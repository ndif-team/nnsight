# vLLM-path construct gaps: iteration / barrier / session / edit / scan

> Diagnosed 2026-06-11 on `dev` @ `944c805`, python 3.12.13, vllm 0.15.1, torch 2.9.1+cu128,
> GPT-2, single GPU. Discovered by the interp-serve-bench micro tier (one minimal probe per
> primitive × backend), then root-caused here. All five re-verified identical on vllm **0.19.1**
> (`ndif-dev` env) and on `pp-on-dev` @ f12891f — branch-independent failures in the shared
> integration layer. Verify any fix on 0.19.1, not the stale 0.15.1 pin.
>
> All repros below are standalone scripts: save, then
> `CUDA_VISIBLE_DEVICES=<gpu> python <script>.py` with an env where `nnsight` resolves to this
> checkout (e.g. `/disk/u/zikai/anaconda3/envs/ndif-dev/bin/python` with `PYTHONPATH=<checkout>/src`).
> Every script needs the `if __name__ == "__main__"` guard (vLLM EngineCore uses spawn). Wrap
> runs in `timeout 600`.

| # | construct | sync engine | async engine | status |
|---|---|---|---|---|
| 1 | unbounded `for step in tracer.iter[:]` | all saves lost (`UnboundLocalError`) | all saves lost (finished output has no `.saves`) | **FIXED 2026-06-11** — see the fix subsection below. Was undisclosed: `docs/models/vllm.md` documents the idiom as working; tests only covered the deprecated `with tracer.iter[...]` form, sync-only |
| 2 | `tracer.barrier(n)` across invokes | **silent**: trace exits cleanly, saved dict EMPTY | loud (stacks with the async multi-prompt gate) | Open — `barrier-vllm-not-shared.md`; the silent sync flavor is new. Re-verified unchanged after the iteration fix |
| 3 | `model.session()` un-saved cross-trace var | `UnboundLocalError` (misleading; on 0.19.1 a clearer `NameError` naming the un-saved upstream var) | broken for everything (no drain point) | Open — known gap, never localized; **saved-value flow WORKS on sync** |
| 4 | `model.edit()` then trace | `PicklingError: source code unavailable` | same | **GATED 2026-06-11** — `VLLM.edit()` raises `NotImplementedError` at creation; `VLLM.trace()` backstops on pending `_default_mediators` (covers `import_edits`). Real support remains open — see the LoRA-shaped design below |
| 5 | `model.scan()` | `Unexpected keyword argument 'hook'` | same | **GATED 2026-06-11** — `VLLM.scan()` raises `NotImplementedError`. Real fake-mode support remains open |

Remaining fix order: barrier registry → session implicit-saves (design items).

---

## 1. Unbounded `tracer.iter[:]` silently loses ALL saves — FIXED

### Symptom

The documented multi-token idiom (`docs/models/vllm.md` "Multi-token generation with
`tracer.iter`") returns nothing: on sync the saved variable is never bound after the trace; on
async the finished `RequestOutput` carries no `.saves` attribute at all. Generation itself
completes normally. **Bounded slices work** (`tracer.iter[0:3]` passes on both engines), so this
is specifically the unbounded forms (`iter[:]`, for-form `tracer.all()`).

### Repro

```python
"""iter_repro.py — expect: sync/for-bounded OK, sync/for-unbounded UnboundLocalError;
async/for-unbounded 'no saves'. After the fix, all OK with len=3."""
import sys

MSG = "Madison Square Garden is located in the city of"


def report(name, fn):
    try:
        print(f"CASE {name}: OK -> {fn()}", flush=True)
    except Exception as e:
        m = str(e).strip().splitlines()
        print(f"CASE {name}: FAIL -> {type(e).__name__}: {m[-1] if m else ''}", flush=True)


def main():
    mode = sys.argv[1]  # "sync" | "async"
    from nnsight.modeling.vllm import VLLM
    model = VLLM("openai-community/gpt2", mode=mode, dispatch=True, gpu_memory_utilization=0.15)

    if mode == "sync":
        def bounded():
            with model.trace(MSG, temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
                rows = list().save()
                for step in tracer.iter[0:3]:
                    rows.append(model.logits)
            return f"len={len(rows)}"

        def unbounded():
            with model.trace(MSG, temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
                rows = list().save()
                for step in tracer.iter[:]:
                    rows.append(model.logits)
            return f"len={len(rows)}"

        report("sync/for-bounded", bounded)
        report("sync/for-unbounded", unbounded)
    else:
        import asyncio, contextvars
        loop = asyncio.new_event_loop()

        def unbounded_async():
            async def go():
                with model.trace(MSG, temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
                    rows = list().save()
                    for step in tracer.iter[:]:
                        rows.append(model.logits)
                last = None
                async for output in tracer.backend:
                    last = output
                if not hasattr(last, "saves"):
                    raise RuntimeError("finished output has NO saves")
                return {k: len(v) for k, v in last.saves.items()}
            return contextvars.copy_context().run(loop.run_until_complete, go())

        report("async/for-unbounded", unbounded_async)


if __name__ == "__main__":
    main()
```

### Root cause (three pieces stacking)

1. **No stop bound on vLLM.** The unbounded loop terminates via `mediator.all_stop` or
   `interleaver.default_all` (`src/nnsight/intervention/tracing/iterator.py:253-263`).
   `default_all` is set only on the HF path (`src/nnsight/modeling/language.py:192`, from
   `max_new_tokens`) and diffusion (`diffusion.py:432`) — never on vLLM. `all_stop` is never
   assigned anywhere (only initialized from a constructor param nobody passes). So the loop runs
   past the last generation step and blocks requesting a value that will never fire.
2. **Cancelation unwinds the body.** At request end, `finalize_mediators`
   (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:227`) calls `mediator.cancel()`,
   which answers the blocked read with `Cancelation()`
   (`src/nnsight/intervention/interleaver.py:1013`); the exception is swallowed as internal
   bookkeeping (`interleaver.py:1181`). The body thread unwinds.
3. **Saves only exist after `push()`.** Worker-side, `mediator.info.frame` is a fake frame
   populated ONLY by `mediator.push()` (`interleaver.py:1410-1430`), and `collect_saves`
   (`GPUModelRunner.py:257`) reads `info.frame.f_locals`. The for-form body has exactly one final
   push — after the loop — which the unwind skips. Everything is lost.

**Why nobody noticed:** the deprecated `with tracer.iter[...]:` form is immune *by accident* —
its compiled body does `pull()`/`push()` per iteration (`iterator.py:311-316`), checkpointing
saves each step. `tests/test_vllm.py` uses only the with-form, only on the sync engine; the async
test class (`TestAsyncEngine`) never saves under iteration. So the recommended form
(`IteratorTracer.execute` itself warns to use it) is untested on vLLM.

### Fix (landed 2026-06-11, verified on vllm 0.19.1)

Three changes — the diagnosis above anticipated the first two; the third was uncovered while
verifying the eos-early tail:

1. **Publish saves on the exception path.** `Mediator.exception()` now calls `self.push()`
   before posting the EXCEPTION event (`src/nnsight/intervention/interleaver.py`), mirroring
   `end()` and `stop()`. Any unwind of the body — Cancelation at request teardown or a deferred
   user exception — ships already-computed saves. This makes the "ships this exception back
   alongside any saves that were already collected" comment true. (Implemented at the protocol
   level rather than as a compiled-source `try/finally`: the for-form loop body is part of the
   invoker's compiled function, whose except branch already routes through `exception()`.)
2. **Per-request stop bound.** `NNsightRequestHelper.process_new_reqs`
   (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py`) sets
   `mediator.all_stop = sampling_params.max_tokens` (when not None) at worker-side registration.
   Per-mediator `all_stop`, NOT interleaver-wide `default_all` — the worker's interleaver is
   engine-lifetime and concurrently holds mediators from requests with different `max_tokens`.
   With the bound, the loop breaks cleanly through `end()` (the same path bounded slices take).
3. **`Mediator.cancel()` now synchronizes with the worker's unwind.** `Mediator.Value` is a
   slot, not a queue: `get()` does NOT block (returns the current value or None); blocking is a
   separate `wait()`. `cancel()`'s final `event_queue.get()` had no `wait()`, so after sending
   the Cancelation it raced ahead — `collect_saves` read the mediator's frame BEFORE the
   worker's except branch pushed. Without this, change 1 publishes the saves too late to ship.
   `cancel()` now does `wait()` then `get()`, blocking until the worker acks (the EXCEPTION
   event posted after its push).

Behavior note for the eos-early tail: with `stop=["York"]` and `max_tokens=10` on the standard
Madison-Square-Garden prompt, the saved list comes back with **4** entries, not 2 — vLLM's
incremental detokenizer holds back text finalization for a couple of steps past the stop-string
match, so extra forwards run before the request is marked finished. Every step that runs ships
its save; the exact count is a vLLM detokenizer detail, not an nnsight one.

Verified: the repro above flips to `len=3` on both engines; eos-early partial saves survive;
`tests/test_vllm.py` 32 passed / 8 skipped (Ray) including 4 new for-form tests
(`test_bounded_for_iteration`, `test_unbounded_for_iteration`,
`test_unbounded_for_iteration_stop_string`, `test_async_unbounded_for_iteration_saves`);
`tests/test_lm.py` pre/post failure sets identical except one pre-existing failure
(`test_source_operation_not_found`) that now passes; barrier repro re-run unchanged (no
deadlock from the cancel synchronization).

---

## 2. `tracer.barrier(n)` — sync engine drops post-barrier saves SILENTLY

### Symptom

Already documented in `barrier-vllm-not-shared.md` (each invoke deserializes a private Barrier
copy → participant count never reaches n → no-op branch → workers blocked/abandoned →
post-barrier code dropped). New here: on the **sync** engine the failure is **silent** — the
trace exits cleanly and the saved container comes back EMPTY, no error of any kind. (On async it
stacks with the multi-prompt submission gate and at least fails loudly with missing saves.)

### Repro

```python
"""barrier_repro.py — expect: invoke-control OK with keys ['a','b'];
barrier case OK-but-EMPTY (keys=[], no error) = the silent drop. After the fix: keys=['patched']."""
CLEAN, CORRUPT = "The Eiffel Tower is in", "The Colosseum is in"


def main():
    from nnsight.modeling.vllm import VLLM
    model = VLLM("openai-community/gpt2", mode="sync", dispatch=True, gpu_memory_utilization=0.15)

    # control: plain two-invoke trace works on sync
    with model.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
        res = dict().save()
        with tracer.invoke(CLEAN):
            res["a"] = model.logits
        with tracer.invoke(CORRUPT):
            res["b"] = model.logits
    print(f"invoke-control: keys={sorted(res)}", flush=True)

    # barrier: cross-invoke patch
    with model.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
        res = dict().save()
        barrier = tracer.barrier(2)
        with tracer.invoke(CLEAN):
            out = model.transformer.h[5].output
            clean_hs = (out[0] if isinstance(out, tuple) else out)[-1:, :]
            barrier()
        with tracer.invoke(CORRUPT):
            barrier()
            out = model.transformer.h[5].output
            hs = (out[0] if isinstance(out, tuple) else out).clone()
            hs[-1:, :] = clean_hs
            model.transformer.h[5].output = (hs, *out[1:]) if isinstance(out, tuple) else hs
            res["patched"] = model.logits
    print(f"barrier: keys={sorted(res)}  <- EMPTY = silent post-barrier drop", flush=True)


if __name__ == "__main__":
    main()
```

### Root cause / fix

As documented in `barrier-vllm-not-shared.md`: the Barrier object is serialized per invoke, so
each mediator holds a private copy with its own participants set; the count never reaches n and
post-barrier code never runs. Preferred fix from that doc stands: an **interleaver-owned barrier
registry keyed by a serialization-stable id** (alternative: graft the Barrier into canonical
globals). The `finally`-push from gap 1 would additionally convert "saved dict silently empty"
into "partial saves survive", but the registry is the real fix.

---

## 3. `model.session()` — un-saved cross-trace variable flow is broken (saved flow works)

### Symptom

The session contract is that trace-body locals flow across traces. On vLLM (sync), only the
`.save()`d half of the contract holds:

- ✅ `.save()` inside a session trace + read after session exit → real tensor, works.
- ✅ assigning a saved value into a pre-existing outer container inside the session body → works.
- ❌ an **un-saved** variable from trace 1 consumed in trace 2 → the second trace dies and the
  session surfaces `UnboundLocalError: cannot access local variable '<your-saved-var>'` —
  misleading, since the actual problem is the un-saved upstream variable.
- ❌ async engine: broken for everything — there is no drain point (`async for ... in
  tracer.backend` cannot be compiled inside a captured session body: "'async for' outside async
  function"; the tracer handles are also body-locals and don't survive to the caller frame).

### Repro

```python
"""session_repro.py — expect: CASE A FAIL (UnboundLocalError), CASE B OK.
After a fix, A either works (implicit saves) or raises a clear 'save() required' error."""
P = "The Eiffel Tower is in"


def main():
    from nnsight.modeling.vllm import VLLM
    model = VLLM("openai-community/gpt2", mode="sync", dispatch=True, gpu_memory_utilization=0.15)

    try:
        with model.session():
            with model.trace(P, temperature=0.0, top_p=1.0, max_tokens=1):
                v = model.logits                      # NOT saved — relies on session var flow
            with model.trace(P, temperature=0.0, top_p=1.0, max_tokens=1):
                diff = (model.logits - v).abs().max().save()
        print(f"CASE A (unsaved cross-trace): OK |diff|={float(diff.detach()):.2e}", flush=True)
    except Exception as e:
        m = str(e).strip().splitlines()
        print(f"CASE A (unsaved cross-trace): FAIL {type(e).__name__}: {m[-1]}", flush=True)

    try:
        with model.session():
            with model.trace(P, temperature=0.0, top_p=1.0, max_tokens=1):
                lg = model.logits.save()
        print(f"CASE B (saved + post-exit read): OK shape={tuple(lg.shape)}", flush=True)
    except Exception as e:
        m = str(e).strip().splitlines()
        print(f"CASE B (saved + post-exit read): FAIL {type(e).__name__}: {m[-1]}", flush=True)


if __name__ == "__main__":
    main()
```

### Root cause

On HF the trace body executes in-process and `push()` returns ALL body locals to the session
frame — un-saved flow is free. On vLLM the body executes in the EngineCore worker and only values
registered in `Globals.saves` ship back (`GPUModelRunner.collect_saves` filters frame locals on
`id(value) in Globals.saves`). The un-saved variable never materializes client-side; the second
trace's body references a value that doesn't exist, dies, and its own saved variable never binds —
hence the misleading `UnboundLocalError` naming the *downstream* variable.

### Fix proposal

- **Short-term:** detect "session + vLLM + cross-trace reference to an un-saved trace local" and
  raise a clear error instructing `.save()` — converts a confusing UnboundLocalError into a
  documented requirement. (Cheapest version: document it; the saved-flow path already works.)
- **Real fix:** on the vLLM path, implicitly add session-trace body locals to `saved_names` so
  the worker ships them (correct but ships every local tensor; can be narrowed to locals actually
  referenced by later session code via the dependency analysis the cross-invoker already does).
- **Async sessions** need a design decision (session-owned draining); bigger than a patch.

---

## 4. `model.edit()` — stored mediator can't serialize to the worker; crash is PROTECTIVE — GATED

### Symptom

Tracing an edited model on vLLM (sync or async) fails at submit time with
`PicklingError: Cannot serialize function '__nnsight_tracer_N__': source code unavailable`.

### Repro

```python
"""edit_repro.py — expect: PicklingError at the edited trace.
After a REAL fix: edit applied (max|Δ| large); after a naive serialization-only fix:
DANGER — watch for 'edit silently dropped' (max|Δ|≈0)."""
import torch

P = "The Eiffel Tower is in"


def main():
    from nnsight.modeling.vllm import VLLM
    model = VLLM("openai-community/gpt2", mode="sync", dispatch=True, gpu_memory_utilization=0.15)

    with model.trace(P, temperature=0.0, top_p=1.0, max_tokens=1):
        base = model.logits.save()

    with model.edit() as edited:
        out = edited.transformer.h[6].attn.output
        if isinstance(out, tuple):
            edited.transformer.h[6].attn.output = (torch.zeros_like(out[0]), *out[1:])
        else:
            edited.transformer.h[6].attn.output = torch.zeros_like(out)

    with edited.trace(P, temperature=0.0, top_p=1.0, max_tokens=1):
        ed = edited.logits.save()

    moved = (base.float().cpu() - ed.float().cpu()).abs().max().item()
    verdict = "APPLIED" if moved > 1e-3 else "SILENTLY DROPPED  <- the dangerous outcome"
    print(f"edit moved logits by max|delta|={moved:.3f} ({verdict})", flush=True)


if __name__ == "__main__":
    main()
```

### Root cause

1. `EditingBackend` stores `Mediator(fn, info)` on `envoy._default_mediators` **without
   attaching `fn.__source__`** (`src/nnsight/intervention/backends/editing.py:20-22`). Normal
   trace mediators get source attached (`tracer.py:677`, `vllm.py:380`); edit mediators don't.
2. The edited envoy is a shallow copy — NOT covered by the persistent-id registry (only raw
   `_module`s and the Interleaver are, `src/nnsight/modeling/mixins/remoteable.py:115-118`) — so
   when the edited trace's body references it, the envoy pickles **by value** into the mediator's
   globals, dragging `_default_mediators` → the sourceless function → the source-based pickler
   refuses (`src/nnsight/intervention/serialization.py:780-791`).
3. Deeper: even with `__source__` attached, default mediators are only appended to
   `tracer.mediators` client-side (`tracer.py:414-418`); `_serialize_mediators` ships only
   *input* mediators via `extra_args` (`vllm.py:376-405`) and the worker registers only what
   arrives there — so the edit would most likely be **silently dropped**. The PicklingError is
   currently the only thing standing between the user and a silently-unedited model.

### Gate (landed 2026-06-11)

- `VLLM.edit()` raises `NotImplementedError` at creation time — the earliest, clearest point.
- `VLLM.trace()` backstops on non-empty `self._default_mediators` — covers edits arriving
  without `edit()` (`import_edits()` loads mediators from dill, `envoy.py:425`) and pins the
  protective behavior: a future serialization-only "fix" would hit this gate instead of
  silently dropping the edit. `tests/test_vllm.py::TestUnsupportedConstructs` locks both in.
- **`inplace=True` probed (2026-06-11): fails identically** — the trace body's reference to the
  model pickles the envoy by value in both forms (only raw `_module`s and the Interleaver are
  persistent-id'd), dragging `_default_mediators` along. Loud in both forms; no pre-existing
  silent hole.

### Real fix (open) — three channels, three different batching stories

What "editing" means on the transformers path is not one mechanism but three, and they behave
completely differently under vLLM's continuous batching. Any real edit support must decide
per channel:

**Channel 1 — `edit()` callback replay (the `_default_mediators` mechanism).** The edit body
is compiled into an intervention function and replayed per trace as an extra mediator — a
transient activation rewrite during each forward; weights and module tree untouched. This is
the channel the batching machinery already handles: nnsight's input mediators ARE per-request
shipped callbacks scoped by `batch_group` (`batcher.narrow`/`swap` confine reads and writes to
the request's rows — how batched interventions work today). Supporting it is wiring, not
research, and vLLM's LoRA shows the transport pattern (ship a per-request *reference*, cache
heavy state worker-side):
  1. attach `fn.__source__` in `EditingBackend` (serializability);
  2. register the edit once worker-side keyed by a source hash (`NNsightRequestHelper` already
     keeps cross-request state), each request's `extra_args` carrying just the edit-id list —
     or inline per request as a v1;
  3. instantiate a FRESH mediator per request from the cached function (mediators are
     stateful), assign the request's `batch_group`, register BEFORE the input mediator
     (matching HF's prepend order in `InterleavingTracer.compile`);
  4. give the edited envoy persistent-id treatment (or canonicalize body references to the
     root envoy + edit set) so the by-value pickle stops happening.
  **Do NOT fix only the serialization** (one line in `EditingBackend`) — that converts a loud
  failure into a silent one. Use the repro's effect-size check to verify whichever fix lands.

**Channel 2 — module attachment (true in-place structural mutation, computationally inert).**
`envoy.attachment = SomeModule()` routes through `Envoy.__setattr__` → `_add_envoy` →
`setattr(self._module, name, module)` (`envoy.py:736`) — it permanently mutates the real
`nn.Module` tree (`clear_edits()` does not undo it). This is the SAE/probe pattern, exercised
by `tests/test_lm.py::TestEditing::test_edit_with_attachment`, and the canonical HF edit
workflow is a HYBRID: attach in-place, then `edit()` to wire the attachment into the forward
via a callback. Under batching this is engine-wide state but batch-SAFE, because the attached
module never runs in the base forward — only a request's own callback invokes it. Supporting
it on vLLM needs a one-time worker-side mutation mechanism (RPC at registration, analogous to
loading adapter weights) — separate design work from channel 1, tractable. What attachment
does on vLLM TODAY is untested (the client envoy wraps the meta model; the attachment may
partially ride the by-value envoy pickle or silently not exist worker-side) — probe before
designing.

**Channel 3 — module replacement / weight mutation (true in-place, computationally active).**
The same `__setattr__` makes `envoy.mlp = CustomMLP()` replace the child on the real model,
and non-module writes mirror to it (`setattr(self._module, key, value)`, `envoy.py:1067`) —
so parameter assignment is genuine weight mutation. Works on single-process HF. Under
continuous batching this is the genuinely hard case and is OUT OF SCOPE for edit support:
one shared weight copy means co-batched requests cannot see different weights (that requires
LoRA-style structured deltas with custom batched kernels), and even engine-wide application
contaminates in-flight requests mid-generation, stales KV/prefix-cache entries computed under
the old weights, and invalidates CUDA graphs. The only sound forms are drain-and-swap weight
reloading (vLLM's RLHF-style update path; engine-wide, between batches) or native LoRA — both
outside the edit-tracer abstraction.

---

## 5. `model.scan()` — dies in SamplingParams construction, never reaches fake mode — GATED

### Symptom

`model.scan(prompt)` on VLLM raises `Unexpected keyword argument 'hook'` (msgspec's wording).

### Repro

```python
"""scan_repro.py — expect: 'Unexpected keyword argument hook'.
After a gate: a clear NotImplementedError; after a real impl: shape printed with no kernels run."""
import nnsight

P = "The Eiffel Tower is in"


def main():
    from nnsight.modeling.vllm import VLLM
    model = VLLM("openai-community/gpt2", mode="sync", dispatch=True, gpu_memory_utilization=0.15)
    try:
        with model.scan(P):
            shp = nnsight.save(tuple(model.transformer.h[0].output[0].shape))
        print(f"scan OK: shape={tuple(shp)}", flush=True)
    except Exception as e:
        m = str(e).strip().splitlines()
        print(f"scan FAIL: {type(e).__name__}: {m[-1] if m else ''}", flush=True)


if __name__ == "__main__":
    main()
```

### Root cause

`Envoy.scan()` injects `hook=True` into the model call (`src/nnsight/intervention/envoy.py:329`).
On the vLLM path, ALL tracer kwargs are forwarded into `NNsightSamplingParams(**kwargs)`
(`src/nnsight/modeling/vllm/vllm.py:261` and `:280`), and msgspec rejects the unknown field. So
scan dies preparing the input, before `ScanningTracer.execute`'s fake mode is ever entered. Even
with `hook` stripped, the fake-mode forward would still need vLLM's forward context (attention
metadata) — unwired.

### Gate (landed 2026-06-11)

`VLLM.scan()` raises `NotImplementedError` pointing to the working alternatives (a real
`max_tokens=1` trace, or scanning the HuggingFace `LanguageModel` twin). Locked in by
`tests/test_vllm.py::TestUnsupportedConstructs::test_scan_raises`. Note `VLLM.interleave`
already carves out `ScanningTracer` (skips engine dispatch) — partial intent existed, but the
kwarg path and the fake-mode forward context were never wired.

### Real fix (open)

Implement scan locally on the meta model under `FakeTensorMode`, reusing the dummy-run
machinery vLLM itself uses for profiling (`set_forward_context` + dummy attention metadata).
Moderate effort; real value (shape validation with no GPU).

---

## Verification checklist (for the fixing session)

1. Each repro above flips to its "after the fix" output.
2. `tests/test_vllm.py` gains: async × `for step in tracer.iter[:]` saves; sync × unbounded
   for-form; a barrier cross-invoke value test; session saved/un-saved cases; an edit
   **effect-size** test (edited != baseline AND edited == in-trace ablation — never just
   "doesn't crash").
3. The interp-serve-bench micro tier re-run flips the corresponding rows
   (`/disk/u/zikai/interp-serve-bench/scripts/micro.py --backend vllm_async`) and the expected-
   state mechanism reports the flips as surprises — that map is the regression detector.
4. The eos-before-`max_tokens` tail for iteration: generate with a stop string and confirm
   bounded-but-truncated loops still return partial saves (needs the `finally`-push).
