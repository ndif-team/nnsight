---
title: tracer.barrier() is broken on the vLLM path — the Barrier object isn't shared across invokes
one_liner: On vLLM, each invoke is serialized into its own globals, so each gets a private copy of the Barrier; the participant count never completes, both invokes fall through to the no-op branch, and all post-barrier code is silently dropped.
tags: [internals, dev, vllm, barrier, cross-invoke, bug]
status: DIAGNOSED — not fixed (fix is a design choice; see below). Non-PP; reproduces at tp=1/pp=1.
related: [docs/usage/barrier.md, docs/concepts/batching-and-invokers.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/tracing/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py]
---

# `tracer.barrier()` is broken on the vLLM path

> **Scope.** `barrier` is an advanced cross-invoke primitive (capture an activation in one invoke,
> use it in another). This bug is **vLLM-path-specific** and **not** related to pipeline parallelism —
> it reproduces at `tensor_parallel_size=1, pipeline_parallel_size=1`. It was surfaced while
> stress-testing PP but is orthogonal to that work.

## Symptom

A trace that uses `tracer.barrier(n)` to coordinate two invokes silently drops everything after the
`barrier()` call. The clearest form — a value saved/appended **after** the barrier comes back empty:

```python
with model.trace(temperature=0.0, top_p=1) as tracer:
    barrier = tracer.barrier(2)
    captured = list().save()
    out = list().save()
    with tracer.invoke(clean_prompt):
        captured.append(model.model.layers[8].mlp.output.clone())
        barrier()
    with tracer.invoke(corrupt_prompt):
        barrier()
        model.model.layers[8].mlp.output = captured[0]   # patch
        out.append(model.logits)                          # <-- never lands
# out == []   -> IndexError on out[0]
```

Characterization (Qwen2.5-0.5B, tp1/pp1):

| case | result |
|---|---|
| no barrier, save in 2nd invoke | works (saved-globals union, commit `8c08897`) |
| barrier, append **before** `barrier()` | survives |
| barrier, append **after** `barrier()` | **dropped** |
| barrier, append in both (before+after) | only the **before**-barrier one survives |

No exception, no hang.

## Root cause: the `Barrier` object isn't shared across invokes on vLLM

`tracer.barrier(n)` returns a `Barrier` holding a `participants` set; calling it does
(`tracing/tracer.py`):

```python
def __call__(self):
    mediator = self.model.interleaver.current
    self.participants.add(mediator.name)
    if len(self.participants) == self.n_participants:
        mediator.send(Events.BARRIER, self.participants)   # real barrier
    else:
        mediator.send(Events.BARRIER, None)                # not all here yet
```

On the **vLLM path each `tracer.invoke` is serialized and deserialized into its own per-invoke
`__globals__` on the worker** (`GPUModelRunner.process_new_reqs`). The `Barrier` object is referenced by
every invoke, but it is **not a `.save()`d name**, so the canonical-globals *union* added in `8c08897`
(which reconciles only saved names across invokes) never reconciles it. Each invoke therefore gets its
**own copy** of the `Barrier`, each with its **own** `participants` set.

Consequence: each `barrier()` call adds only its own mediator → `len(participants)` is always 1, never
reaches `n` → **every** invoke takes the `else: send(BARRIER, None)` branch. The real barrier walk in
`Interleaver.handle_barrier_event` only runs when `participants is not None`; with `None` it does nothing
and never `respond()`s to the worker. So both worker threads **block at `barrier()` forever** and are
abandoned when the trace exits — every statement after `barrier()` is silently skipped.

**Evidence** (instrumented `handle_barrier_event`, since reverted): the walk fired with
`participants=None` on *both* invokes' calls — the real participant set was never assembled. Body prints
placed after `barrier()` never executed; the (correctly grafted) saved list was collected with length 0.

## Why it works on `LanguageModel`

`LanguageModel` runs invokes in-process with shared globals — there is no per-invoke serialization, so
`b` is genuinely the same object across invokes and the participant count completes normally. The barrier
docs/examples are written against that path. The break is specific to vLLM's serialize-per-invoke model.

## Fix direction (a design choice — not yet implemented)

The barrier's participant/sync state must be **shared across invokes** on the worker, the same way saved
vars are. Two options:

1. **Interleaver-owned barrier registry, keyed by a serialization-stable barrier id.** `barrier()` counts
   participants in `interleaver.barriers[barrier_id]` instead of the per-copy `Barrier.participants`.
   The id is assigned at `tracer.barrier()` creation and travels with the serialized object. Most robust;
   mirrors how cross-invoke saved state is centralized. Preferred.
2. **Graft the `Barrier` object into canonical globals** (like `8c08897` does for saved names) so every
   invoke references the one canonical `Barrier`. Smaller, but requires detecting barrier objects in
   globals (they are not in `nnsight_saved_names`).

Either way, also make the no-participants path not strand the worker (it currently never `respond()`s).

## Workaround (today)

Don't rely on a `barrier()` to gate state you collect afterward on vLLM: do the save **before** the
barrier, or collect it in the **first** invoke. (Cross-invoke value *sharing* for an in-place patch may
still be affected, since the patch also runs after `barrier()`.)

## Reproduce

`/tmp/pp_stress/cp_isolate.py` (variants v1–v7) and `/tmp/pp_stress/cp_dbg.py` isolate the trigger to the
barrier and show the `participants=None` smoking gun under `CP_DBG=1`.
