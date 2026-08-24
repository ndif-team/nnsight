---
title: A Module Value Was Never Provided
one_liner: "OutOfOrderError at the end of a run: a worker was still waiting for a location the model never reached — the module didn't fire, or an iter loop outran the model (a UserWarning, not an error)."
tags: [error, execution-order, dangling-worker]
related: [docs/errors/out-of-order-error.md, docs/errors/cannot-access-outside-interleaving.md, docs/usage/iter-all-next.md, docs/usage/scan.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/interleaver.py:605, src/nnsight/intervention/interleaver.py:638, src/nnsight/intervention/interleaver.py:646]
---

# A Module Value Was Never Provided

This is the **late** flavor of `OutOfOrderError`: the model ran to the end while a
worker was still parked, waiting for a location that never came. Same exception
class as [out-of-order-error.md](out-of-order-error.md), different code path — here
`Interleaver.check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:605`)
surfaces it after the forward pass finishes.


## Symptom

Raised as an exception when a worker is still waiting after the model finished:

```
nnsight.intervention.interleaver.OutOfOrderError: 'model.transformer.h.5.output.i0' was requested but the model already ran past it
```

Downgraded to a **warning** (not an exception) when the unmet request came from
inside a `tracer.iter` loop that asked for more iterations than the model ran:

```
UserWarning: 'model.transformer.h.11.output.i2' was never reached: the model ran fewer iterations than the loop requested. Values from reached iterations are kept.
```

The location reads `<envoy.path>.<output|input>.i<occurrence>` — e.g.
`model.transformer.h.5.output.i0` is layer 5's output on the first (i0) visit.

## Cause — two reasons a location is never reached

### 1. The module never fired

The module path exists in `print(model)`, but its `forward` was **not called** on
this input, so the hook that would deliver its value never ran. The worker parks
forever and `check_dangling_mediators` throws `OutOfOrderError` at the end of the
run. Common cases:

- **Dropout / other eval-mode-disabled layers** — bypassed under `model.eval()`.
- **Branch paths** — `if self.config.use_flash_attention: ... else: ...` runs only one side; a submodule on the path not taken never fires.
- **Modules not reached by this input** — auxiliary heads, MoE experts not routed to, a vision encoder on a text-only query.
- **Children of a skipped module** — after `module.skip(value)`, that module's submodules don't run.

nnsight can't know in advance whether a module fires. Verify it yourself
(fix section).

### 2. An `iter` loop outran the model

`for step in tracer.iter[:]` (or `tracer.iter[:N]` / `tracer.all()`) with more steps
than the model generated leaves the final over-run request dangling. Because the
worker's `iteration` is non-zero, `check_dangling_mediators`
(`src/nnsight/intervention/interleaver.py:646`) throws into the worker to unwind it
(running its `finally` blocks) but **warns instead of raising** — values from steps
that were reached are kept. This is expected, not user error.

Note: an open-ended `iter[:]` that outruns the model unwinds the loop **and every
line after it** — see the note in [docs/usage/iter-all-next.md](../usage/iter-all-next.md).
Prefer a bounded `iter[:N]` when you need code to run after the loop.

## Fix

### Confirm the module actually fires — `model.scan(...)`

`scan` runs the forward with fake tensors (shapes only), hitting the same code
paths the real forward will:

```python
import nnsight

with model.scan("Hello"):
    print(model.transformer.h[5].output.shape)   # raises here too if it never fires
```

See [docs/usage/scan.md](../usage/scan.md).

### Read the forward source

`print(model.transformer.h[0].source)` (see [docs/usage/source.md](../usage/source.md))
or `inspect.getsource(type(module).forward)` shows what actually runs — branches,
eval-mode short-circuits, and dispatch hide behind the module tree.

### Bound your iteration

```python
# WRONG — iter[:] outruns a 3-token run; the over-run request warns and unwinds the loop
with model.generate("Hi", max_new_tokens=3) as tracer:
    for step in tracer.iter[:8]:
        hs = model.transformer.h[-1].output.save()
```

```python
# FIXED — bound the loop to the steps you actually generate
with model.generate("Hi", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:
        hs = model.transformer.h[-1].output.save()
```

### Don't read children of a skipped module

```python
# WRONG — h[1] is skipped, so its mlp never fires
with model.trace("Hi"):
    model.transformer.h[1].skip(model.transformer.h[1].input)
    inner = model.transformer.h[1].mlp.output.save()   # OutOfOrderError
```

```python
# FIXED — read a submodule of a layer that actually runs
with model.trace("Hi"):
    model.transformer.h[1].skip(model.transformer.h[1].input)
    inner = model.transformer.h[2].mlp.output.save()
```

## Related

- [out-of-order-error.md](out-of-order-error.md) — the eager "asked in the wrong order" flavor of the same class.
- [docs/usage/scan.md](../usage/scan.md) — verify a module fires.
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — bounded vs open-ended iteration.
- [docs/usage/source.md](../usage/source.md) — read inside-the-forward execution order.
