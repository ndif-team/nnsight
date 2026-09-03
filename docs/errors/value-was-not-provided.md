---
title: A Module Value Was Never Provided
one_liner: "OutOfOrderError at the end of a run: a worker was still parked on a location the model never reached — the module didn't fire, or a loop asked for more steps than the run made."
tags: [error, execution-order, dangling-worker]
related: [docs/errors/out-of-order-error.md, docs/errors/cannot-access-outside-interleaving.md, docs/usage/iter-all-next.md, docs/usage/scan.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/iterator.py]
---

# A Module Value Was Never Provided

This is the **late** flavor of `OutOfOrderError`: the model ran to the end while a
worker was still parked, waiting for a location that never came. Same exception
class as [out-of-order-error.md](out-of-order-error.md);
`Interleaver.check_dangling_mediators` surfaces it once the forward pass finishes.

## Symptom

A worker parked outside any loop — the module simply never fired:

```
nnsight.intervention.interleaver.OutOfOrderError: 'model.transformer.h.5.output.i0' was requested but the model already ran past it
```

A worker parked inside a loop that named an end — `iter[:8]`, `iter[2]`,
`iter[[0, 2, 7]]` — which the run did not reach:

```
nnsight.intervention.interleaver.OutOfOrderError: 'model.transformer.h.11.output.i3' was never reached: the loop asked for iteration 3 of 'model.transformer.h.11.output' and the run reached it 3 times, so the loop was cut short and nothing after it ran. Bound the loop to the iterations the run makes (`min_new_tokens=` holds a generation to a step count), or loop with `tracer.all()` and put what follows the loop after the `with` block.
```

A worker parked inside an **open** loop — `tracer.iter[:]` or `tracer.all()` —
which ends by asking for a step the run does not make. That one is a warning, not
an exception:

```
UserWarning: 'model.transformer.h.11.output.i3' was never reached: an open `tracer.iter[:]` / `tracer.all()` loop ends by asking for a step the run does not make. Values saved inside the loop are kept; the statements after it did not run.
```

The location reads `<envoy.path>.<output|input>.i<occurrence>` — e.g.
`model.transformer.h.5.output.i0` is layer 5's output on the first (i0) visit.

## Cause — two reasons a location is never reached

### 1. The module never fired

The module path exists in `print(model)`, but its `forward` was **not called** on
this input, so nothing ever delivered its value. The worker parks forever and
`check_dangling_mediators` throws `OutOfOrderError` at the end of the run. Common
cases:

- **Dropout / other eval-mode-disabled layers** — bypassed under `model.eval()`.
- **Branch paths** — `if self.config.use_flash_attention: ... else: ...` runs only one side; a submodule on the path not taken never fires.
- **Modules not reached by this input** — auxiliary heads, MoE experts not routed to, a vision encoder on a text-only query.
- **Children of a skipped module** — after `module.skip(value)`, that module's submodules don't run.

nnsight can't know in advance whether a module fires. Verify it yourself (see the
fixes below).

### 2. A loop asked for more steps than the run made

`for step in tracer.iter[:8]` against a three-step generation, or against one that
stops early on an EOS, leaves the worker parked inside the loop body on a step
that never comes. Unwinding it takes the worker out *at the loop*, so every
statement the block has after the loop is discarded.

Whether that is an error depends on what the loop claimed:

| Loop | Outran the run | Why |
|---|---|---|
| `tracer.iter[:8]`, `tracer.iter[2]`, `tracer.iter[[0, 2, 7]]` | **raises** | The count is part of what the block says it does. Warning here would leave the names after the loop holding whatever they held before, so the block returns a stale value and says nothing about it. |
| `tracer.iter[:]`, `tracer.iter[2:]`, `tracer.all()` | **warns** | An open loop has no end of its own; outrunning the model is how it finishes. Values saved inside it are kept — but the statements after it are lost the same way, which is what the warning says. |

A body that reads locations out of order inside a loop strands the worker one
occurrence past the loop's own selection, and surfaces here too — see
[out-of-order-error.md](out-of-order-error.md) for how to tell the two apart.

## Fix

### Confirm the module actually fires — `model.scan(...)`

`scan` runs the forward with fake tensors (shapes only), hitting the same code
paths the real forward will:

```python
with model.scan("Hello"):
    print(model.transformer.h[5].output.shape)   # raises here too if it never fires
```

See [docs/usage/scan.md](../usage/scan.md).

### Read the forward source

`print(model.transformer.h[0].source)` (see [docs/usage/source.md](../usage/source.md))
or `inspect.getsource(type(module).forward)` shows what actually runs — branches,
eval-mode short-circuits, and dispatch hide behind the module tree.

### Hold the run to the step count you loop over

A bound only holds if the generation makes that many steps, so pin both ends:

```python
# WRONG — iter[:8] over a run that stops after 3 steps
with model.generate("Hi", max_new_tokens=8) as tracer:
    hs = nnsight.save([])
    for step in tracer.iter[:8]:
        hs.append(model.transformer.h[-1].output[0, -1])
    ids = tracer.result.save()
```

```python
# FIXED — min_new_tokens holds the generation to the count the loop asks for
with model.generate("Hi", max_new_tokens=3, min_new_tokens=3) as tracer:
    hs = nnsight.save([])
    for step in tracer.iter[:3]:
        hs.append(model.transformer.h[-1].output[0, -1])
    ids = tracer.result.save()
```

If you do not know the step count in advance, loop with `tracer.all()` and move
whatever follows the loop out past the `with` block — the loop's own saved values
survive:

```python
with model.generate("Hi", max_new_tokens=8) as tracer:
    hs = nnsight.save([])
    for step in tracer.all():
        hs.append(model.transformer.h[-1].output[0, -1])
print(len(hs))      # however many steps the run made
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
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — bounded vs open iteration.
- [docs/usage/source.md](../usage/source.md) — read inside-the-forward execution order.
