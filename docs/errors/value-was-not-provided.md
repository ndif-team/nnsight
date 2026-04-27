---
title: Execution Complete But Value Was Not Provided
one_liner: "Raised by check_dangling_mediators when the model finished but a mediator is still waiting for a module's value — almost always means out-of-order access OR a module that didn't actually fire."
tags: [error, execution-order, dangling-mediator]
related: [docs/errors/missed-provider-error.md, docs/errors/out-of-order-error.md, docs/usage/iter-all-next.md, docs/usage/scan.md, docs/concepts/source-tracing.md, docs/concepts/threading-and-mediators.md, docs/gotchas/order-and-deadlocks.md]
sources: [src/nnsight/intervention/interleaver.py:652, src/nnsight/intervention/interleaver.py:667, src/nnsight/intervention/interleaver.py:677]
---

# Execution Complete But Value Was Not Provided

## Symptom

Raised as a `Mediator.MissedProviderError` (which is an `Exception`) at the end of the forward pass:

```
Execution complete but `model.transformer.h.5.output.i0` was not provided. Did you call an Envoy out of order? Investigate why this module was not called.
```

Variant emitted as a `UserWarning` instead of an exception when the unsatisfied request happened beyond the first iteration of a generator loop (e.g., `.iter[:]` after generation finished):

```
UserWarning: Execution complete but `<requester>` was not provided. If this was in an Iterator at iteration <N> this iteration did not happen. If you were using `.iter[:]`, this is likely not an error.
```

The `<requester>` is a dotted path of the form `<envoy.path>.<key>.i<iteration>` — for example `model.transformer.h.5.output.i0` means "layer 5's output on iteration 0".

## Cause — read this carefully

There are exactly two reasons a value is "not provided" by the time the model finishes. Most agents hit (1); (2) is sneakier and requires reading the model's actual source.

### Cause 1 — out-of-order access (the common one)

> **You MUST access module eproperties (`.output`, `.input`, `.inputs`) in the order they actually fire during the forward pass.**

This is not a soft guideline. Each invoke runs its body in a single worker thread that synchronizes step-by-step with the model. When the worker requests `module.output`, it blocks until the model fires that hook. If you ask for layer 1's output **after** layer 5's output, the layer-1 hook has already fired and been consumed by the time your request arrives — there is no way to retroactively deliver it.

The mediator detects two flavors:

- **Eager** — `OutOfOrderError`: the mediator already saw a provider with that requester string fire, so it raises immediately when the request arrives. (`src/nnsight/intervention/interleaver.py:1049`.)
- **Late** — `MissedProviderError` from `check_dangling_mediators`: the model finished, the worker is still waiting, and now the model has nothing left to provide. (`src/nnsight/intervention/interleaver.py:652`.)

Both have the same root cause: the requested provider passed during forward and the request arrived too late.

The exception is then re-raised in the user's call stack via the worker-thread join in `Mediator.handle_exception_event` (`src/nnsight/intervention/interleaver.py:1119`).

### Cause 2 — the module never fired (the sneaky one)

The module path you asked for **exists in `print(model)`**, but **was not called** during this forward pass. The hook that would deliver its value is registered, but never fires, because the model's forward code never reaches the call. The dangling-mediator check (`:652`) raises at the end of the pass.

This is harder to spot because there's no red flag in your intervention code — the path is real, the module is real, you just got it wrong about whether it runs. Common situations where modules don't fire:

- **Dropout layers in eval mode.** `nn.Dropout(p)` is bypassed when the model is in `model.eval()`. If you wrote `model.transformer.h[0].dropout.output.save()`, the request is queued, but `dropout.forward` is never reached because eval mode short-circuits it.
- **Branch paths in conditional forward code.** A module's `forward` may dispatch between two implementations (`if self.config.use_flash_attention: ... else: ...`). Only one path's submodules fire per call. If you target a submodule on the path not taken, you'll hit this.
- **Modules that exist on the model but aren't reached by the current input.** Adapters, auxiliary heads, mixture-of-experts experts that aren't routed to for this input, vision-language models where the vision encoder isn't called for a text-only query.
- **Skipped modules.** You called `module.skip(value)` and then asked for `.output` on a child of the skipped module — that child never fired.
- **Iter steps that never happened.** `tracer.iter[N]` for an `N` past the last actually-generated step. The warning variant (`:677`) covers this case.

There is no way for nnsight to know in advance whether your target module will fire. You have to verify it yourself — see the fix section.

### Reading the requester string

The requester string is your single biggest diagnostic clue. It is structured as `<envoy_path>.<key>.i<iteration>`:

- `model.transformer.h.5.output.i0` — layer 5's output, on the first call (e.g. forward pass for the first generation step).
- `model.lm_head.input.i2` — `lm_head`'s first positional input, on the third call.
- `model.transformer.h.0.attn.attention_interface_0.output.i0` — operation-level: the output of `attention_interface_0` (a `wrap_operation` call site) inside layer 0's attention, on its first call.

The `.iN` suffix is the per-provider call counter. For a single forward pass it is always `.i0`; for generation, `.i1`, `.i2`, … for subsequent steps.

## Common triggers

- Reading modules in reverse order inside a single invoke (`out5` before `out1`).
- Requesting `.output` on a module path that exists in `print(model)` but is **not called** on this input (model dispatches between two branches; only one branch fires).
- Code placed after an unbounded `for step in tracer.iter[:]:` loop — the iterator never yields a final batch, so anything past the loop is left waiting.
- Using `tracer.iter[N]` for an `N` larger than the number of steps actually generated (early EOS, `max_new_tokens` smaller than `N+1`).
- Skipping a module via `module.skip(value)` and then trying to read `.output` on a child of the skipped module — child modules don't fire.
- Constructing a swap (`module.output = ...`) on a module that the model never calls.
- Hitting a dropout / batchnorm / other eval-mode-disabled layer.
- Targeting a module on a branch the model didn't take this call.

## Fix

### When the cause is out-of-order access

Lay your intervention code in the same order modules execute. Use `print(model)` as a guide:

```python
# WRONG — layer 5 runs before layer 1; the request for h[1].output arrives too late
with model.trace("Hello"):
    out5 = model.transformer.h[5].output.save()
    out1 = model.transformer.h[1].output.save()
```

```python
# FIXED — top-to-bottom matches forward order
with model.trace("Hello"):
    out1 = model.transformer.h[1].output.save()
    out5 = model.transformer.h[5].output.save()
```

To genuinely access modules out of forward order, run a second forward pass via an empty invoke. Each invoke is its own worker thread:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out5 = model.transformer.h[5].output.save()
    with tracer.invoke():           # empty invoke = new pass on the same batch
        out1 = model.transformer.h[1].output.save()
```

### When the cause is unbounded iter

Use a bounded slice or split with empty invokes:

```python
# WRONG — `final` never gets fulfilled; check_dangling_mediators raises
with model.generate("Hi", max_new_tokens=3) as tracer:
    for step in tracer.iter[:]:
        hs = model.transformer.h[-1].output.save()
    final = model.lm_head.output.save()
```

```python
# FIXED — split the post-iter request into an empty invoke or use a bounded slice
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("Hi"):
        for step in tracer.iter[:]:
            hs = model.transformer.h[-1].output.save()
    with tracer.invoke():
        final = model.lm_head.output.save()
```

### When the cause is "the module didn't fire"

Verify the module actually executes for your input. Two ways:

**1) `model.scan(input)`** — runs forward with fake tensors, hits exactly the same code paths the real forward will, but cheaply. If your target path raises here, it doesn't fire on real input either:

```python
import nnsight

with model.scan("Hi"):
    nnsight.save(model.transformer.h[5].output.shape)
    # Add scan reads of every path you intend to access in the real trace.
```

See [docs/usage/scan.md](../usage/scan.md) for full scan semantics.

**2) Read the model's actual forward source.** Use `print(module.source)` (see [docs/concepts/source-tracing.md](../concepts/source-tracing.md)) or open the file:

```python
import inspect
print(inspect.getsource(type(model.transformer.h[0]).forward))
```

This is non-negotiable when the model has conditional branches, mixture-of-experts routing, or training-mode-only modules. **You cannot guess what fires from the module tree alone** — the tree shows what *can* be called, the source shows what *is* called.

For dropout-in-eval-mode specifically:

```python
# WRONG — dropout doesn't fire in eval mode
model.eval()
with model.trace("Hi"):
    drop = model.transformer.h[0].dropout.output.save()  # MissedProviderError
```

```python
# FIXED — target the surrounding module that actually fires, or put model in train mode if you really need it
with model.trace("Hi"):
    block_out = model.transformer.h[0].output.save()  # the block runs
```

### When the cause is a skipped module's child

```python
# WRONG — child of skipped module never fires
with model.trace("Hi"):
    h0_out = model.transformer.h[0].output  # capture
    model.transformer.h[1].skip(h0_out)
    inner = model.transformer.h[1].mlp.output.save()  # MissedProviderError — h[1] was skipped, mlp never fires
```

```python
# FIXED — read the inner only on layers that actually run
with model.trace("Hi"):
    h0_out = model.transformer.h[0].output
    model.transformer.h[1].skip(h0_out)
    inner = model.transformer.h[2].mlp.output.save()  # h[2] runs
```

## Mitigation / how to avoid

- **Default order:** access modules top-to-bottom in the same order as `print(model)`.
- **Always reach unbounded iteration via a separate empty invoke** for any "after-generation" code. Or prefer bounded `tracer.iter[:N]` when the step count is known.
- **Use `model.scan(input)` to confirm a module fires** before relying on it inside a real trace. This is the single most reliable check.
- **Read the forward source** of any module in the path before targeting one of its submodules — branch paths, eval-mode short-circuits, and conditional dispatch all hide behind `print(model)`.
- The warning variant (iter-after-generation) is benign and silenceable with `warnings.filterwarnings("ignore")` on `UserWarning` if it's noisy in your pipeline.

## Related

- [missed-provider-error.md](missed-provider-error.md) — base class
- [out-of-order-error.md](out-of-order-error.md) — subclass for already-seen providers
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md)
- [docs/usage/scan.md](../usage/scan.md) — verify modules fire
- [docs/concepts/source-tracing.md](../concepts/source-tracing.md) — read inside-the-forward operation order
- [docs/gotchas/order-and-deadlocks.md](../gotchas/order-and-deadlocks.md)
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md)
