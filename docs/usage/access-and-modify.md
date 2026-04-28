---
title: Access and Modify Module Values
one_liner: Use `.output`, `.input`, `.inputs` to read activations; in-place slice or assign to modify.
tags: [usage, intervention, eproperty]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/skip.md, docs/usage/source.md]
sources: [src/nnsight/intervention/envoy.py:168, src/nnsight/intervention/interleaver.py:60, src/nnsight/intervention/hooks.py]
---

# Access and Modify Module Values

## What this is for

Every wrapped module exposes three special properties that read or replace the module's runtime values during the forward pass:

| Property | Returns | Source |
|---|---|---|
| `module.output` | The module's forward-pass return value | `envoy.py:168` |
| `module.input` | The first positional input | `envoy.py:191` |
| `module.inputs` | `(args, kwargs)` tuple of all inputs | `envoy.py:178` |

These are `eproperty` descriptors (`src/nnsight/intervention/interleaver.py:60`). Reading one issues a blocking request to the worker thread's mediator; the request unblocks when the corresponding one-shot PyTorch hook fires (`hooks.py: requires_output`, `requires_input`).

## When to use / when not to use

- Use any of these inside a `with model.trace(...)` / `model.generate(...)` body to read activations.
- Use slice assignment `[:] = value` for in-place modification.
- Use direct assignment `module.output = value` for replacement.
- Outside a tracing context, accessing `.output` raises `ValueError: Cannot access ... outside of interleaving.`

## Canonical pattern

```python
with model.trace("Hello"):
    # Read (in transformers 5+, transformer blocks return a tensor)
    hidden = model.transformer.h[-1].output.save()

    # In-place modify (zero the residual)
    model.transformer.h[0].output[:] = 0

    # Replacement
    model.transformer.h[0].output = torch.zeros_like(hidden)
```

## In-place vs replacement

```python
# IN-PLACE: mutates the tensor underlying the model's value.
# All later references through the same descriptor see the mutation.
model.transformer.h[0].output[:] = 0

# REPLACEMENT: triggers the eproperty's __set__, which calls
# mediator.swap(...) — the model's downstream computation gets the new value.
model.transformer.h[0].output = my_new_tensor
```

Replacement goes through `eproperty.__set__` (`interleaver.py:306`) which calls `self._postprocess` (if any), builds the requester string, registers the hook, and emits a SWAP event.

## Tuple outputs

Some modules return a tuple. The most common in HuggingFace LLMs is the **attention module**, which returns `(attn_out, attn_weights)`. In transformers <5, transformer blocks themselves also returned tuples; in transformers 5+ they return a plain tensor.

```python
with model.trace("Hello"):
    full = model.transformer.h[0].attn.output    # tuple (attn_out, weights)
    attn_out = full[0]                            # tensor

    # In-place on the first element
    model.transformer.h[0].attn.output[0][:] = 0

    # Replace the entire tuple (preserve other elements)
    model.transformer.h[0].attn.output = (
        torch.zeros_like(attn_out),
    ) + model.transformer.h[0].attn.output[1:]
```

## `.input` vs `.inputs`

`.inputs` returns the raw `(args, kwargs)` tuple as captured by the pre-forward hook. `.input` is a convenience that returns the first positional argument (or first kwarg value if no positional). It is built by chaining `.inputs` through a `preprocess`/`postprocess` pair:

```python
@input.preprocess
def input(self, value):
    return [*value[0], *value[1].values()][0]

@input.postprocess
def input(self, value):
    inputs = self.inputs
    return (value, *inputs[0][1:]), inputs[1]
```

So `module.input = new_tensor` correctly repacks into `(args, kwargs)` for the model.

## Cloning before modification

In-place modifications happen on the live tensor — **after** the modification, reading `.output` again returns the modified value:

```python
with model.trace("Hello"):
    before = model.transformer.h[0].output.clone().save()  # capture pre-mod
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()           # post-mod
# without the clone, before == after
```

The `.clone()` is a real `torch.Tensor.clone()` — the worker thread receives the actual tensor.

## Forward-pass-order rule

Within a single invoke, you **must** request modules in the order they execute. The worker thread blocks on each request and the model's forward pass produces values in execution order. Asking for layer 5 then layer 1 in the same invoke deadlocks → `OutOfOrderError` is raised once the model finishes (`Mediator.handle_value_event` in `interleaver.py:1013`).

To access modules out of order, use additional invokes — see `docs/usage/invoke-and-batching.md`.

## Calling modules directly inside a trace

```python
with model.trace("Hello"):
    hs = model.transformer.h[5].output
    # Calling the envoy directly uses .forward() (no hook),
    # so this re-runs ln_f + lm_head WITHOUT triggering interleaving.
    logits = model.lm_head(model.transformer.ln_f(hs)).save()
```

`Envoy.__call__` checks if the interleaver has a current mediator; if so it calls `module.forward(...)` directly instead of `module(...)`. Pass `hook=True` to opt back into the hook path (e.g. when you want to observe an injected SAE module's output) — see `Envoy.__call__` in `envoy.py:239`.

## Module skipping

`module.skip(value)` bypasses the module's computation entirely. See `docs/usage/skip.md`.

## Source-level access

For sub-module operations (e.g. `attention_interface_0`, `self_c_proj_0`), use `module.source.<op>.output`. See `docs/usage/source.md`.

## Gotchas

- Within an invoke, modules **must** be accessed in forward-pass order. See `docs/gotchas/out-of-order.md`.
- For tuple-returning modules (e.g. attention), `module.output[0] = x` is a `__setitem__` on the underlying tuple and raises `TypeError`. Use `module.output[0][:] = x` for in-place on the first tuple element, or build a new tuple `(x,) + module.output[1:]` and assign to `module.output`.
- Reading `.output` returns the actual runtime tensor — `print`, `.shape`, `.mean()`, etc. all work. There is no proxy layer to unwrap.
- Outside interleaving, accessing `.output` raises `ValueError: Cannot access ...`. Use `model.scan(...)` if you only need shapes without execution.
- If a module's class defines an attribute named `input` or `output`, nnsight remounts its proxy to `.nns_input` / `.nns_output` (with a warning). See `Envoy._handle_overloaded_mount` in `envoy.py:733`.

## Related

- `docs/usage/trace.md`
- `docs/usage/save.md`
- `docs/usage/skip.md`
- `docs/usage/source.md`
- `docs/usage/scan.md`
