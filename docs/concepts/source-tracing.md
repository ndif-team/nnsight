---
title: Source Tracing
one_liner: .source turns every call and assignment inside a module's forward into a location with its own .input / .output / .skip — the same mechanism and rules as modules, one level finer.
tags: [concept, mental-model, source-tracing]
related: [docs/concepts/envoy.md, docs/concepts/interleaver-and-controller.md, docs/usage/source.md, docs/developing/source-internals.md]
sources: [src/nnsight/intervention/source.py]
---

# Source Tracing

## The idea

A module exposes two locations, its input and its output. Everything computed in
between — an activation function, an attention call, a `torch.matmul`, a running
value in a loop — is invisible, because it isn't a submodule.

`.source` makes it visible. nnsight rewrites the module's `forward` so that every
**call** and every **assignment** becomes an operation with the same handles an
Envoy has: `.input` / `.inputs`, `.output`, `.skip(value)`. At run time each
operation is offered to the interleaver exactly like a module — input, skip gate,
output — so everything you know about modules carries over:

- **Forward order** applies to operations too, interleaved with module locations.
- **Occurrences** count per *fire*: an operation in a loop (an MoE expert loop) fires
  several times per forward, and each fire is its own occurrence; one outside a loop
  counts once per forward, like its module
  ([interleaver-and-controller.md](interleaver-and-controller.md#occurrences)).
- **Skip** returns your value as the operation's output and the call never runs.

```python
print(model.transformer.h[0].mlp.source)           # the forward, every op labelled

with model.trace("Hello world"):
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()   # read
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0      # edit
```

Operation names are the full dotted callee plus an occurrence index
(`self.c_proj(x)` → `self_c_proj_0`), or the assigned name for an assignment
(`hidden_states_1`). They come from the library's source, so they move when it
changes — `print(module.source)` is the source of truth. Naming, discovery and
recipes: [usage/source.md](../usage/source.md).

## Working with it

- **`.source` on an operation drills into the function it calls** — e.g. into the
  attention implementation. That needs a trace, because the callee is whatever
  value flows through the call at run time, so which function you land in depends
  on config (`attn_implementation`). Print the drilled source rather than assuming.
- **A call to a submodule is not drillable** — address the submodule directly.
  Neither is an assignment, or a builtin / C function with no Python source; all
  raise `SourceNotAvailable`.
- **The listing is static; the forward is not.** It names operations on branches
  this model never takes; asking for one ends in `OutOfOrderError`
  ([usage/source.md](../usage/source.md#operations-that-never-run)).

How the rewrite, decorators and caching work:
[developing/source-internals.md](../developing/source-internals.md).

## Related

- [Envoy](envoy.md) — the module-level handles operations mirror.
- [usage/source.md](../usage/source.md) — using `.source` in practice.
