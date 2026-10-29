---
title: Interleaver and Controller
one_liner: Every wrapped module's forward is a controller that passes straight through when no trace is running and, inside one, hands its input, skip gate and output to the one shared Interleaver; each visit to a location is a numbered occurrence.
tags: [concept, mental-model, controller]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/envoy.md, docs/concepts/source-tracing.md, docs/developing/controller.md, docs/developing/interleaver-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/source.py]
---

# Interleaver and Controller

## The model side

Workers park on locations ([threading-and-mediators.md](threading-and-mediators.md));
this page is how the running model serves them.

- **Every module's forward is a controller**, installed once when the module is
  wrapped. No PyTorch hooks are registered. When no trace is running (or the trace
  has no workers) the controller just calls the real forward, so an instrumented
  model runs at normal speed outside a trace.
- **Inside a trace the controller is a gate with three handoffs**: it offers the
  module's input at `{path}.input`, asks whether anyone wants to skip it at
  `{path}.skip`, runs the real forward (unless skipped), and offers the output at
  `{path}.output`. Whatever comes back from each handoff — possibly edited or
  replaced by a worker — is what the forward continues with.
- **One `Interleaver` per envoy tree, one call for everything.** Each handoff is
  `Interleaver.handle(location, value)`: offer the value to every worker in turn
  (each sees the previous one's edits), then to any active `tracer.cache()` — so a
  cache records values *after* interventions — and return the result to the forward.
  Source operations, `.skip` and `tracer.result` use the same call.

Because the input and skip handoffs come before the forward runs, a block can read a
module's `.input` and then `.skip()` it with that value. The controller's code and its
tensor-parallel handling are in [developing/controller.md](../developing/controller.md).

## Occurrences

A location can be reached many times in one run — every generation step revisits
every module, and a source operation inside a loop fires once per iteration. Each
visit is a numbered **occurrence**, and a request names the one it wants:
`model.transformer.h.5.output.i0` is the first visit.

- With no `tracer.iter`, every request is `.i0` — the first visit.
- Inside `tracer.iter[k]`, requests target step `k`; earlier visits pass by and the
  request binds on the k-th ([usage/iteration.md](../usage/iteration.md)).
- A module advances its count once per forward call; a source operation once per
  fire ([source-tracing.md](source-tracing.md)).

When the run ends, a worker still waiting on a location raises
[`OutOfOrderError`](../errors/out-of-order-error.md) — it asked for an occurrence the
model already passed or never reached. Inside a `tracer.iter` loop the same situation
is a warning and a cut-short loop ([errors/module-never-fired.md](../errors/module-never-fired.md)).

## Working with it

- **Don't replace a module's `forward` by hand**, and don't let another library do
  it: the controller would be gone. Reassign modules through the envoy so it is
  re-installed ([out-of-order-error.md](../errors/out-of-order-error.md#another-cause-something-replaced-the-modules-forward)).
- **A custom driver feeds values the same way** — a runtime that produces values no
  module returns (logits, samples) calls `Interleaver.handle` with its own location
  ([developing/adding-a-new-runtime.md](../developing/adding-a-new-runtime.md)).

## Related

- [Threading and Mediators](threading-and-mediators.md) — the worker side.
- [Envoy](envoy.md) — the properties that name locations.
- [developing/interleaver-internals.md](../developing/interleaver-internals.md) — `handle`, caches and teardown in code.
