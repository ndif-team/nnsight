---
title: Interleaver and Hooks
one_liner: Lazy one-shot PyTorch hooks registered per-access by each Mediator, anchored by a sentinel forward hook and inserted in mediator-index order.
tags: [concept, mental-model, hooks]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/envoy-and-eproperty.md, docs/concepts/source-tracing.md]
sources: [src/nnsight/intervention/interleaver.py:481, src/nnsight/intervention/hooks.py:92, src/nnsight/intervention/hooks.py:154, src/nnsight/intervention/hooks.py:271, src/nnsight/intervention/hooks.py:356, src/nnsight/intervention/hooks.py:495]
---

# Interleaver and Hooks

## What this is for

How values get from the running model into the worker thread. nnsight's hook architecture as of `refactor/transform`:

- **Sentinel forward hook**: every wrapped module gets one no-op forward hook so PyTorch always takes the slow dispatch path.
- **Lazy one-shot hooks**: when a worker accesses `.output` / `.input`, an `eproperty` decorator registers a single forward hook that fires once, delivers the value to the mediator, and self-removes.
- **Persistent hooks**: caches and iteration trackers register permanent hooks that fire on every forward pass.
- **Ordered insertion**: when multiple mediators hook the same module, hooks fire in definition order via `add_ordered_hook` and a `mediator_idx` attribute.

This replaces the older permanent-hook approach where every module had a forward + pre-forward hook installed for the lifetime of the wrapper.

## When to use / when not to use

You don't call these directly — `eproperty` decorators (`requires_output`, `requires_input`) do it for you. Read this doc to:

- Understand why an unhooked module has near-zero overhead.
- Debug "missed provider" / `OutOfOrderError` issues.
- Implement a custom backend (vLLM, sglang-style) where you need to plumb values into the interleaver.

## Module wrapping

`Interleaver.wrap_module(module)` (`interleaver.py:481`) does two things on first wrap:

1. Replaces `module.forward` with `nnsight_forward`, a thin wrapper that:
   - Pops `__nnsight_skip__` from kwargs and returns it directly if present (this is how `Envoy.skip(...)` works).
   - Otherwise routes through the module's `__source_accessor__` if one exists (see [Source Tracing](source-tracing.md)), else through `__nnsight_forward__` (the saved original).
2. Registers a sentinel `register_forward_hook(lambda _, __, output: output)`. This exists purely so PyTorch never takes the fast-path-with-no-hooks branch in `Module.__call__` — without it, hooks added *during* a forward pass would be silently ignored.

```python
# What wrap_module produces, conceptually:
module.__nnsight_forward__ = original_forward          # saved
module.forward = nnsight_forward                        # skippable wrapper
module.register_forward_hook(lambda _, __, o: o)        # sentinel
```

## One-shot hook registration

When a worker accesses `model.transformer.h[0].output`:

1. The `eproperty` for `output` runs its decorated stub, which is wrapped by `requires_output` (`hooks.py:271`).
2. `requires_output` checks `interleaver.batcher.current_provider` — if the value for this `(path, iteration)` is already being served (e.g. nested eproperty access on the same key), hook registration is skipped.
3. Otherwise, `output_hook(mediator, module, path)` (`hooks.py:224`) registers a one-shot forward hook via `add_ordered_hook`. The hook captures the target iteration at registration time.
4. The eproperty then issues the blocking `request(...)` call.
5. When the model fires that module's forward hook chain, the one-shot hook checks `iteration_tracker[path] == target_iter`. If matched, it self-removes (`handle.remove()`), then calls `mediator.handle(...)` to deliver the value.

`input_hook` (`hooks.py:154`) is the symmetric variant for forward pre-hooks. It uses `_forward_pre_hooks_with_kwargs=True` so the hook receives both args and kwargs.

## add_ordered_hook and mediator_idx

`add_ordered_hook(module, hook, type)` (`hooks.py:92`) inserts a hook into the module's hook dict at a position determined by `hook.mediator_idx`:

- Lower `mediator_idx` fires first.
- Intervention hooks use `mediator_idx = mediator.idx` (matches the order invokes were defined).
- Cache hooks and iteration-tracker hooks use `mediator_idx = float('inf')` so they always fire **last** — this guarantees caches see post-intervention values, and iteration counters advance only after every intervention has matched against the current step.

PyTorch fires hooks in dict-insertion order. To preserve mediator order while inserting mid-flight, `add_ordered_hook` rebuilds the dict with the new hook spliced in.

## Persistent hooks

Persistent hooks are registered once and **don't** self-remove:

- **Cache hooks** (`hooks.py:356`, `hooks.py:397`): `cache_output_hook` / `cache_input_hook` fire on every forward pass and append to a `Cache` object. Only fire when the owning mediator's `batch_group` is scheduled for this pass (`batch_group[0] != -1`).
- **Iter-tracker hooks** (see [Source Tracing](source-tracing.md) and `intervention/tracing/iterator.py`): registered when `tracer.iter[...]` is entered, removed in the iter loop's `finally`. Bump `mediator.iteration_tracker` for both `.input` and `.output` paths after every forward pass.

All persistent hooks are registered through `add_ordered_hook` with `mediator_idx = float('inf')` so they fire after intervention hooks, and they're tracked on `mediator.hooks` so `Mediator.remove_hooks()` cleans them up at cancel.

## Operation hooks (source tracing)

For `.source` operation tracing, hooks live on plain Python lists on an `OperationAccessor`, not on a PyTorch module. `OperationHookHandle` (`hooks.py:61`) wraps these so they support the same `.remove()` interface as PyTorch handles.

Three operation hook types in `hooks.py`:

- `operation_input_hook` (`hooks.py:549`) — appended to `op_accessor.pre_hooks`.
- `operation_output_hook` (`hooks.py:495`) — appended to `op_accessor.post_hooks`.
- `operation_fn_hook` (`hooks.py:589`) — appended to `op_accessor.fn_hooks`. Used for recursive `.source` to substitute an injected function for the original on a single call.

These have analogous `requires_operation_input` / `requires_operation_output` decorators (`hooks.py:438`, `hooks.py:464`) used by `OperationEnvoy` eproperties.

## Mediator.hooks lifecycle

Every handle returned by these registration functions is appended to `mediator.hooks`. `Mediator.remove_hooks()` (`interleaver.py:1354`) drains the list at cancel time. The cancel path runs in two places:

1. After each intervention finishes (END event).
2. In `Interleaver.cancel()` for any leftover mediators.

`.remove()` is idempotent on every handle type used (PyTorch's `RemovableHandle` and the custom `OperationHookHandle`), so calling it twice — once via the hook's self-removal path and once via `remove_hooks` — is safe.

## Why lazy hooks

Permanent hooks pay a cost on every forward pass, even when no intervention is active. With lazy hooks:

- A module that nobody traces has only the sentinel hook (constant time, no work).
- A module that *is* traced has at most one extra hook per intervention, and it self-removes after firing.
- The sentinel keeps PyTorch on the slow dispatch path so a hook added mid-forward is still seen.

This was the headline performance win in 0.6.0 — empty traces dropped from ~1.2 ms to ~0.3 ms.

## Gotchas

- **The sentinel hook is required.** Removing it (or any other hook) such that the module has zero hooks at the start of a forward call will cause PyTorch to fast-path past hook dispatch and silently skip any hooks added later in the same call.
- **`mediator_idx` must be set on every hook.** `add_ordered_hook` defaults to `float('-inf')` if missing, but cache / iter-tracker hooks rely on `inf` for correctness — set it explicitly.
- **One-shot hooks check the iteration tracker.** During multi-step generation, a one-shot hook registered at step 0 will refuse to fire on step 1 unless it was constructed with `iteration=1`. The `iteration_tracker[path]` is the source of truth.
- **`current_provider` skips registration.** When two eproperties share a key (e.g. `Envoy.input` and `Envoy.inputs`), accessing one back-to-back inside the same hook chain re-uses the same value without a second hook registration. See `requires_output` (`hooks.py:271`) for the check.

## Related

- [Threading and Mediators](threading-and-mediators.md) — what fires the hooks (the mediator event loop).
- [Envoy and eproperty](envoy-and-eproperty.md) — the descriptor that ties hook registration to user-facing properties.
- [Source Tracing](source-tracing.md) — operation hooks for `.source`.
- Source: `src/nnsight/intervention/interleaver.py` (`wrap_module`, `Mediator.remove_hooks`), `src/nnsight/intervention/hooks.py`.
