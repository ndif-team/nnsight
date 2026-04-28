---
title: Lazy Hook System
one_liner: One-shot, mediator-ordered PyTorch hooks installed on demand and self-removing after firing.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/interleaver-internals.md, docs/developing/eproperty-deep-dive.md, docs/developing/source-accessor-internals.md]
sources: [src/nnsight/intervention/hooks.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/tracing/iterator.py]
---

# Lazy Hook System

## What this covers

The single biggest architectural change in `refactor/transform`: PyTorch hooks for input and output access are no longer permanently registered on every wrapped module. Instead, each mediator registers its own one-shot hook *only when the worker thread asks for that value*, and the hook self-removes after firing.

This doc covers:

- The motivation (why permanent hooks were a problem).
- `add_ordered_hook` — the mediator-ordered insert into PyTorch's hook dicts.
- `input_hook` / `output_hook` — the one-shot factories.
- `requires_output` / `requires_input` — the eproperty pre-setup decorators.
- The sentinel hook in `Interleaver.wrap_module` and why it is required.
- Persistent hooks (cache, iter-tracker) and their `mediator_idx = inf` ordering.
- `OperationHookHandle` — the `RemovableHandle`-shaped wrapper for source operation hooks.
- `Mediator.hooks` and unified cleanup.

## Architecture

### Why "lazy"

In previous versions of nnsight, every wrapped module had two permanent hooks (a `forward_pre_hook` and a `forward_hook`) installed at wrap time. Each hook iterated over every active mediator, checked whether that mediator was waiting for this module's input/output, and dispatched accordingly.

That had three costs:

1. **Per-call overhead on every module, even when nobody is observing it.** Every forward pass paid for hook dispatch + a Python-level loop over mediators.
2. **No way to fast-path zero-mediator modules.** The hooks always fired.
3. **Tight coupling.** Anything new that wanted to hook a module had to extend the central dispatch loop.

The lazy system inverts this. Permanent hooks are gone. Each `requires_*` decorator on an `eproperty` registers a hook only at the moment the worker accesses the property, with a target iteration baked into the closure. After it fires once, it removes itself.

### The hook lifecycle

```
worker thread accesses model.layer.output
    |
    v
eproperty.__get__ -> requires_output(stub) wrapper
    |
    v
output_hook(mediator, module, path):
    iteration = mediator.iteration or iteration_tracker[path]
    handle = add_ordered_hook(module, hook, "output")
    mediator.hooks.append(handle)
    return handle
    |
    v
descriptor calls mediator.request(requester) -> worker BLOCKS
    |
    | (model is now running on main thread)
    v
PyTorch fires hooks on this module's output:
   sentinel hook (returns output unchanged)
   ... possibly other mediators' one-shot hooks (sorted by mediator_idx) ...
   our one-shot hook:
     if iteration_tracker[path] != iteration: return output  # not our turn
     handle.remove()
     output = mediator.handle(f"{path}.i{iteration}", output)
     return output
   ... possibly persistent cache/iter hooks (mediator_idx = inf) ...
    |
    v
worker resumes with output, possibly mutated by user code
```

The handle is appended to `mediator.hooks` *and* held by the closure. After the hook self-removes, the closure clears `handle._target` and the entry in `mediator.hooks` becomes a no-op handle (its `.remove()` is idempotent). `Mediator.remove_hooks` (`interleaver.py:1354`) drains the entire list at session cleanup, so any hooks that never fired get cleaned up too.

### add_ordered_hook

`add_ordered_hook(module, hook, type)` (`src/nnsight/intervention/hooks.py:92`) is the only place nnsight pokes at PyTorch's internal hook dicts directly. It exists because:

- PyTorch fires hooks in **dict insertion order**.
- Nnsight needs hooks to fire in **mediator definition order**, regardless of which mediator registered first.
- For invokes 1 and 2 both hooking layer X, invoke 1's hook must run before invoke 2's hook so the worker threads see values in the right batch slice.

The function:

1. Constructs a `RemovableHandle` against `module._forward_hooks` (or `_forward_pre_hooks`) and registers the new entry in `_forward_hooks_with_kwargs` so PyTorch passes `(module, args, kwargs, output)` style.
2. If the dict is empty, just inserts and returns.
3. Otherwise, reads `hook.mediator_idx` (defaults to `-inf` if unset), iterates the current entries (using `getattr(v, "mediator_idx", -inf)` for the existing ones), and rebuilds the dict with the new hook inserted at the correct sorted position.
4. Clears the dict and re-fills it in the new order. PyTorch's iteration of `_forward_hooks` uses normal `dict.values()`, so the rebuilt order is what fires.

This is a deliberate violation of PyTorch's "use `register_forward_hook` only" contract. It is necessary because PyTorch's public API has no concept of hook priority. The behavior is verified to match dict-iteration order in the test suite.

### Sentinel hook (the reason wrap_module installs a hook)

`Interleaver.wrap_module` registers exactly one persistent hook:

```python
module.register_forward_hook(lambda _, __, output: output)
```

This is required because of a PyTorch fast path: `Module.__call__` checks whether the module has any hooks at the time it is called. If `_forward_hooks` is empty, it bypasses the entire hook-dispatch pipeline. A one-shot hook that the worker thread registers *during* the forward pass would, in that case, never fire.

The sentinel keeps the dict non-empty so PyTorch always takes the dispatch path. Its `mediator_idx` is implicitly `-inf` (no attribute), so it always fires first when other hooks are inserted via `add_ordered_hook`.

There is no equivalent sentinel for `_forward_pre_hooks`. Instead, `requires_input` will register a pre-hook the first time it is needed, and `add_ordered_hook` initializes the dict from empty if necessary. This works because pre-hooks fire before the forward call returns its output, and at the time `requires_input` runs, the worker is already in a known-active state — there is no fast-path concern.

### input_hook and output_hook

`input_hook(mediator, module, path)` (`src/nnsight/intervention/hooks.py:154`) and `output_hook(mediator, module, path)` (`src/nnsight/intervention/hooks.py:224`) are the per-mediator factories. They:

1. **Capture the target iteration at registration time** — either `mediator.iteration` (set by `IteratorTracer` for explicit iter loops) or `mediator.iteration_tracker[path]` (the "next" forward pass for this path).
2. **Define a closure hook** that:
   - Returns immediately if `iteration_tracker[path]` has not advanced to the target yet (the worker is waiting for a later step).
   - Clears `mediator.iteration = None` if `iteration != 0` (so subsequent requests fall back to tracker mode).
   - Self-removes via `handle.remove()`.
   - Calls `mediator.handle(f"{path}.i{iteration}", value)` to deliver the value, possibly receiving a mutated value back.
   - Returns the (possibly mutated) value to PyTorch.
3. **Sets `hook.mediator_idx = mediator.idx`** for ordering.
4. **Calls `add_ordered_hook(module, hook, type)`** and tracks the handle on `mediator.hooks`.

The hook does **not** maintain `iteration_tracker[path]` itself. That is the job of the persistent iter hook registered by `IteratorTracer.register_iter_hooks` (see below). The intervention hook only compares.

### requires_output and requires_input — the eproperty glue

`requires_output(fn)` (`src/nnsight/intervention/hooks.py:271`) is a decorator applied to an `eproperty` stub. It does pre-setup work each time the stub is invoked from `eproperty.__get__`:

1. Read `mediator = self.interleaver.current` and resolve the target iteration the same way `output_hook` will.
2. Build the requester `f"{self.path}.output.i{iteration}"`.
3. **Skip hook registration if `batcher.current_provider == requester`.** This is the case where the mediator is already inside a hook firing — for example, if `Envoy.input` is accessed and then `Envoy.inputs` is accessed back-to-back inside the same hook step, both share the requester `"<path>.input.i<iter>"` and only one hook is needed.
4. Otherwise, call `output_hook(mediator, self._module, f"{self.path}.output")`.
5. Call the wrapped stub `fn(self, *args, **kwargs)` (no-op body).

`requires_input` is symmetric for pre-hooks.

The point of the `current_provider` check is performance: when both `Envoy.input` and `Envoy.inputs` are accessed on the same module in the same step, we install one hook, not two. The single hook will deliver `(args, kwargs)` once, and both eproperty descriptors will see it via the worker reading the value from `mediator.request()`.

### Persistent cache hooks

`cache_output_hook` (`src/nnsight/intervention/hooks.py:356`) and `cache_input_hook` (`hooks.py:397`) register persistent hooks that record values into a `Cache` object on every forward pass. Key differences from one-shot hooks:

- Not self-removing. They live for the duration of the trace and are cleaned up when the interleaver exits via `Mediator.remove_hooks`.
- `mediator_idx = float('inf')`. They fire **after** all intervention hooks, ensuring the cache captures post-intervention values.
- They read `mediator.batch_group` *live* on each fire. This matters for vLLM, where decode-step batch positions change between iterations; the cache hook must see the current batch_group, not a snapshot.
- They skip when `batch_group is None or batch_group[0] == -1` — meaning "this mediator's request is not scheduled in this forward pass" (a vLLM scheduler decision).

The 13.2x cache speedup quoted in commit logs comes from making cache hooks lazy at *registration* time (only when the user calls `tracer.cache(...)`) rather than running on every module unconditionally. Once registered, they fire on every forward pass — they are not lazy at firing time.

### Persistent iter-tracker hooks

`register_iter_hooks(mediator, model)` (`src/nnsight/intervention/tracing/iterator.py:95`) registers persistent output hooks on every wrapped module when the user enters a `for step in tracer.iter[...]:` loop. Each hook does exactly:

```python
def hook(module, _, output, _path=path):
    mediator.iteration_tracker[f"{_path}.input"] += 1
    mediator.iteration_tracker[f"{_path}.output"] += 1
    accessor = getattr(module, "__source_accessor__", None)
    if accessor is not None:
        bump_source_paths(mediator, accessor)
```

These hooks are the **single source of truth** for "which generation step is this provider on?" Every other piece of the system reads `mediator.iteration_tracker` but only the iter hooks write to it (with one exception: `Interleaver.handle(..., iterate=True)` bumps the tracker for provider-pushed values).

Why `mediator_idx = inf`:

- One-shot intervention hooks check `iteration_tracker[path] == iteration` *before* the iter hook fires. We want them comparing against the "current" step's value, not the bumped one.
- Cache hooks also fire before the iter hook and capture the current step's data.

Both `.input` and `.output` paths are bumped from a single output hook because every forward pass runs the input pre-hook chain and the output post-hook chain in lockstep — the two counters stay synchronized.

`bump_source_paths` (`tracing/iterator.py:75`) walks the module's `SourceAccessor` (if any) and bumps every operation-level path, recursing through nested `OperationAccessor._source_accessor` for recursive `.source` calls. This keeps op-level iteration counters in sync with the parent module.

WrapperModules (`generator`, `streamer`, `samples`, `logits`) are skipped — they don't go through PyTorch's forward dispatch on every step. Their values flow through `eproperty.provide`, which bumps the tracker via `Interleaver.handle(..., iterate=True)`.

### OperationHookHandle

`OperationHookHandle` (`src/nnsight/intervention/hooks.py:61`) is a `RemovableHandle`-shaped wrapper for hooks that live on plain Python lists (the `pre_hooks`, `post_hooks`, `fn_hooks` on `OperationAccessor`). It exists for one reason: to be uniformly storable in `mediator.hooks` alongside PyTorch's `RemovableHandle` so that `Mediator.remove_hooks` can drain everything with a single loop.

```python
class OperationHookHandle:
    def __init__(self, target: List[Callable], hook: Callable):
        self._target = target
        self._hook = hook

    def remove(self) -> None:
        if self._hook is None:
            return
        try:
            self._target.remove(self._hook)
        except ValueError:
            pass
        self._hook = None
        self._target = None
```

`.remove()` is idempotent. The hook closure clears its own list entry on first fire (via `handle.remove()`), and `Mediator.remove_hooks` calling `.remove()` again is a no-op. This is critical because both paths can run in either order: a hook may fire and self-remove before session cleanup, or session cleanup may run before some hooks have fired.

### Mediator.hooks — unified cleanup

Every dynamic hook the interleaver creates appends its handle to `mediator.hooks`:

- `output_hook` / `input_hook` (one-shot module hooks).
- `operation_output_hook` / `operation_input_hook` / `operation_fn_hook` (one-shot operation hooks).
- `cache_output_hook` / `cache_input_hook` (persistent cache hooks).
- `register_iter_hooks` (persistent iter-tracker hooks).
- `wrap_grad`'s tensor backward hook (`tracing/backwards.py:43`).

`Mediator.remove_hooks` (`interleaver.py:1354`) iterates and calls `.remove()` on each. Because every handle's `.remove()` is idempotent, this is the single safe cleanup path even if some hooks have already self-removed. `Interleaver.cancel` calls `mediator.remove_hooks()` for every mediator after canceling its worker thread.

### Replacing SkipException with __nnsight_skip__

The old skip mechanism raised `SkipException(value)` from a permanent input hook; the exception unwound through PyTorch's call stack. With one-shot hooks, that approach doesn't work — the exception would unwind through hooks the worker hasn't seen yet.

The new mechanism (`hooks.py` and `interleaver.py:1154`):

- The worker calls `mediator.skip(requester, value)`.
- `handle_skip_event` sets `kwargs["__nnsight_skip__"] = value` directly on `batcher.current_value` (which is the `(args, kwargs)` tuple held by the input hook).
- The hook returns; PyTorch calls `nnsight_forward(*args, **kwargs)`; `nnsight_forward` checks for the key and returns the value, skipping the actual forward.

This works because the input hook runs before the forward, and `__nnsight_skip__` is poked into the kwargs dict that PyTorch will then pass through.

## Key files / classes

- `src/nnsight/intervention/hooks.py:61` — `OperationHookHandle`. Idempotent `RemovableHandle` clone for list-based hooks.
- `src/nnsight/intervention/hooks.py:92` — `add_ordered_hook`. Mediator-ordered insertion into PyTorch hook dicts.
- `src/nnsight/intervention/hooks.py:154` — `input_hook`. One-shot forward pre-hook factory.
- `src/nnsight/intervention/hooks.py:224` — `output_hook`. One-shot forward hook factory.
- `src/nnsight/intervention/hooks.py:271` — `requires_output`. Eproperty pre-setup decorator.
- `src/nnsight/intervention/hooks.py:314` — `requires_input`. Eproperty pre-setup decorator.
- `src/nnsight/intervention/hooks.py:356` — `cache_output_hook`. Persistent (mediator_idx=inf) cache hook.
- `src/nnsight/intervention/hooks.py:397` — `cache_input_hook`. Persistent cache hook for inputs.
- `src/nnsight/intervention/hooks.py:438` — `requires_operation_output`. Operation-level analogue of `requires_output`.
- `src/nnsight/intervention/hooks.py:464` — `requires_operation_input`. Operation-level analogue.
- `src/nnsight/intervention/hooks.py:495` — `operation_output_hook`. One-shot list-based hook on `OperationAccessor.post_hooks`.
- `src/nnsight/intervention/hooks.py:549` — `operation_input_hook`. One-shot hook on `pre_hooks`.
- `src/nnsight/intervention/hooks.py:589` — `operation_fn_hook`. One-shot hook for recursive `.source` substitution.
- `src/nnsight/intervention/interleaver.py:481` — `Interleaver.wrap_module`. Sentinel hook + skip-injection forward.
- `src/nnsight/intervention/interleaver.py:1354` — `Mediator.remove_hooks`. Drains every dynamic hook.
- `src/nnsight/intervention/tracing/iterator.py:95` — `register_iter_hooks`. Persistent tracker-bumping hooks.

## Lifecycle / sequence

For `with model.trace("hi"): a = model.layer.output.save(); b = model.layer.input.save()`:

1. Mediator starts. Worker enters compiled function.
2. Worker accesses `model.layer.output`. `requires_output` runs:
   - `iteration_tracker["model.layer.output"]` is 0.
   - `batcher.current_provider` is None (model hasn't started); not equal to requester. Install hook.
   - `output_hook` constructs hook with `iteration=0`, calls `add_ordered_hook(module, hook, "output")`. Handle added to `mediator.hooks`.
3. `eproperty.__get__` calls `mediator.request("model.layer.output.i0")`. Worker blocks.
4. Note: the user's code also references `model.layer.input` after `output`. **This is an order error** in the worker thread because layer's input fires in PyTorch *before* its output. To handle inputs and outputs from the same layer, the worker must access input first or use a separate invoker.
5. Suppose the user wrote `model.layer.input.save()` *first*, then `model.layer.output.save()`. Worker accesses `.input`:
   - `requires_input` runs; `iteration_tracker["model.layer.input"]` is 0.
   - `current_provider` is None; install hook via `input_hook`.
   - Handle added to `mediator.hooks`.
6. `eproperty.__get__` calls `mediator.request("model.layer.input.i0")`. Worker blocks again.
7. Main thread runs the model. Module's pre-hook chain fires:
   - `_forward_pre_hooks` is iterated in dict insertion order. Our pre-hook is the only entry (no sentinel pre-hook), so it fires.
   - `iteration_tracker["model.layer.input"] == 0 == iteration`, so the hook's body runs.
   - `handle.remove()` deletes itself from `_forward_pre_hooks`.
   - `mediator.handle("model.layer.input.i0", (args, kwargs))` matches; worker is responded to.
   - Worker continues briefly, hits `model.layer.output` access, registers an `output_hook`, and blocks again on `model.layer.output.i0`.
8. Module's forward runs.
9. Module's `_forward_hooks` chain fires:
   - Sentinel hook fires first (returns output unchanged).
   - Our one-shot output hook fires; matches and self-removes.
   - `mediator.handle("model.layer.output.i0", output)` responds to worker.
10. Worker resumes, `.save()`s, completes the function, sends `END`.
11. Mediator dies. `Interleaver.cancel` -> `mediator.remove_hooks` drains `mediator.hooks` (all four handles are already removed; `.remove()` calls are no-ops).

## Extension points

- **A new `requires_*` decorator.** If you have a custom `eproperty` stub that needs a different hook installed (e.g. a backward hook on a tensor), write a decorator that registers your hook before delegating to the stub. Add the handle to `mediator.hooks` so it gets cleaned up.
- **A new persistent hook.** If you need a hook that fires on every forward pass (cache-style), build it with `mediator_idx = float('inf')` and append to `mediator.hooks`. It will be drained in cleanup automatically. Make sure your hook reads relevant per-mediator state live on each fire — `mediator.batch_group`, `mediator.iteration_tracker`, etc.
- **A new operation-level hook.** Operation hooks live on `OperationAccessor`'s `pre_hooks` / `post_hooks` / `fn_hooks` lists, not on PyTorch dicts. Use `OperationHookHandle` for the handle so cleanup is uniform. See `docs/developing/source-accessor-internals.md`.
- **Custom hook ordering.** `add_ordered_hook` reads `hook.mediator_idx`. If you need finer-grained ordering between hooks owned by the same mediator (e.g. cache before iter), set `mediator_idx` to a value between the participating idxes (e.g. `mediator.idx + 0.5`) — Python sorts floats fine. The current code only uses int idxes plus `inf`.

## Related

- `docs/developing/interleaver-internals.md` — what `mediator.handle` does after a hook fires.
- `docs/developing/eproperty-deep-dive.md` — how `requires_output` integrates with the descriptor.
- `docs/developing/source-accessor-internals.md` — operation-level hooks.
- `docs/developing/performance.md` — measured impact of laziness on benchmarks.
- `NNsight.md` Section 3 — the original (pre-refactor) hook architecture; this doc supersedes it.
