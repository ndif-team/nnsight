---
title: Source Tracing
one_liner: .source rewrites a module's forward AST so each call site becomes a hookable operation; SourceAccessor and OperationAccessor own the per-module hook state, OperationEnvoy / SourceEnvoy are the per-Envoy wrappers.
tags: [concept, mental-model, source-tracing]
related: [docs/concepts/envoy-and-eproperty.md, docs/concepts/interleaver-and-hooks.md]
sources: [src/nnsight/intervention/source.py:359, src/nnsight/intervention/source.py:274, src/nnsight/intervention/source.py:571, src/nnsight/intervention/source.py:688, src/nnsight/intervention/source.py:548, src/nnsight/intervention/source.py:210, src/nnsight/intervention/hooks.py:495, src/nnsight/intervention/hooks.py:589]
---

# Source Tracing

## What this is for

`module.source` exposes intermediate operations *inside* a module's forward method — every function call, method call, and operator becomes a hookable provider path. nnsight reaches this by parsing the module's forward source with `ast`, wrapping every `Call` node with a per-call-site dispatcher, and re-executing the rewritten function as the new forward.

This is how you intercept `attention_interface(...)`, `self.c_proj(...)`, `torch.nn.functional.scaled_dot_product_attention(...)` etc. without subclassing the module.

## When to use / when not to use

- Use when you need access to a value that isn't a module's input or output — an internal intermediate.
- Use recursive `.source` (calling `.source` on an `OperationEnvoy`) to descend into the called function's body.
- **Do not** call `.source` on a sub-module from inside another `.source` — the system raises `ValueError` and tells you to access the module directly. Sub-modules already have their own source accessor.
- Source rewriting fails if the forward is not a regular Python function (compiled CUDA kernels, extension functions, etc.). Plain Python forwards always work.

## Canonical pattern

```python
import nnsight

model = nnsight.LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# Discover: print .source to see the rewritten forward with operation names.
print(model.transformer.h[0].attn.source)

with model.trace("Hello"):
    # Read an internal operation's output.
    attn_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()

    # Modify an internal operation's output.
    model.transformer.h[0].attn.source.self_c_proj_0.output[:] = 0
```

## Architecture

Two layers of objects, each split into a global "accessor" (per-module hook state) and a per-Envoy "wrapper" (user-facing API):

| Layer | Global (per-module) | Per-Envoy wrapper |
|-------|---------------------|-------------------|
| Module forward | `SourceAccessor` (`source.py:359`) | `SourceEnvoy` (`source.py:688`) |
| Single call site | `OperationAccessor` (`source.py:274`) | `OperationEnvoy` (`source.py:571`) |

The accessors are cached on the module itself as `module.__source_accessor__`. Per-Envoy wrappers live on the owning Envoy. Multiple Envoys / Interleavers wrapping the same module **share** the underlying accessors — hook lists, fn-replacement state, and nested SourceAccessors for recursive source.

## Discovery: print .source

Outside a trace, `print(model.transformer.h[0].attn.source)` shows the rewritten forward with operation names and line numbers:

```
                                *  def forward(self, hidden_states, ...):
                                  1     ...
  self_c_attn_0                -> 18    qkv = self.c_attn(hidden_states)
                                  19    ...
  attention_interface_0        -> 31    attn_output, attn_weights = attention_interface(
                                  32        self, query, key, value, ...
                                  33    )
  attn_output_reshape_0        -> 34    attn_output = attn_output.reshape(...)
  self_c_proj_0                -> 35    attn_output = self.c_proj(attn_output)
```

Operation names are `<dotted_func>_<index>`. Indexing disambiguates repeated calls to the same function.

Print a single operation to see it in surrounding context:

```python
print(model.transformer.h[0].attn.source.attention_interface_0)
# -> shows the operation highlighted with --> ... <-- markers
```

## How rewriting works

`SourceAccessor.__init__` (`source.py:374`) does the AST surgery via `convert(fn, self.wrap, path)` (`source.py:161`):

1. Read the module's forward source via `inspect.getsource`. Strip decorators (some, like transformers' `@auto_docstring`, fail when re-executed outside the class).
2. Parse with `ast.parse`. A `FunctionCallWrapper` `NodeTransformer` visits every `Call` node *inside* the function body and replaces it:
   ```python
   # Before:
   self.c_proj(attn_output)
   # After:
   wrap(self.c_proj, name="model.transformer.h.0.attn.self_c_proj_0")(attn_output)
   ```
3. Compile the rewritten AST and `exec` it against a global namespace cloned from the original module + a `wrap` binding. The exec produces a new function with the same name.
4. Build one `OperationAccessor` per call site, keyed on `<path>.<op_short_name>`.

The injected forward is **not** written onto the module — instead, `nnsight_forward` (the wrapper installed by `Interleaver.wrap_module`) checks `module.__source_accessor__` and calls the injected function via the accessor when present (`interleaver.py:533`).

## OperationAccessor: the hook state

Each `OperationAccessor` (`source.py:274`) owns four things:

- `pre_hooks: List[Callable]` — appended by `operation_input_hook`. Called with `(args, kwargs)`; non-`None` return replaces them.
- `post_hooks: List[Callable]` — appended by `operation_output_hook`. Called with the return value; non-`None` return replaces it.
- `fn_hooks: List[Callable]` — appended by `operation_fn_hook` for recursive `.source`. Called with the current fn; returns a (possibly replaced) fn.
- `fn_replacement: Optional[Callable]` — a one-shot fn replacement installed by `OperationEnvoy.source` for recursive source tracing. `wrap_operation` consumes and clears it before each call.

The `hooked` property is `True` if any list is non-empty or `fn_replacement` is set. `SourceAccessor.wrap` (`source.py:399`) takes the zero-overhead fast path (returns `fn` unchanged) when `hooked` is `False`.

## wrap_operation: the per-call-site dispatcher

When a call site is hooked, `SourceAccessor.wrap` returns a wrapper built by `wrap_operation` (`source.py:210`):

```python
@wraps(fn)
def inner(*args, **kwargs):
    actual_fn = op.fn_replacement or fn
    op.fn_replacement = None  # one-shot, clear immediately

    for hook in list(op.fn_hooks):
        actual_fn = hook(actual_fn)

    for hook in list(op.pre_hooks):
        result = hook((args, kwargs))
        if result is not None:
            args, kwargs = result

    value = actual_fn(*args, **kwargs)  # (with bound_obj handling)

    for hook in list(op.post_hooks):
        result = hook(value)
        if result is not None:
            value = result

    return value
```

Hook lists are read **live** at call time — hooks registered after the wrapper was built are still seen. `fn_replacement` is consumed *before* invocation to avoid races with the worker thread setting up the next step's replacement.

## OperationEnvoy: the per-Envoy view

`OperationEnvoy` (`source.py:571`) is a thin per-Envoy wrapper that satisfies `IEnvoy`. It hosts three eproperties:

```python
@eproperty()
@requires_operation_output
def output(self): ...

@eproperty(key="input")
@requires_operation_input
def inputs(self): ...

@eproperty(key="input")
@requires_operation_input
def input(self): ...   # with preprocess to extract first, postprocess to repack
```

The `requires_operation_*` decorators (`hooks.py:438`, `hooks.py:464`) register one-shot hooks on the underlying `OperationAccessor`'s hook lists rather than on a PyTorch module. See [Interleaver and Hooks](interleaver-and-hooks.md).

## SourceEnvoy: the per-module wrapper

`SourceEnvoy` (`source.py:688`) is what you get from `module.source`. It exposes one `OperationEnvoy` per call site as a regular attribute:

```python
src = model.transformer.h[0].attn.source     # SourceEnvoy
op  = src.attention_interface_0                # OperationEnvoy
val = op.output                                # eproperty access -> mediator request
```

Multiple `SourceEnvoy`s may wrap the same accessor (one per Envoy that touched `.source`); they share hook state via the underlying `OperationAccessor`s.

## Recursive .source

To descend into an operation's called function:

```python
sdpa = (
    model.transformer.h[0].attn
        .source.attention_interface_0
        .source.torch_nn_functional_scaled_dot_product_attention_0
        .output.save()
)
```

`OperationEnvoy.source` (`source.py:632`) handles this:

1. First access: register an `operation_fn_hook` and request the operation's currently-bound fn via `mediator.request("...fn")`. The model is currently *inside* `wrap_operation` for that call site; the fn-hook fires and delivers the fn to the worker.
2. The worker builds a nested `SourceAccessor(fn, path)` and stores it on `op.fn_replacement` so the operation, currently mid-flight, uses the injected version this step.
3. The injected fn is also cached as `accessor._source_accessor` so subsequent `.source` accesses re-install it.
4. A `SourceEnvoy(nested_accessor, interleaver)` is returned for user access.

The fn-hook + swap dance is critical: by the time the user accesses `.source`, the operation is already running. nnsight has to substitute the injected fn for the rest of *this* call.

## Caching across forward replacement

`get_or_create_source_accessor(module)` (`source.py:548`) caches the `SourceAccessor` on `module.__source_accessor__`. This survives:

- `torch.compile` re-binding `forward`.
- accelerate's `_old_forward` swap on dispatch.
- nnsight's own `_update` (e.g. when meta-tensor weights are loaded).

`Envoy._update` (`envoy.py:796`) detects the swap and calls `accessor.rebind(new_fn)` (`source.py:431`) — this re-injects against the new fn but **preserves all OperationAccessor hook state** (lists, `fn_replacement`, nested accessors). Pre-existing `OperationEnvoy` / `SourceEnvoy` references stay valid.

## Iteration tracking for source

The persistent iter-tracker hooks (registered by `IteratorTracer`) bump operation-level paths in lockstep with the parent module via `bump_source_paths` (`tracing/iterator.py:75`). The recursion descends through `OperationAccessor._source_accessor` so deeply nested `.source` chains stay in sync.

**Known limitation**: if a `SourceAccessor` is built for the first time mid-iter-loop (step N>0), op-path trackers start at 0 instead of N — the user's first hook captures `iteration=N` but checks against `tracker[op]=0`, so that one access misses. Subsequent steps work because per-fire bumping advances both counters together.

## Lifetimes

- `SourceAccessor` and `OperationAccessor` live as long as the module itself.
- `SourceEnvoy` / `OperationEnvoy` live as long as their owning Envoy.
- Operation hooks (input / output / fn) are one-shot and self-remove on fire. They're also tracked on `mediator.hooks` for cleanup.
- `fn_replacement` is one-shot per call to `wrap_operation`. Re-accessing `.source` reinstalls it from the cached nested accessor.

## Gotchas

- **Don't call `.source` on a sub-module from inside another `.source`.** Access the sub-module directly (`module.attn.something.source`, not `module.attn.source.something.source`). The system raises `ValueError` if you try.
- **Forward must be a real Python function.** `inspect.getsource` is required. Compiled extensions (custom CUDA kernels exposed as functions) won't work.
- **Decorators on the forward are stripped.** This is intentional — re-executing a decorator outside its class context can fail (e.g. transformers' `@auto_docstring`). If a decorator was load-bearing (changing behavior beyond docstrings), source tracing will diverge from the original.
- **Operation names depend on parsing.** `self.c_proj(x)` becomes `self_c_proj_0`. If the module's source changes (e.g. a transformers version bump renames an internal call), the operation name changes too.
- **First-time source accessor mid-iter-loop misses one step** (see Known limitation above). Build accessors before entering the loop if you need step 0 — usually by accessing `.source` once before `for step in tracer.iter[:]`.

## Related

- [Envoy and eproperty](envoy-and-eproperty.md) — `OperationEnvoy` is an `IEnvoy` like `Envoy`.
- [Interleaver and Hooks](interleaver-and-hooks.md) — operation hook registration and `OperationHookHandle`.
- Source: `src/nnsight/intervention/source.py` (full module), `src/nnsight/intervention/hooks.py` (`operation_input_hook`, `operation_output_hook`, `operation_fn_hook`, `requires_operation_*`), `src/nnsight/intervention/tracing/iterator.py` (`bump_source_paths`).
