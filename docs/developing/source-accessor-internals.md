---
title: Source Accessor Internals
one_liner: AST-based forward injection for in-module operation tracing.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/lazy-hook-system.md, docs/developing/eproperty-deep-dive.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/hooks.py, src/nnsight/intervention/envoy.py]
---

# Source Accessor Internals

## What this covers

`.source` is the API for hooking individual operations *inside* a module's forward pass. For example:

```python
with model.trace("hi"):
    attn_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()
```

Behind that line is a complete pipeline that AST-rewrites the module's forward function, replaces every call site with a hook-routing wrapper, and stores hook state on per-call-site `OperationAccessor` objects shared across all Envoys that touch the module.

This doc covers:

- The split between *global* per-module accessors and *per-Envoy* user-facing wrappers.
- `FunctionCallWrapper` — the AST transformer that rewrites every `Call` node.
- `convert` — the source-string -> compiled-fn pipeline.
- `wrap_operation` — the per-call-site hook dispatcher.
- `SourceAccessor.wrap` — the runtime lookup that decides whether to install a wrapper.
- Recursive `.source` and the `fn_hooks` + `fn_replacement` mechanism.
- `resolve_true_forward` — unwrapping accelerate and nnsight forward shims.
- `SourceAccessor.rebind` — surviving `module.forward` replacement (e.g. dispatch).

## Architecture

### Global accessors vs per-Envoy wrappers

```
                    PyTorch module (one)
                          |
                          v
         module.__source_accessor__: SourceAccessor   <-- global, lifetime of module
            |                  |
            |   {op_name: OperationAccessor}          <-- global, lifetime of module
            v                  v
       _forward fn        OperationAccessor
       (injected)             pre_hooks
                              post_hooks               <-- one-shot per access
                              fn_hooks
                              fn_replacement
                              _source_accessor (recursive)
                              ^
                              |
              SourceEnvoy --> OperationEnvoy (one per Envoy)  <-- per-Envoy wrappers
              ^                  ^
              |                  |
        Envoy.source        SourceEnvoy.<op_name>
```

Two reasons for the split:

1. **Hook state must be shared.** When two Envoys wrap the same module (re-wrapping for `model.edit()` non-inplace, or wrapping the same underlying model with two `NNsight` instances), they must register hooks on the *same* `OperationAccessor` lists. Otherwise their hooks would be in separate state and the runtime forward would only see one set.
2. **The injected forward must be cached.** AST parsing + `compile()` is expensive. Building the injected forward once per module and reusing it across Envoys is the only way to keep `.source` access cheap.

The accessor is cached at `module.__source_accessor__` (`source.py:548`-`563`). This survives even if `module.forward` gets re-bound by `torch.compile`, accelerate's hot-swap, or other wrappers — the routing through the accessor is done by `nnsight_forward` reading the attribute on every call.

### FunctionCallWrapper — the AST transformer

`FunctionCallWrapper` (`src/nnsight/intervention/source.py:92`) is an `ast.NodeTransformer` that rewrites every `ast.Call` node in the module's forward into:

```python
wrap(original_fn, name=qualified_op_name)(original_args, **original_kwargs)
```

That is, the original call:

```python
attn_output = self.attn(hidden_states, attention_mask)
```

becomes:

```python
attn_output = wrap(self.attn, name="model.transformer.h.0.self_attn_0")(hidden_states, attention_mask)
```

where `wrap` is `SourceAccessor.wrap` bound at compile time, and the name is built from the dotted module path plus a per-name index (e.g. `self_attn_0`, `self_attn_1` for two calls to `self.attn` in the same forward).

#### How names are built (`get_name`, source.py:104)

For an `ast.Name` node like `func()`: name = `"func_<idx>"`.
For an `ast.Attribute` node like `self.foo.bar()`: name = `"self_foo_bar_<idx>"` (joined by underscores, reversed traversal of the attribute chain).
For anything else (e.g. a subscript): name = `"unknown_<idx>"`.

The `name_index` `defaultdict(int)` ensures repeated calls to the same fn get distinct names.

#### Why decorators are stripped (source.py:183-185)

Before walking the AST, `convert()` strips `decorator_list` from **every** `FunctionDef` and `AsyncFunctionDef` it finds:

```python
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
```

This is unconditional — `ast.walk` visits every function definition, and *all* decorators are removed regardless of what they are. That includes `@staticmethod`, `@classmethod`, `@property`, `@torch.no_grad()`, and class-state-dependent decorators like transformers' `@auto_docstring`. Confirmed by direct AST inspection: every `decorator_list` becomes empty after `convert()`.

**Why strip them at all?** The rewritten forward is `exec`'d outside its original class context. Decorators that depend on class state (e.g. `@auto_docstring`, which reads `cls.__mro__`) would crash. Stripping unconditionally is simpler than trying to detect which decorators are "safe" to keep.

**Practical consequence — `@staticmethod` / `@classmethod` on a forward.** If a module's `forward` (or a nested function inside it) is declared with `@staticmethod` or `@classmethod`, the rewritten version loses that marker. In typical PyTorch modules this is **not** an issue — `forward` is always a regular instance method. The rewritten function expects the module as its first positional argument, which `SourceAccessor.__call__` provides (`source.py:459`).

The edge case worth knowing: if `source` tracing is being applied to a non-module function that *did* depend on a method-descriptor decorator, the stripped version may not behave identically to the original. In practice this hasn't surfaced because `.source` is invoked on regular module forwards.

#### Why the function-start filter (source.py:127)

`visit_Call` only wraps calls *after* the first `FunctionDef`/`AsyncFunctionDef` line. Calls in the function's signature (default arguments) or in module-level code preceding the function (top-of-file imports, etc.) are left alone. This is a defensive measure — you don't want to wrap `Optional[Tensor]` in a default value, for example.

### convert — the compile pipeline

`convert(fn, wrap, name)` (`src/nnsight/intervention/source.py:161`) is the single entry point that takes an unwrapped function and returns:

1. The original source string (used for pretty-printing in `__str__`).
2. A `{op_name: line_number}` map.
3. The compiled, executed wrapped function.

Steps:

1. `inspect.getsource(fn)` -> dedent -> AST parse.
2. Strip decorators from every function definition.
3. Run `FunctionCallWrapper(name).visit(tree)`.
4. `ast.fix_missing_locations(tree)`.
5. Build a globals namespace combining `inspect.getmodule(fn).__dict__` and the local `wrap` function.
6. `compile(tree, "<nnsight>", "exec")` and `exec()` into the namespace.
7. Pull the rewritten function out of the namespace by name and return.

The `<nnsight>` filename is used for the compiled code object. This is what shows up in tracebacks if the rewritten code raises — paired with `Tracer.Info`'s line-number reconstruction in `wrap_exception`, this is enough to give users a usable stack.

### wrap_operation — the per-call-site dispatcher

`wrap_operation(fn, name, bound_obj=None, op_accessor=None)` (`src/nnsight/intervention/source.py:210`) builds the actual wrapper that dispatches hooks at runtime:

```python
@wraps(fn)
def inner(*args, **kwargs):
    actual_fn = (
        op_accessor.fn_replacement
        if op_accessor.fn_replacement is not None
        else fn
    )
    op_accessor.fn_replacement = None  # one-shot, consume immediately

    for hook in list(op_accessor.fn_hooks):
        actual_fn = hook(actual_fn)

    for hook in list(op_accessor.pre_hooks):
        result = hook((args, kwargs))
        if result is not None:
            args, kwargs = result

    if not inspect.ismethod(actual_fn) and bound_obj is not None:
        value = actual_fn(bound_obj, *args, **kwargs)
    else:
        value = actual_fn(*args, **kwargs)

    for hook in list(op_accessor.post_hooks):
        result = hook(value)
        if result is not None:
            value = result

    return value
```

Three hook lists, each with different semantics:

- **`fn_hooks`** — receive the current fn, return the (possibly replaced) fn. Used by `operation_fn_hook` for recursive `.source` to substitute the AST-injected version of the op's own fn.
- **`pre_hooks`** — receive `(args, kwargs)`, return the (possibly replaced) tuple or `None` to leave unchanged. Used by `requires_operation_input` for `.input` access.
- **`post_hooks`** — receive the return value, return the (possibly replaced) value or `None`. Used by `requires_operation_output` for `.output` access.

Plus `fn_replacement`, which is a one-shot replacement consumed *before* invoking the hooks. The order matters: `fn_replacement` is consumed first because clearing it later would race with the worker thread, which might set `fn_replacement` for the next forward pass while `actual_fn` is still running. See the comment in source.py:238.

The `bound_obj` handling (source.py:254) is for cases where the original `fn` is an unbound method that needs `self` passed explicitly. This happens when the AST rewriter sees `self.attn(...)` — the rewritten call is `wrap(self.attn, name=...)(...)`, and `self.attn` may resolve to a bound method or to an unbound function depending on whether `self` is `module` (where `attn` is a sub-module attribute) or some other object. `wrap` handles both cases by checking `inspect.ismethod(fn)`.

### SourceAccessor.wrap — the per-call dispatcher

`SourceAccessor.wrap(fn, **kwargs)` (`src/nnsight/intervention/source.py:399`) is what `FunctionCallWrapper` substitutes into the rewritten AST. It runs *every time* an op call site is hit:

```python
def wrap(self, fn, **kwargs):
    name = kwargs["name"]
    op = self.operations.get(name)
    if op is None or not op.hooked:
        return fn  # zero-overhead fast path
    bound_obj = (
        fn.__self__
        if inspect.ismethod(fn) and getattr(fn, "__name__", None) != "forward"
        else None
    )
    return wrap_operation(fn, name=name, bound_obj=bound_obj, op_accessor=op)
```

The `op.hooked` fast path is essential for performance. `op.hooked` (source.py:326) is true if any of `pre_hooks`, `post_hooks`, `fn_hooks`, or `fn_replacement` is non-empty. When no consumer has accessed this operation, `wrap` returns the original fn unchanged — the call site pays the cost of one dict lookup + one bool check, nothing more.

The `bound_obj` heuristic excludes `forward` to avoid double-binding when the call site is `self.forward(...)` on a wrapped sub-module — the wrapped module's `__call__` already binds `self`.

### Forward routing — wrap_module's role

`Interleaver.wrap_module` (`interleaver.py:481`) replaces `module.forward` with `nnsight_forward`. That wrapper checks `module.__source_accessor__` on every call:

```python
source_accessor = getattr(m, "__source_accessor__", None)
if source_accessor is not None:
    return source_accessor(m, *args, **kwargs)
return m.__nnsight_forward__(m, *args, **kwargs)
```

`source_accessor(m, *args, **kwargs)` calls `SourceAccessor.__call__` (source.py:459), which invokes the AST-rewritten `_forward` with the module as the first positional argument. The rewritten code then proceeds, hitting `wrap` calls at every original call site.

Note the lack of a `source_accessor.hooked` check here. Even if no operation under the accessor is currently hooked, we still route through the injected forward. The reason is documented at interleaver.py:543-548: hooks may be registered *mid-forward* (e.g. if an upstream op-level hook delivers a value to the worker, the worker may register a downstream op-level hook before resuming), and a gate at entry would have already taken the un-injected path. The per-op fast path inside `wrap` is sufficient.

### OperationAccessor

`OperationAccessor` (`src/nnsight/intervention/source.py:274`) holds:

```python
class OperationAccessor:
    path: str                     # fully-qualified, e.g. "model.transformer.h.0.attn.split_1"
    source_code: str              # source of the parent forward (for __str__)
    line_number: int              # line within source_code where this op lives
    pre_hooks: List[Callable]
    post_hooks: List[Callable]
    fn_hooks: List[Callable]
    fn_replacement: Optional[Callable]
    _source_accessor: Optional[SourceAccessor]   # for recursive .source
```

`hooked` is True if any of the four state fields is non-empty.

`__str__` pretty-prints a code excerpt around the operation, with the target line highlighted as `--> ... <--`. Used by `print(model.layer.source.op_0)`.

### SourceAccessor

`SourceAccessor` (`src/nnsight/intervention/source.py:359`) is built once per `(module, fn)` pair and cached:

- For modules: `module.__source_accessor__`.
- For recursive `.source`: cached on `OperationAccessor._source_accessor`.

```python
class SourceAccessor:
    path: str                                 # "model.transformer.h.0.attn"
    source: str                               # original source string
    line_numbers: Dict[str, int]              # op_name -> line
    _forward: Callable                        # the AST-rewritten fn
    operations: Dict[str, OperationAccessor]  # full_name -> accessor
```

Construction:

1. `convert(fn, self.wrap, path)` builds the rewritten fn.
2. For each `(op_short_name, line)` in `line_numbers`, build an `OperationAccessor` with `full_name = f"{path}.{op_short_name}"`.

`__call__(self, *args, **kwargs)` invokes `self._forward(*args, **kwargs)`. The first positional argument should be the module — the rewritten function expects an unbound first arg.

`__iter__` yields each operation's short name, deduplicated by line number (so a single line with multiple chained calls only yields once).

`__str__` pretty-prints the source with op names left-aligned next to their lines.

### rebind — surviving forward replacement

`SourceAccessor.rebind(fn)` (`src/nnsight/intervention/source.py:431`) is called when the underlying module's true forward changes — most importantly, on dispatch (when the meta-tensor module is replaced by the loaded module). It:

1. Re-runs `convert(fn, self.wrap, self.path)` with the new fn.
2. Replaces `self.source`, `self.line_numbers`, `self._forward`.
3. For each `(op_short_name, line)` in the new line_numbers:
   - If an existing `OperationAccessor` is found by full name, update only its `line_number` and `source_code`.
   - Otherwise, create a fresh `OperationAccessor`.

The crucial point: existing `OperationAccessor` instances are kept. Their `pre_hooks`, `post_hooks`, `fn_hooks`, `fn_replacement`, and `_source_accessor` survive the rebind. Any `OperationEnvoy` or `SourceEnvoy` references the user code is holding remain valid.

`Envoy._update` (`envoy.py:796`) is the caller. The flow on dispatch is:

```python
old_accessor = getattr(self._module, "__source_accessor__", None)
self._module = module
self.interleaver.wrap_module(module)
if old_accessor is not None:
    old_accessor.rebind(resolve_true_forward(module))
    module.__source_accessor__ = old_accessor
```

### resolve_true_forward — unwrapping accelerate and nnsight

`resolve_true_forward(module)` (`src/nnsight/intervention/source.py:517`) finds the unwrapped function whose AST should be injected. A module's `forward` may have been wrapped by:

- **accelerate**: device-mapped models replace `forward` with `partial(new_forward, module)`, which calls `module._old_forward(*args, **kwargs)`. The true fn is `module._old_forward.__func__` (or `partial.func`).
- **nnsight**: `Interleaver.wrap_module` replaces `forward` with `nnsight_forward`, which calls `module.__nnsight_forward__(module, *args, **kwargs)`. The true fn is `module.__nnsight_forward__`.

Resolution priority:

1. `module._old_forward` if present (accelerate).
2. `module.__nnsight_forward__` if present (nnsight).
3. Fallback: `type(module).forward`.

The returned fn is unbound — it expects `module` as its first positional argument. This matches how `SourceAccessor.__call__` invokes it.

### Recursive .source

`OperationEnvoy.source` (`src/nnsight/intervention/source.py:632`-`685`) is a property that builds a nested `SourceAccessor` over the *operation's own fn*. This is how you trace into the body of `attention_interface`:

```python
with model.trace():
    nested = model.transformer.h[0].attn.source.attention_interface_0.source.torch_nn_functional_scaled_dot_product_attention_0.output.save()
```

The dance:

1. First-ever access for this op:
   - `operation_fn_hook(mediator, accessor)` registers a one-shot hook on `accessor.fn_hooks`.
   - `mediator.request(f"{path}.fn")` blocks the worker.
2. The op fires (because the user's outer trace runs the model). `wrap_operation` invokes `fn_hooks`, which in turn calls `mediator.handle(f"{op_path}.fn", fn)`. The worker sees the fn arriving as a VALUE event.
3. The worker checks the fn isn't a Module (raise if so — see source.py:658), then constructs a fresh `SourceAccessor(fn, self.path)` and stores it on `accessor._source_accessor`.
4. The worker `swap()`s the injected fn back into the model: `mediator.swap(f"{path}.fn", nested._forward)`. `Mediator.handle_swap_event` matches and updates `batcher.current_value`. Back in `Interleaver.handle`, the swapped value is returned to `wrap_operation`'s `fn_hooks` loop, which uses it as `actual_fn`.
5. As a backup (because `fn_hooks` is one-shot — the hook self-removes after firing), the worker also sets `accessor.fn_replacement = nested._forward`. `wrap_operation` consumes `fn_replacement` on each call until cleared, so subsequent forward passes still see the injected fn.

For subsequent `.source` accesses (cached `_source_accessor`), only step 5 runs — `fn_replacement` is reinstalled directly without going through the hook dance.

### Per-Envoy SourceEnvoy and OperationEnvoy

`SourceEnvoy` and `OperationEnvoy` (`src/nnsight/intervention/source.py:571`, source.py:688) are per-Envoy wrappers over the global accessors. They:

- Implement `IEnvoy` (`path`, `interleaver`) so they can host `eproperty` descriptors.
- Delegate hook registration to the underlying accessor, so multiple Envoys wrapping the same module land on the same hook lists.
- Provide attribute access (`source.attention_interface_0`) to the user; this attribute is set in `SourceEnvoy.__init__` from `accessor.line_numbers.keys()`.

`OperationEnvoy` has three eproperties (source.py:603):

```python
@eproperty()
@requires_operation_output
def output(self): ...

@eproperty(key="input")
@requires_operation_input
def inputs(self): ...

@eproperty(key="input")
@requires_operation_input
def input(self): ...
```

The `key="input"` on `inputs` and `input` is intentional — both share a single hook lifecycle (one operation pre-hook fires, both descriptors see the value via `current_provider` matching). `input.preprocess` extracts the first positional/keyword arg; `input.postprocess` repacks a single value back into `(args, kwargs)`. Same pattern as the module-level `Envoy.input` / `Envoy.inputs`.

## Key files / classes

- `src/nnsight/intervention/source.py:92` — `FunctionCallWrapper`. AST rewriter for `Call` nodes.
- `src/nnsight/intervention/source.py:161` — `convert`. Source -> AST -> compile pipeline.
- `src/nnsight/intervention/source.py:210` — `wrap_operation`. Per-call-site hook dispatcher.
- `src/nnsight/intervention/source.py:274` — `OperationAccessor`. Hook state for one call site.
- `src/nnsight/intervention/source.py:359` — `SourceAccessor`. Per-module injected forward + op map.
- `src/nnsight/intervention/source.py:399` — `SourceAccessor.wrap`. Runtime dispatcher; the fast path lives here.
- `src/nnsight/intervention/source.py:431` — `SourceAccessor.rebind`. Re-injects against a new fn.
- `src/nnsight/intervention/source.py:459` — `SourceAccessor.__call__`. Invokes the injected forward.
- `src/nnsight/intervention/source.py:517` — `resolve_true_forward`. Unwraps accelerate / nnsight shims.
- `src/nnsight/intervention/source.py:548` — `get_or_create_source_accessor`. Builds + caches on the module.
- `src/nnsight/intervention/source.py:571` — `OperationEnvoy`. Per-Envoy wrapper with eproperties.
- `src/nnsight/intervention/source.py:632` — `OperationEnvoy.source`. Recursive `.source` entry point.
- `src/nnsight/intervention/source.py:688` — `SourceEnvoy`. Per-Envoy wrapper over `SourceAccessor`.
- `src/nnsight/intervention/hooks.py:495` — `operation_output_hook`. One-shot list-based output hook.
- `src/nnsight/intervention/hooks.py:549` — `operation_input_hook`. One-shot input hook.
- `src/nnsight/intervention/hooks.py:589` — `operation_fn_hook`. Recursive-source fn replacement hook.

## Lifecycle / sequence

For `model.transformer.h[0].attn.source.attention_interface_0.output.save()` inside a trace:

1. `model.transformer.h[0].attn.source` (`Envoy.source` property, envoy.py:215) is accessed:
   - `get_or_create_source_accessor(self._module)` looks up `module.__source_accessor__`. None present.
   - `resolve_true_forward(module)` finds the true fn (e.g. `GPT2Attention.forward`).
   - `SourceAccessor(fn, "model.transformer.h.0.attn")` runs `convert`, which AST-rewrites the forward and builds `OperationAccessor` instances for each call site (e.g. `attention_interface_0`, `self_c_proj_0`, etc.).
   - Cached on `module.__source_accessor__`.
   - `SourceEnvoy(accessor, interleaver)` is built and cached on `Envoy._source`. `__init__` walks `accessor.line_numbers` and creates an `OperationEnvoy` per op, attached as attributes.
2. `.attention_interface_0` returns the `OperationEnvoy` for that op.
3. `.output` is an `eproperty` with `requires_operation_output`:
   - `requires_operation_output` runs; current_provider doesn't match; calls `operation_output_hook(mediator, self.accessor)`.
   - `operation_output_hook` builds a one-shot closure with `iteration=0`, appends to `accessor.post_hooks`, builds an `OperationHookHandle`, appends to `mediator.hooks`.
   - `eproperty.__get__` calls `mediator.request("model.transformer.h.0.attn.attention_interface_0.output.i0")`. Worker blocks.
4. Main thread runs the model. PyTorch enters `attn.forward`:
   - `nnsight_forward` checks `__source_accessor__`, finds it, invokes `source_accessor(module, ...)` -> the AST-rewritten fn.
   - The rewritten fn hits `wrap(attention_interface, name="model.transformer.h.0.attn.attention_interface_0")(...)`.
   - `SourceAccessor.wrap` looks up the operation; `op.hooked == True`; returns `wrap_operation(...)` wrapper.
   - The wrapper is called: `fn_hooks` empty, `pre_hooks` empty, calls `attention_interface(...)`, gets `value`, runs `post_hooks`.
   - Our one-shot post-hook fires: matches `iteration_tracker == 0`, removes itself, calls `mediator.handle("...attention_interface_0.output.i0", value)`. Mediator delivers value to worker.
   - Worker resumes, possibly mutates and swaps, then continues.
5. `attn.forward` finishes; the rest of the model runs.
6. `Mediator.remove_hooks` at session end drains any leftover entries (the post_hook is already removed by the closure; that's a no-op).

## Extension points

- **Custom op-level eproperty.** Subclass `OperationEnvoy` and add new eproperties decorated with `requires_operation_input` / `requires_operation_output`, or write your own decorator that registers a hook on `accessor.pre_hooks` / `post_hooks`. Be sure to wrap the handle in `OperationHookHandle` and append to `mediator.hooks`.
- **Alternative AST transforms.** `FunctionCallWrapper` only wraps `Call` nodes. If you want to expose other constructs (e.g. attribute reads, comprehensions), you can build a parallel `NodeTransformer` and integrate it into `convert`. The hook list pattern (`pre_hooks`/`post_hooks`/`fn_hooks`) generalizes to whatever runtime hook semantics you need.
- **Source for non-PyTorch fns.** `convert` works on any `Callable` whose `inspect.getsource` succeeds. Use `SourceAccessor(fn, path)` directly to AST-rewrite arbitrary functions. The vLLM integration uses this to instrument scheduler-side callbacks.
- **Forward shim integration.** If you add a new forward shim (analogous to accelerate's `_old_forward` or nnsight's `__nnsight_forward__`), extend `resolve_true_forward` to detect it. Otherwise `.source` will instrument your shim instead of the user's actual code.

## Related

- `docs/developing/lazy-hook-system.md` — the hook lifecycle that operation hooks plug into.
- `docs/developing/eproperty-deep-dive.md` — how `OperationEnvoy.output` reaches `requires_operation_output`.
- `docs/developing/interleaver-internals.md` — `mediator.handle` semantics for op-level providers.
- `NNsight.md` Section 4.3 — the original design narrative for source tracing.
