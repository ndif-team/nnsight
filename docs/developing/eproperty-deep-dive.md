---
title: eproperty Deep Dive
one_liner: The descriptor protocol for hookable values, the formal extension API for envoys.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/lazy-hook-system.md, docs/developing/interleaver-internals.md, docs/developing/source-accessor-internals.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/hooks.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py]
---

# eproperty Deep Dive

## What this covers

`eproperty` is the descriptor that powers every hookable value in nnsight: `Envoy.output`, `Envoy.input`, `Envoy.inputs`, `OperationEnvoy.output` / `.input` / `.inputs`, `InterleavingTracer.result`, vLLM's `model.logits` and `model.samples`, and any custom property a downstream consumer adds. It is the **formal extension API** for telling the interleaver "here is a new value users can read or write inside a trace."

This doc walks through:

- The `IEnvoy` protocol that every host class must satisfy.
- `eproperty.__init__` arguments: `key`, `description`, `iterate`.
- The decorated stub idiom: an empty body whose decorators do all the real work.
- `eproperty.__get__` and `eproperty.__set__` flow.
- `preprocess`, `postprocess`, and `transform` and how they compose.
- `eproperty.provide` for runtime-side pushes.
- Reading the source: per-head attention as a worked example.

## Architecture

### The IEnvoy protocol

```python
@runtime_checkable
class IEnvoy(Protocol):
    interleaver: "Interleaver"
    path: str
```

Defined at `src/nnsight/intervention/interleaver.py:42`. Any class that wants to host `eproperty` descriptors must satisfy this protocol — that is, expose `.interleaver` and `.path` attributes. Current implementors:

- `Envoy` (`envoy.py:54`) — `path` like `"model.transformer.h.0"`.
- `OperationEnvoy` (`source.py:571`) — `path` like `"model.transformer.h.0.attn.split_1"`.
- `SourceEnvoy` (`source.py:688`) — `path` like `"model.transformer.h.0.attn"` (mostly for delegation).
- `InterleavingTracer` (`tracing/tracer.py:269`) — `path` is empty / not set; just the bare `key`.
- vLLM `VLLM` — runtime-side push via `eproperty.provide`.

`path` is allowed to be `None` or empty. In that case, `_build_requester` (`interleaver.py:260`) returns just the `key` without a prefix. This is how `InterleavingTracer.result` produces the bare requester `"result"`.

### The eproperty descriptor

```python
class eproperty:
    name: str                 # set from decorated stub's __name__
    key: str                  # the suffix appended to path; defaults to name
    description: str          # shown in repr; None hides from repr
    iterate: bool             # whether to append .iN suffix; default True

    _hook: Callable           # the decorated stub (with pre-setup decorators applied)
    _preprocess: Optional[Callable]
    _postprocess: Optional[Callable]
    _transform: Optional[Callable]
```

Defined at `src/nnsight/intervention/interleaver.py:60`-`335`.

- **`key`** is what gets appended to `path` to build the requester. Defaults to `name` (the stub's `__name__`). Can be set explicitly to share a key across multiple eproperties: `Envoy.input` and `Envoy.inputs` both use `key="input"` so a single hook fires both.
- **`description`** is a short human-readable label shown in the Envoy's `repr` tree (e.g. `(output): Module output`). Eproperties without a description are hidden from `repr` — useful for utility properties that aren't meant to be discoverable.
- **`iterate=True`** appends `.iN` where N is the resolved iteration. Set `iterate=False` for non-iteration values like `InterleavingTracer.result` (a trace produces exactly one result, no iteration semantics).

### The decorated stub idiom

The defining shape of an eproperty:

```python
class Envoy(Batchable):
    @eproperty()
    @requires_output      # <-- the decorator that does the actual work
    def output(self) -> Object:
        """Get the output of the module's forward pass."""
        # body is intentionally empty
```

Three things happen at class-definition time:

1. `requires_output(output_stub)` returns a `wrapper` that, when called, registers a one-shot output hook on the underlying module.
2. `eproperty()` returns an `eproperty` instance.
3. `eproperty.__call__(wrapper)` is invoked (because `eproperty()` is a callable factory). It:
   - Stores `wrapper` as `self._hook`.
   - Reads `wrapper.__name__` (which is `"output"`, preserved by `functools.wraps` in the decorator) and stores it as `self.name`.
   - Defaults `self.key = self.name` if not explicitly set.
   - Returns `self` (the descriptor) so the class attribute is the eproperty.

The body of the stub is *never executed for its return value*. It is a placeholder. The two jobs of the stub are:

1. **Donate `__name__` and `__doc__`** to the descriptor. The name becomes the default key; the docstring is what users see in `help(model.layer.output)`.
2. **Carry the decorators stacked on top of it.** Those decorators register the PyTorch hook (or operation hook) at the right moment.

This is unusual for Python descriptors but it has a major payoff: defining a new hookable value is just three lines, and the descriptor is uniform across module-level, op-level, and runtime-side values.

### __get__ flow

```python
def __get__(self, obj, owner):
    if obj is None:
        return self                           # accessing on the class, not instance
    interleaver = obj.interleaver
    if interleaver.interleaving:
        requester = self._build_requester(obj)
        self._hook(obj)                        # <-- pre-setup work happens here
        if self.iterate:
            requester = interleaver.iterate_requester(requester)
        value = interleaver.current.request(requester)
        if self._preprocess is not None:
            value = self._preprocess(obj, value)
        if self._transform is not None:
            interleaver.current.transform = partial(self._transform, value)
    else:
        raise ValueError(f"Cannot access `{label}` outside of interleaving.")
    return value
```

(`src/nnsight/intervention/interleaver.py:264`-`304`)

Step-by-step:

1. **Access on class** (`obj is None`) — return the descriptor itself. Used by `Envoy.__repr__` to find description fields.
2. **Outside of interleaving** — raise. Eproperties only mean something when there is a model running.
3. **Build the requester** — `f"{path}.{key}"` if `path` is truthy, else just `key`.
4. **Run the stub** (`self._hook(obj)`) — this is the magic. The stub itself is a no-op, but the `requires_*` decorator stacked on top of it has wrapped it into a function that, when called, **registers a one-shot PyTorch hook** on the underlying module so the value will be produced. After the hook fires, the worker thread receives the value via `mediator.request`.
5. **Apply iteration suffix** if `iterate=True` — `iterate_requester` resolves between explicit `mediator.iteration` (set inside `tracer.iter[N]`) and `mediator.iteration_tracker[requester]` (the persistent tracker).
6. **Block the worker** — `interleaver.current.request(requester)` enqueues a VALUE event and waits on the response queue. The worker is blocked here until the model fires the hook installed in step 4.
7. **Preprocess** — if registered, transform the raw value before returning to the user. For example, `Envoy.input.preprocess` extracts the first positional arg from `(args, kwargs)`.
8. **Bind transform** — if `transform` is registered, partial-bind the preprocessed value into it and store on `mediator.transform`. The mediator will fire it later (see "Transform" below).
9. **Return** — the user's code in the worker thread receives the value.

### __set__ flow

```python
def __set__(self, obj, value):
    if self._postprocess is not None:
        value = self._postprocess(obj, value)
    interleaver = obj.interleaver
    if interleaver.interleaving:
        requester = self._build_requester(obj)
        self._hook(obj)
        if self.iterate:
            requester = interleaver.iterate_requester(requester)
        interleaver.current.swap(requester, value)
    else:
        raise ValueError(f"Cannot set `{label}` outside of interleaving.")
```

(`src/nnsight/intervention/interleaver.py:306`-`326`)

Almost the same as `__get__`, but:

1. **Postprocess** runs *first*, on the user-supplied value, before the hook is installed. For example, `Envoy.input.postprocess` repacks a single value back into the `(args, kwargs)` shape the model's hook expects.
2. **Run the stub** to register the hook (same as `__get__`).
3. **Swap** — `interleaver.current.swap(requester, value)` enqueues a SWAP event. The mediator will, on `handle_swap_event`, call `batcher.swap` to overwrite the live value before the model proceeds.

### preprocess

`@<eprop>.preprocess` registers a function called on `__get__` between the request returning and the value reaching the user. Signature: `(self, value) -> transformed_value`.

Use cases:

- **Extract a sub-value.** `Envoy.input.preprocess` extracts the first positional/keyword arg from `(args, kwargs)`:
  ```python
  @input.preprocess
  def input(self, value):
      return [*value[0], *value[1].values()][0]
  ```
- **Reshape a view.** Per-head attention: reshape `[B, S, H]` into `[B, n_heads, S, head_dim]`.

The user receives whatever `preprocess` returns. If they want to mutate it in place and have those mutations reach the model, they need a `transform` (see below).

### postprocess

`@<eprop>.postprocess` registers a function called on `__set__` before the hook is installed. Signature: `(self, value) -> transformed_value`.

Use cases:

- **Repack into the model's expected shape.** `Envoy.input.postprocess` repacks a single value back into `(args, kwargs)`:
  ```python
  @input.postprocess
  def input(self, value):
      inputs = self.inputs
      return (value, *inputs[0][1:]), inputs[1]
  ```
  Note this triggers a *recursive* `__get__` on `self.inputs` to read the rest of the args/kwargs that the user *didn't* override.

### transform — closing the mutation loop

`preprocess` returns a new object — a clone, a view, or a reshape. In-place mutations the user makes to that object are **invisible to the running model** because the model still holds the original tensor. `transform` closes that loop.

`@<eprop>.transform` registers a function called *after* `preprocess` returns to the user but *before* the model's forward proceeds. Signature: `() -> Any` (no arguments — the preprocessed value is bound into the closure via `functools.partial` at request time, see `interleaver.py:298`).

The mediator stores the partial-bound transform on `mediator.transform`. After the worker is responded to with the preprocessed value (in `handle_value_event`), the main thread immediately invokes `mediator.transform()` and `batcher.swap`s the result back into the model. The transform runs synchronously — the worker thread gets back the (now-mutated) preprocessed value, the model gets the swapped-back transform return value, and execution continues.

`mediator.transform` is **one-shot**. After firing, it's cleared. The next eproperty access that registers a transform will set it again.

### Worked example: per-head attention access

Suppose you want `model.layer.heads.output` to return attention activations as `[B, n_heads, S, head_dim]`, let users edit them, and have edits flow back into the model:

```python
class MyEnvoy(Envoy):
    n_heads = 12

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        B, S, H = value.shape
        return value.view(B, S, self.n_heads, H // self.n_heads).transpose(1, 2)

    @heads.transform
    @staticmethod
    def heads(value):
        # value is the preprocessed view; mutations from user code are visible here
        # (closure captures it). Reshape back to model's expected layout.
        return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)
```

Flow when a user writes `model.layer.heads.output[:, 5] = 0`:

1. `__get__` fires `requires_output` to install an output hook.
2. Worker blocks; model runs; hook fires; `mediator.handle` matches and `respond`s with the raw `[B, S, H]` value.
3. `preprocess` reshapes to `[B, n_heads, S, head_dim]`.
4. `transform` is bound to that reshaped view via `partial`.
5. Worker receives the reshaped value; mutates it (`[:, 5] = 0`).
6. Worker yields control (e.g. on the next eproperty access). Main thread fires `mediator.transform()`.
7. `transform` reshapes back; `batcher.swap` updates the live model value with the user's edits.

### provide — runtime-side pushes

`eproperty.provide(obj, value)` (`src/nnsight/intervention/interleaver.py:328`) is for values that don't come from a PyTorch hook. The runtime (e.g. vLLM, a streaming generator, the `InterleavingTracer.result` after a forward pass) pushes a value into the interleaver:

```python
def provide(self, obj, value):
    requester = self._build_requester(obj)
    return obj.interleaver.handle(requester, value, iterate=self.iterate)
```

Internally it calls `Interleaver.handle` (see `docs/developing/interleaver-internals.md`). When `iterate=True`, the per-mediator iteration tracker for this provider path is bumped after each mediator handles the value — the analog of what `register_iter_hooks` does for module-level forward passes.

This is the third way a value reaches a worker thread:

1. **Module forward hook** (`requires_output` / `requires_input`) — `eproperty.__get__` triggers a hook install; the model firing the hook delivers the value.
2. **Operation hook** (`requires_operation_*`) — same, but on `OperationAccessor` lists.
3. **Provider push** (`provide`) — the runtime calls `eproperty.provide(envoy, value)` directly, fanning out to all mediators.

For (3), the eproperty stub is **bare** — no `requires_*` decorator. Example:

```python
@eproperty(iterate=False)
def result(self) -> Object:
    """The return value of the method being traced."""
```

(`tracing/tracer.py:581` — `InterleavingTracer.result`.)

This is fed by `Envoy.interleave` (`envoy.py:605`):

```python
self.interleaver.handle("result", result)
```

Note this is the bare interleaver call rather than `eproperty.provide` — because `result` is an eproperty on `InterleavingTracer` but is fed by `Envoy.interleave`, the call site already knows the requester is `"result"` and skips the descriptor indirection. Either path works.

### description and repr

`description` is a short label shown in `Envoy.__repr__`. From `envoy.py:942`:

```python
for cls in type(self).__mro__:
    for attr_name, attr_val in cls.__dict__.items():
        if isinstance(attr_val, eproperty) and attr_val.description is not None:
            eproperty_lines.append("(" + attr_val.name + "): " + attr_val.description)
```

So `description="Module output"` would show as `(output): Module output` in `print(model)`.

Eproperties without a description are hidden — useful for internal-only properties that aren't meant to be browsed.

### iterate=False — when not to suffix

The default `iterate=True` is right for any value that fires per-forward-pass. Set `iterate=False` for:

- **Whole-trace results** — `InterleavingTracer.result`. The trace produces one result no matter how many forward passes.
- **Static metadata** — anything provider-pushed once at trace setup.
- **External streams** — e.g. an async runtime that delivers tokens out of band.

When `iterate=False`, the requester is just `path.key` with no `.iN` suffix. The `provide` call must match (also pass `iterate=False`). The persistent iter-tracker hooks won't bump this provider; you bump it yourself if you need to.

### Multiple eproperties sharing a key

`Envoy.input` and `Envoy.inputs` both use `key="input"`:

```python
@eproperty(key="input")
@requires_input
def inputs(self) -> Tuple[Tuple[Object], Dict[str, Object]]:
    """Get the inputs to the module's forward pass."""

@eproperty(key="input")
@requires_input
def input(self) -> Object:
    """Get the first input to the module's forward pass."""

@input.preprocess
def input(self, value):
    return [*value[0], *value[1].values()][0]
```

Why: a single PyTorch pre-hook fires once per call, delivering `(args, kwargs)` once. Both descriptors share the requester `"<path>.input.iN"`. When the worker accesses `.inputs`, it gets the raw `(args, kwargs)`; when it accesses `.input`, it gets the preprocessed first arg. Both descriptors trigger `requires_input`, but `requires_input` skips re-installation if `batcher.current_provider` already matches the requester (the hook is already mid-fire). Net result: one hook, two views.

The same pattern is used in `OperationEnvoy` (`source.py:613`-`630`).

### Key resolution and the path attribute

`_build_requester` (`interleaver.py:260`):

```python
def _build_requester(self, obj):
    path = getattr(obj, "path", "")
    return f"{path}.{self.key}" if path else self.key
```

If the host has no `path` (or path is empty), the requester is just `key`. This matters for tracer-level eproperties like `InterleavingTracer.result` — `result` is the full requester with no module prefix.

### eproperty inherits from property in spirit, not in code

The class is a freestanding descriptor — it does *not* inherit from Python's built-in `property`. (Earlier versions did, see commit `b16637d`; the inheritance was reverted in `823961e` to allow `Generic[T]` typing without breaking IDE hints.) The implication: if you need `del obj.<eproperty>`, you'd need to add a `__delete__`. Currently no eproperty supports deletion — it would not have well-defined semantics in an interleaving context.

## Key files / classes

- `src/nnsight/intervention/interleaver.py:42` — `IEnvoy` protocol.
- `src/nnsight/intervention/interleaver.py:60` — `eproperty` descriptor.
- `src/nnsight/intervention/interleaver.py:144` — `eproperty.__init__`.
- `src/nnsight/intervention/interleaver.py:157` — `eproperty.__call__` (registers the stub).
- `src/nnsight/intervention/interleaver.py:174` — `eproperty.postprocess`.
- `src/nnsight/intervention/interleaver.py:184` — `eproperty.preprocess`.
- `src/nnsight/intervention/interleaver.py:198` — `eproperty.transform`.
- `src/nnsight/intervention/interleaver.py:260` — `_build_requester`.
- `src/nnsight/intervention/interleaver.py:264` — `eproperty.__get__`.
- `src/nnsight/intervention/interleaver.py:306` — `eproperty.__set__`.
- `src/nnsight/intervention/interleaver.py:328` — `eproperty.provide`.
- `src/nnsight/intervention/hooks.py:271` — `requires_output`. The most common pre-setup decorator.
- `src/nnsight/intervention/hooks.py:314` — `requires_input`.
- `src/nnsight/intervention/hooks.py:438` — `requires_operation_output`.
- `src/nnsight/intervention/hooks.py:464` — `requires_operation_input`.
- `src/nnsight/intervention/envoy.py:168` — `Envoy.output` (canonical eproperty example).
- `src/nnsight/intervention/envoy.py:178` — `Envoy.inputs` (shared-key example).
- `src/nnsight/intervention/envoy.py:191` — `Envoy.input` with `preprocess` and `postprocess`.
- `src/nnsight/intervention/source.py:603` — `OperationEnvoy.output` / `inputs` / `input`.
- `src/nnsight/intervention/tracing/tracer.py:581` — `InterleavingTracer.result` (bare eproperty, no decorator).

## Lifecycle / sequence

For `model.layer.output[:, -1] = 0` inside a trace:

1. `model.layer.output[:, -1] = 0` evaluates to `model.layer.output.__setitem__((slice(None), -1), 0)` — but `model.layer.output` is an eproperty `__get__` *first*, returning the value, then `__setitem__` is called on that value. So this is in-place mutation, not eproperty `__set__`.
2. `__get__` runs `requires_output` to install a one-shot output hook.
3. `mediator.request` blocks the worker.
4. Model fires; hook fires; `mediator.handle` responds with the output tensor.
5. Worker resumes with the actual tensor; `[:, -1] = 0` mutates it in place.
6. The mutation is visible to the model because the tensor is the same object — the hook returned it unchanged. Cool.

For `model.layer.output = some_tensor` (replacement, not mutation):

1. `__set__` runs.
2. `postprocess` is None (no postprocess on `Envoy.output`), so value is unchanged.
3. `requires_output` runs to install a hook.
4. `mediator.swap("model.layer.output.i0", some_tensor)` enqueues a SWAP event.
5. Model fires; hook fires; `mediator.handle_swap_event` matches; `batcher.swap` overwrites the value.
6. Hook returns the swapped value to PyTorch, which uses it as the module's output.

## Extension points

- **A new property on `Envoy`.** Subclass `Envoy` and add an `@eproperty()` with `@requires_output` or `@requires_input` (or write a custom `requires_*` decorator that registers a different kind of hook). Optionally add `preprocess`, `postprocess`, `transform`, `description`.
- **A new property on a custom IEnvoy host.** Implement the protocol (`interleaver`, `path`) and add eproperties. The vLLM `VLLM` class (`src/nnsight/modeling/vllm/vllm.py`) does this for `logits` and `samples`, which are runtime-side pushes via `eproperty.provide`.
- **Custom transform semantics.** `transform` is currently one-shot (cleared after firing). If you wanted persistent transforms (e.g. always reshape every output) you would not use `transform`; you would build a custom Envoy method. The `transform` mechanism is specifically for "I gave you a view, now please reshape my mutations back."
- **Bare eproperty for runtime values.** No `requires_*` decorator. The runtime is responsible for calling `eproperty.provide(envoy, value)` (or directly calling `interleaver.handle(requester, value)`) at the right moment.
- **Sharing a key.** When two views must back the same hook, give them the same `key=`. Both `requires_*` invocations will short-circuit if the hook is already installed.

## Related

- `docs/developing/lazy-hook-system.md` — how `requires_output` / `requires_input` install hooks.
- `docs/developing/interleaver-internals.md` — what `mediator.request` and `mediator.swap` do.
- `docs/developing/source-accessor-internals.md` — `requires_operation_*` for op-level eproperties.
- `docs/developing/architecture-overview.md` — where eproperty fits in the layer stack.
- `NNsight.md` Section 4.2 — the original "accessing values" narrative; this doc supersedes it for the descriptor protocol.
