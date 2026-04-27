---
title: Envoy and eproperty
one_liner: Envoy wraps a torch.nn.Module and exposes hookable properties via the eproperty descriptor; preprocess / postprocess / transform extend the interception API.
tags: [concept, mental-model, envoy]
related: [docs/concepts/interleaver-and-hooks.md, docs/concepts/source-tracing.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/envoy.py:54, src/nnsight/intervention/envoy.py:168, src/nnsight/intervention/interleaver.py:42, src/nnsight/intervention/interleaver.py:60, src/nnsight/intervention/interleaver.py:264, src/nnsight/intervention/hooks.py:271]
---

# Envoy and eproperty

## What this is for

`Envoy` (`envoy.py:54`) is the user-facing wrapper around a `torch.nn.Module`. It exposes the module's children as nested Envoys and provides hookable attributes:

- `.output` — the module's forward return value
- `.input` — the first positional (or first kwarg) input
- `.inputs` — the full `(args, kwargs)` tuple
- `.skip(value)` — bypass this module's forward and return `value`
- `.next(step)` — advance to the next generation step
- `.source` — operation-level access (see [Source Tracing](source-tracing.md))

These properties are backed by the `eproperty` descriptor (`interleaver.py:60`). Each `eproperty` access blocks the worker thread on a value request and routes through the appropriate one-shot hook.

This is the public API surface for extending nnsight with custom interception points.

## When to use / when not to use

- Use `Envoy` directly if you're wrapping a non-LM PyTorch model: `nnsight.NNsight(my_pytorch_model)`.
- Subclass `Envoy` to add new eproperties when you need a new hookable concept (e.g. attention heads as a separate dim, vLLM `logits` / `samples`).
- Don't reach into `Envoy._module` to manipulate hooks directly — use eproperties so cleanup, ordering, and iteration tracking stay correct.

## Canonical pattern

```python
import nnsight

model = nnsight.LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("Hello"):
    # Read
    out = model.transformer.h[0].output[0].save()
    args, kwargs = model.transformer.h[0].inputs

    # Write (in-place)
    model.transformer.h[0].output[0][:] = 0

    # Skip
    model.transformer.h[1].skip((model.transformer.h[1].input, None))
```

## IEnvoy: the protocol

Any object that hosts `eproperty` descriptors must satisfy the `IEnvoy` protocol (`interleaver.py:42`):

```python
class IEnvoy(Protocol):
    interleaver: "Interleaver"
    path: Optional[str]  # absent / empty for tracer-level eproperties
```

`path` is the provider-path prefix used to build the requester string. `eproperty._build_requester` (`interleaver.py:260`) uses it like this:

- If the IEnvoy has no `path` attribute, or `path` is `None` / empty, the requester is just the eproperty key (e.g. `"result"`).
- Otherwise the requester is `f"{path}.{key}"` (e.g. `"model.transformer.h.0.output"`).

Concretely: `Envoy.path` is the dotted module path; `InterleavingTracer` has no `path`, so `tracer.result` resolves to the bare key `"result"`.

Implementors in this codebase:

- `Envoy` — module-level (`.output`, `.input`, `.inputs`, `.skip`)
- `OperationEnvoy` — operation-level for `.source` (`.output`, `.input`, `.inputs`)
- `InterleavingTracer` — tracer-level (`.result`, no path)
- vLLM `VLLM` — `.logits`, `.samples`

## eproperty: the descriptor

`eproperty` (`interleaver.py:60`) is decorated onto stub methods. The stub body is **never run for its return value** — its purpose is to carry a stack of decorators that perform pre-setup work (registering a hook) before the descriptor blocks on a value request.

```python
@eproperty()           # outer decorator: registers as eproperty
@requires_output       # inner decorator: registers a one-shot output hook on access
def output(self) -> Object:
    """Get the output of the module's forward pass."""
```

What happens on `module.output`:

1. `eproperty.__get__` is invoked (`interleaver.py:264`).
2. It calls `self._hook(obj)` — runs the stub. The decorator side effect registers a one-shot PyTorch hook on `obj._module` (or skips if a value is already being provided).
3. It builds the requester string from `obj.path + "." + key` and appends an iteration suffix via `interleaver.iterate_requester`.
4. It calls `interleaver.current.request(requester)` — the worker thread blocks here.
5. When the value arrives, optional `_preprocess(obj, value)` runs.
6. If `_transform` is set, the preprocessed value is bound into a `partial` and parked on the mediator. After the worker yields control, the mediator invokes the transform and swaps the result back into the model.

## Anatomy: input vs inputs

Both `Envoy.input` and `Envoy.inputs` (`envoy.py:178` and `envoy.py:191`) share `key="input"` — they hook the same provider path but expose different views:

```python
@eproperty(key="input")
@requires_input
def inputs(self) -> Tuple[Tuple, Dict]:
    """Returns (args, kwargs)."""

@eproperty(key="input")
@requires_input
def input(self):
    """Returns the first positional or first kwarg."""

@input.preprocess
def input(self, value):
    return [*value[0], *value[1].values()][0]   # extract first

@input.postprocess
def input(self, value):                         # repack on assignment
    inputs = self.inputs
    return (value, *inputs[0][1:]), inputs[1]
```

- `preprocess` runs on `__get__`, *after* the value comes back from the hook, *before* it is returned to the user.
- `postprocess` runs on `__set__`, on the user's assigned value, *before* it is sent to `interleaver.current.swap(...)`.

When the user writes `module.input = x`, `postprocess` repacks `x` back into a `(args, kwargs)` tuple and the swap replaces the full inputs.

## transform: closing the loop on mutable views

`preprocess` returning a *new* object (clone, reshape, view onto a slice) breaks aliasing — the model still holds the original. `transform` restores the loop:

```python
class MyEnvoy(Envoy):
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
        # Reshape back to [B, S, H] so the model continues with the user's edits.
        return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)
```

Mechanics (`interleaver.py:298`):

1. On access, `preprocess` runs and returns the reshaped view.
2. The transform is bound to that view via `functools.partial` and stored on `mediator.transform`.
3. The worker receives the reshaped view; in-place edits are visible inside the partial's closure.
4. After the worker yields (next event), `Mediator.handle_value_event` calls `self.transform()` and swaps the return value back into the model via `batcher.swap`.
5. `mediator.transform` is cleared — transforms are one-shot per access.

## provide: model-side push into the interleaver

For values that don't come from a PyTorch hook (vLLM logits, generation results, etc.), use `eproperty.provide(envoy, value)` (`interleaver.py:328`). This calls `interleaver.handle(...)` which fans the value out to every mediator and bumps the iteration counter for that path. See `Envoy.interleave` (`envoy.py:590`) which uses this for `result`.

## A bare eproperty (no hook decorator)

An `eproperty` without a `requires_*` decorator is valid for values that are pushed externally rather than pulled from a hook. Example: `InterleavingTracer.result` (`tracing/tracer.py:581`) — fed by `Envoy.interleave` calling `interleaver.handle("result", result)`.

## Calling modules in a trace

`Envoy.__call__(*args, hook=False, **kwargs)` (`envoy.py:239`) lets you invoke a module from inside a trace as an "ad hoc" computation. The `hook` flag controls whether interleaving hooks fire:

```python
def __call__(self, *args, hook: bool = False, **kwargs):
    return (
        self._module.forward(*args, **kwargs)
        if self.interleaver.current is not None and not hook
        else self._module(*args, **kwargs)
    )
```

- **Default (`hook=False`) inside an active trace:** routes through `module.forward(...)`, bypassing the wrapped `__call__` and therefore **not** firing the `.output` / `.input` hooks. The call runs as a one-off computation that doesn't show up in the interleaver's request stream.
- **`hook=True`:** routes through `module(...)`, firing all registered hooks just like a normal forward pass.
- **Outside a trace** (`interleaver.current is None`): always routes through `module(...)`.

The canonical use is logit-lens-style decoding, where you want to run `lm_head` on intermediate hidden states without interception:

```python
with model.trace("The Eiffel Tower is in"):
    for i in range(12):
        hs = model.transformer.h[i].output[0]
        # Call lm_head and ln_f directly — no .output / .input hooks fire.
        logits = model.lm_head(model.transformer.ln_f(hs))
        tokens = logits.argmax(dim=-1).save()
```

Pass `hook=True` when you need the auxiliary call to participate in interception — e.g. caching, source tracing, or feeding through another envoy's eproperty:

```python
# Apply an auxiliary SAE module and access its hooked .output afterwards.
with model.trace("Hello"):
    hidden = model.transformer.h[5].output[0]
    reconstructed = model.sae(hidden, hook=True)   # hooks fire on model.sae
    model.transformer.h[5].output[0][:] = reconstructed
```

## Custom Envoy classes

Wire custom Envoy classes into the model via the `envoys=` parameter:

```python
model = nnsight.LanguageModel(
    "gpt2",
    envoys={torch.nn.Linear: MyLinearEnvoy, "self_attn": MyAttnEnvoy},
)
```

- Type keys match against `type(module).__mro__`.
- String keys match a dotted suffix of the path (alias-aware via `rename`).
- Type keys take precedence; falls back to base `Envoy`. See `Envoy._resolve_envoy_class` (`envoy.py:615`).

## Module renaming

`rename={...}` on `LanguageModel` / `Envoy` creates aliases. Single-component renames (e.g. `{"self_attn": "attn"}`) propagate to type-key matching; multi-component renames (e.g. `{".transformer.h": ".layers"}`) mount the target path under a new alias.

## Gotchas

- **eproperty `__get__` raises outside interleaving.** `model.transformer.h[0].output` outside a trace gives `ValueError: Cannot access ...`. Wrap in `with model.trace(...)`.
- **Don't mutate `Envoy._module` directly.** Reassign through `setattr` so child Envoys are reconstructed and the source accessor cache is preserved.
- **`Envoy.__call__` defaults to `hook=False` inside a trace.** It routes through `module.forward(...)` and skips interception — see [Calling modules in a trace](#calling-modules-in-a-trace).
- **`transform` is one-shot per access.** Re-accessing the same eproperty re-binds a fresh transform. Don't rely on the previous transform persisting.
- **`@input.preprocess` / `@input.postprocess` reuse the property name.** It's a mini-DSL on the descriptor — Python sees three `def input` blocks but only the first one (the `@eproperty(...)` call) is the descriptor; the others register hooks on it.

## Related

- [Interleaver and Hooks](interleaver-and-hooks.md) — what `requires_output`, `requires_input` actually register.
- [Source Tracing](source-tracing.md) — `Envoy.source` and the per-Envoy `OperationEnvoy` / `SourceEnvoy` wrappers.
- [Threading and Mediators](threading-and-mediators.md) — what blocks on the request and what fulfills it.
- Source: `src/nnsight/intervention/envoy.py` (`Envoy`), `src/nnsight/intervention/interleaver.py` (`eproperty`, `IEnvoy`).
