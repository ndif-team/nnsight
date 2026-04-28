---
title: Extending nnsight
one_liner: Wrap modules in custom Envoy subclasses (envoys=) and define your own hookable values via eproperty (preprocess / postprocess / transform).
tags: [usage, extending, envoy, eproperty, library-development]
related: [docs/usage/access-and-modify.md, docs/usage/source.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/envoy.py:74, src/nnsight/intervention/envoy.py:615, src/nnsight/intervention/interleaver.py:60, src/nnsight/intervention/interleaver.py:198, src/nnsight/intervention/hooks.py:271, src/nnsight/modeling/base.py:77, src/nnsight/modeling/vllm/vllm.py:102]
---

# Extending nnsight

## What this is for

Two extension points let you teach nnsight new tricks without forking the library:

1. **`envoys=`** on `NNsight` / `Envoy` — controls which `Envoy` *class* wraps each descendant module. Lets you attach module-type-specific hookable values (`MyAttnEnvoy.heads.output`, `MyLinearEnvoy.weight_proxy`, …) without touching the model itself.
2. **`eproperty`** — the descriptor that backs every `.output` / `.input` / `.inputs` / `.logits` / `.samples` access. Defining your own eproperty gives you a new value-channel that participates in the request/swap mechanism. With `preprocess`, `postprocess`, and the newer `transform` hook, you can present the value in a different shape (per-head reshape, slice view, denormalized version) and swap user-edits back into the model.

Both are stable public APIs. If you are integrating a new runtime (vLLM-style) or building an analysis library on top of nnsight, this is the file you are looking for.

## When to use / when not to use

- Use `envoys=` when you want the same behavior to attach to every Linear / every attention module / etc. across a whole model.
- Use `eproperty` when you have a value the user should be able to read and write that is **not** already a module's `.input` / `.output` — e.g. per-head attention slices, sampler logits, KV cache slabs.
- Skip both for one-off interventions inside a single trace — `.source` and direct `.output` slicing are simpler.

---

## Part 1 — Custom `Envoy` subclasses

### The `envoys=` argument

`Envoy.__init__` accepts an `envoys=` kwarg (`src/nnsight/intervention/envoy.py:80`):

```python
envoys: None | Type[Envoy] | Dict[Union[Type[torch.nn.Module], str], Type[Envoy]]
```

Three forms:

| Form                                        | Behavior                                                                     |
|---------------------------------------------|------------------------------------------------------------------------------|
| `None` (default)                            | every descendant wrapped in the base `Envoy`.                                |
| A single `Envoy` subclass                   | every descendant wrapped in that class.                                      |
| `Dict[type \| str, Type[Envoy]]`            | per-module mapping. Type keys win over string keys.                          |

Type-key lookup walks the module's MRO, so `{torch.nn.Linear: MyLinearEnvoy}` matches every concrete `Linear` subclass.

String-key lookup matches the descendant's envoy path as a **dotted suffix** (component-wise, not substring): `{"transformer.h.0.attn": MyAttnEnvoy}` matches the path ending in `transformer.h.0.attn`. With a `rename={...}` in play, single-component aliases also count — `{"attn": MyAttnEnvoy}` matches `self_attn` if you passed `rename={"self_attn": "attn"}` (`src/nnsight/intervention/envoy.py:654`).

The mapping **propagates down the tree** — each child Envoy is constructed with the same `envoys=` value (`src/nnsight/intervention/envoy.py:718`). One declaration covers the whole model.

### Pattern: attach an Envoy subclass to every Linear

```python
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import eproperty
from nnsight.intervention.hooks import requires_output
import torch

class MyLinearEnvoy(Envoy):
    @eproperty(key="output")
    @requires_output
    def normalized(self): ...

    @normalized.preprocess
    def normalized(self, value):
        return value / value.norm(dim=-1, keepdim=True)

model = NNsight(my_torch_model, envoys={torch.nn.Linear: MyLinearEnvoy})

with model.trace(x):
    n = model.layer1.normalized.save()       # MyLinearEnvoy.normalized fires
```

### Pattern: subclass-level default

If you ship a model wrapper class (like `LanguageModel` or `VLLM`), set `envoys` as a class attribute on your `NNsight` subclass and users don't have to pass anything:

```python
class MyModel(NNsight):
    envoys = {torch.nn.Linear: MyLinearEnvoy, "self_attn": MyAttnEnvoy}

# Per-instance override is still possible:
m = MyModel(net, envoys=None)              # opt out of the default
m = MyModel(net, envoys={torch.nn.Conv2d: MyConvEnvoy})  # full override
```

The constructor `kwargs.setdefault("envoys", type(self).envoys)` (`src/nnsight/modeling/base.py:82`) gives the class default while letting users override per-instance.

### Resolution rules (in order)

`Envoy._resolve_envoy_class` (`src/nnsight/intervention/envoy.py:615`) tries:

1. If `envoys is None` → base `Envoy`.
2. If `envoys` is a class → that class for every descendant.
3. Walk `type(module).__mro__` and look for any class key that matches.
4. Fall back to string-key suffix match (with rename-alias-aware components).
5. No match → base `Envoy`.

Type keys always beat string keys.

---

## Part 2 — Custom `eproperty`

`eproperty` is a descriptor for IEnvoy classes (`src/nnsight/intervention/interleaver.py:60`). It exposes a value through the interleaving request/swap protocol — reading blocks until a hook delivers it; writing schedules a swap.

### IEnvoy contract

Any class that hosts an `eproperty` must satisfy the `IEnvoy` protocol (`src/nnsight/intervention/interleaver.py:43`):

```python
class IEnvoy(Protocol):
    interleaver: "Interleaver"
    path: str
```

That's the whole interface — an attribute pointing at the live `Interleaver` and a path prefix. `Envoy`, `OperationEnvoy`, `InterleavingTracer`, and `VLLM` all satisfy it.

### Defining an eproperty

```python
from nnsight.intervention.interleaver import eproperty
from nnsight.intervention.hooks import requires_output, requires_input

class MyEnvoy(Envoy):
    @eproperty()
    @requires_output
    def output(self): ...           # body is a no-op stub
```

The stub's body is **never executed for its return value**. Two things matter:

- The **decorators stacked on top** — these do the real work of registering the hook that will produce the value.
- The function's `__name__` — used as the default `key`, and the docstring shows up in `help()`.

`__get__` calls `self._hook(obj)` to fire those decorators, then issues `interleaver.current.request(requester)` and blocks until a hook calls `mediator.handle(provider, value)` with a matching key.

### The setup decorators (hooks.py)

Pre-setup decorators live in `src/nnsight/intervention/hooks.py`. The contract: "make sure a provider for this requester string will fire before `request()` blocks." Built-in options:

| Decorator                       | Hook installed                                              |
|---------------------------------|-------------------------------------------------------------|
| `requires_output`               | One-shot forward hook on `self._module`                     |
| `requires_input`                | One-shot forward pre-hook on `self._module`                 |
| `requires_operation_output`     | Operation-level post-hook (used by `OperationEnvoy`)        |
| `requires_operation_input`      | Operation-level pre-hook (used by `OperationEnvoy`)         |

A bare `@eproperty()` with no setup decorator is also valid for values you provide externally via `eproperty.provide(obj, value)` (`src/nnsight/intervention/interleaver.py:328`). That's how `InterleavingTracer.result` and `VLLM.logits` work — they are pushed into the system from outside the model rather than caught by a hook on a `nn.Module`.

Custom backends supply their own decorators in the same pattern. To integrate a new runtime, write a hook function that delivers the value via `mediator.handle(requester, value)` and a decorator that registers it before the request blocks.

### `eproperty(...)` constructor args

```python
eproperty(key=None, description=None, iterate=True)
```

| Arg          | Meaning                                                                                     |
|--------------|---------------------------------------------------------------------------------------------|
| `key`        | The interleaving key appended to `obj.path` (`<path>.<key>`). Defaults to the stub's name. |
| `description`| Short label shown in the repr tree. Only eproperties with a description appear in the tree.|
| `iterate`    | Whether to append `.i0` / `.i1` / … iteration suffix. Default `True`.                       |

Multiple eproperties can share a key — `Envoy.input` and `Envoy.inputs` both use `"input"` to provide different views on the same underlying value (`src/nnsight/intervention/envoy.py:178`).

### `preprocess`, `postprocess`, `transform`

Three optional decorators reshape the value as it flows through the descriptor.

| Decorator       | Fires on    | Receives          | Purpose                                              |
|-----------------|-------------|-------------------|------------------------------------------------------|
| `@x.preprocess` | `__get__`   | `(self, value)`   | Reshape the value before the user sees it.           |
| `@x.postprocess`| `__set__`   | `(self, value)`   | Reshape user-supplied value before swapping into the model. |
| `@x.transform`  | After value delivery, before next event | `()` (preprocessed value via closure) | Reshape back after user edits and swap into the model. |

**`preprocess` / `postprocess`** are the simple pair. The canonical `Envoy.input` example (`src/nnsight/intervention/envoy.py:205`):

```python
@input.preprocess
def input(self, value):
    return [*value[0], *value[1].values()][0]   # extract first arg from (args, kwargs)

@input.postprocess
def input(self, value):
    inputs = self.inputs                         # repack into (args, kwargs)
    return (value, *inputs[0][1:]), inputs[1]
```

**`transform`** (`src/nnsight/intervention/interleaver.py:198`) closes the loop when `preprocess` returns a *new* object (clone, view, reshape). In-place edits the user makes to the new object are invisible to the model — the model still holds the original. `transform` solves this:

1. At request time, `preprocess` produces value V.
2. `eproperty.__get__` binds V into the transform via `functools.partial` and parks it on `mediator.transform`.
3. The user receives V and edits it in place.
4. When the worker thread yields control back to the mediator, `Mediator.handle_value_event` fires `mediator.transform()` and `batcher.swap`s the result back into the model (`src/nnsight/intervention/interleaver.py:1039`).

It is one-shot per access — fired and cleared.

### Pattern: per-head attention view

```python
class MyAttnEnvoy(Envoy):
    n_heads = 12

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        # Expose attention heads as a separate dim for the user.
        B, S, H = value.shape
        return value.view(B, S, self.n_heads, H // self.n_heads).transpose(1, 2)

    @heads.transform
    @staticmethod
    def heads(value):
        # Reshape back to [B, S, H] so the model continues with the user-edited heads.
        return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)
```

```python
with model.trace("Hello"):
    h = model.transformer.h[0].attn.heads     # [B, n_heads, S, head_dim]
    h[:, 4] = 0                                # ablate head 4
# transform reshapes back to [B, S, H] and swaps it into the model.
```

### Pattern: safe mutable view

A common idiom is `preprocess` returning `value.clone()` so users can do `thing[:] = 0` without aliasing surprises, with `transform` returning the (mutated) clone so the model still sees their edits. Without `transform`, the clone would be discarded and the model would proceed with the original.

### Pattern: external value channel (no module hook)

For values the model produces outside any single `nn.Module` — e.g. final logits, sampled tokens — define a bare eproperty and `provide` it from your runtime:

```python
class VLLM(NNsight):
    @eproperty(description="Logits", iterate=True)
    def logits(self): ...      # no setup decorator

    @eproperty(description="Sampled token ids", iterate=True)
    def samples(self): ...
```

```python
# In the runtime's forward path:
type(self).logits.provide(self, logits_tensor)
type(self).samples.provide(self, sampled_token_ids)
```

`provide` calls `interleaver.handle(requester, value, iterate=self.iterate)` (`src/nnsight/intervention/interleaver.py:328`). This is exactly how vLLM's `WrapperModule`s work — see `src/nnsight/modeling/vllm/vllm.py:102` for the full pattern.

---

## Worked example: an SAE Envoy

Putting it together — a custom Envoy subclass that exposes the latent activations of an attached Sparse Autoencoder as a hookable value:

```python
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.interleaver import eproperty
from nnsight.intervention.hooks import requires_output

class SAEResidualEnvoy(Envoy):
    """Wraps a residual-stream module and exposes SAE latent activations."""

    sae = None    # set by user after construction

    @eproperty(key="output")
    @requires_output
    def latents(self): ...

    @latents.preprocess
    def latents(self, value):
        # value is the residual-stream output: tuple(hidden_states, ...)
        return self.sae.encode(value[0])

    @latents.transform
    def latents(self, latents):
        # Reconstruct from (possibly edited) latents and rebuild the tuple.
        # `self` is bound by the closure — value carries the rest of the tuple
        # via the captured original input on `self`.
        return (self.sae.decode(latents),) + self._last_output[1:]

model = NNsight(my_model, envoys={"transformer.h.5": SAEResidualEnvoy})
model.transformer.h[5].sae = my_sae

with model.trace("Hello"):
    z = model.transformer.h[5].latents
    z[:, :, dead_features] = 0           # zero out dead features
    out = model.lm_head.output.save()
```

(In real code you would also need to capture the rest of the residual tuple — this elides that for brevity.)

---

## Gotchas

- **`envoys=` is propagated, not inherited.** Each child Envoy is constructed with the SAME `envoys=` dict you passed to the root. Mutating it after construction does not affect already-built Envoys.
- **String keys in `envoys=` are dotted suffixes, not substrings.** `{"attn": MyEnvoy}` matches `transformer.h.0.attn` but not `transformer.h.0.attn_layernorm` (`src/nnsight/intervention/envoy.py:654`).
- **Type keys win over string keys.** If both `torch.nn.Linear` and `"lm_head"` would match `model.lm_head`, the type key wins.
- **Stub bodies are no-ops.** Whatever you put in the body of the eproperty stub is never run for its return value. The decorators do the work.
- **`preprocess` returning a new object without `transform` means user edits are lost.** The model still holds the original. Either avoid making a new object in `preprocess`, or pair it with a `transform`.
- **`transform` is one-shot per access.** It fires once when the value event for that access is processed, then `mediator.transform = None`. If you read the same eproperty twice, you get two fires.
- **Decorator stacking order matters.** `@eproperty()` must be on the OUTSIDE; `@requires_output` (or your custom setup decorator) is on the INSIDE. The inside decorator wraps the stub; the eproperty descriptor wraps the result.
- **Don't share an `Envoy` subclass across multiple models without thinking.** Class-level state on the Envoy subclass is shared across instances. If you need per-instance config, set it on `__init__` on the subclass — or wire it via a module attribute the eproperty reads.

## Related

- [access-and-modify](access-and-modify.md) — The built-in `.output` / `.input` eproperties.
- [source](source.md) — `OperationEnvoy` is also an `IEnvoy`; its `.output` / `.input` are eproperties on operation accessors.
- [docs/concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) — Architecture deep dive.
- [NNsight.md](../../NNsight.md) — Full internal architecture.
- [nnsight.net](https://www.nnsight.net) — Tutorials and API reference.

## Reference commits

- `cff5d2b` — `eproperty.transform` documentation, fixes swapped preprocess/postprocess docstrings, transform tests.
- `eb9829b` — Adds `envoys=` mapping (None | class | dict) propagated through the tree; class-attribute default on `NNsight`.
- `baff345` — Adds string-key support to `envoys=` (dotted-suffix matching with rename-alias awareness).
