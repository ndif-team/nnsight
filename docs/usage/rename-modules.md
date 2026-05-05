---
title: Rename Modules
one_liner: Alias module paths via `rename={...}` on model construction; supports list-of-aliases and re-mounting subtrees.
tags: [usage, models, rename, aliases]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/envoy.py:74, src/nnsight/intervention/envoy.py:1058, src/nnsight/modeling/mixins/meta.py:38, src/nnsight/modeling/base.py:81]
---

# Rename Modules

## What this is for

Different model architectures use different attribute names for the same role (`transformer.h` vs `model.layers` vs `gpt_neox.layers`). The `rename={...}` constructor kwarg installs aliases so your intervention code is portable across model families.

Aliases are bidirectional in tooling (the repr shows them, `Cache` keys can be looked up by alias) and they participate in the `envoys=` class-mapping system so e.g. `{"attn": MyAttnEnvoy}` matches paths ending in `self_attn` when you've renamed `self_attn` to `attn`.

## When to use / when not to use

- Use when writing analysis code that should work across multiple HuggingFace architectures.
- Use to mount a deep subtree at a shorter path (`{".model.layers": ".layers"}` makes `model.layers[0]` work directly on the root).
- Use when the underlying module name conflicts with an nnsight reserved name like `input` / `output` (although nnsight handles those automatically by remounting to `.nns_input` / `.nns_output`, see `Envoy._handle_overloaded_mount`).

## Canonical pattern

```python
from nnsight import LanguageModel

model = LanguageModel(
    "openai-community/gpt2",
    rename={
        "transformer.h": "layers",        # mount at new path
        "mlp": "feedforward",             # rename every MLP child
        ".transformer": ["model", "mdl"], # multiple aliases for one path
    },
)

with model.trace("Hello"):
    a = model.layers[0].feedforward.output.save()    # via alias
    b = model.transformer.h[0].mlp.output.save()     # original still works
    c = model.model.layers[0].output.save()          # via .transformer alias
    d = model.mdl.layers[0].output.save()            # via second alias
```

## Forms of `rename` keys and values

`rename` is `Dict[str, Union[str, List[str]]]`. The semantics depend on the **key** shape:

- **Single component key** (`"mlp"`): renames every descendant whose attribute name matches. Applied component-wise.
- **Dotted key** (`".transformer.h"` or `"transformer.h"`): treated as a path from the root envoy. Mounts the subtree at the alias name on the root. The leading dot is optional but conventional.
- **Single value** (`"layers"`): one alias.
- **List of strings** (`["model", "mdl"]`): multiple aliases for the same path.

Source: `Aliaser` (`src/nnsight/intervention/envoy.py:1058`).

## How alias resolution works

`Aliaser.build(envoy)` walks the rename dict, validates each path exists, and populates two maps:

- `alias_to_name`: alias string → original attribute name (used for `__getattr__` lookup).
- `name_to_aliases`: original attribute name → list of aliases (used for repr).

`Envoy.__getattr__` (`envoy.py:980`) checks `_alias.alias_to_name` first:

```python
if self._alias is not None and name in self._alias.alias_to_name:
    return util.fetch_attr(self, self._alias.alias_to_name[name])
```

So `model.layers` becomes `model.transformer.h` under the hood.

## Repr shows aliases

```python
print(model)
# GPT2LMHeadModel(
#   (transformer/model/mdl): GPT2Model(
#     (h/layers): ModuleList(...)
#     ...
```

Aliases are joined with `/` next to the original name in the repr (`Envoy.__repr__`, `envoy.py:907`).

## Interaction with `envoys=` (custom Envoy classes)

When you pass `envoys={"attn": MyAttnEnvoy}` and rename `self_attn -> attn`, the suffix-match in `Envoy._path_matches_key` (`envoy.py:654`) consults the rename dict component-wise. So a path ending in `self_attn` will match the `"attn"` key when the rename is in play.

```python
model = LanguageModel(
    "meta-llama/Llama-3.1-8B",
    rename={"self_attn": "attn"},
    envoys={"attn": MyAttnEnvoy},
)
```

Type-keyed envoys (e.g. `{torch.nn.Linear: ...}`) are tried first; string keys are a fallback.

## Updating aliases after construction

```python
model._update_alias({"new_name": "transformer.h"})
```

`Envoy._update_alias` (`envoy.py:829`) replaces the current `_alias` and re-runs `build`.

## Cache keys honor the rename

`tracer.cache(...)` (`src/nnsight/intervention/tracing/tracer.py:465`) passes `rename` and the inverted alias map into `Cache.CacheDict`, so:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

cache["model.layers.0"].output    # works via alias
cache.layers[0].output            # also works
```

## Gotchas

- Aliases work **only on the root envoy** (the one created with `rename=`). Subtree renames you do mid-tree do not propagate up.
- A single-component rename key applies to **every** matching component — `{"mlp": "feedforward"}` renames `mlp` on every block, not just the first.
- A rename to a name that conflicts with an existing Envoy attribute (`output`, `input`, `trace`, ...) will collide with `Envoy.__getattr__` resolution. Avoid those names.
- When wrapping a pre-loaded model, pass `rename=` to the constructor — aliases must be set up before `Aliaser.build` runs (which happens during `Envoy.__init__`).
- `MetaMixin.__init__` (`src/nnsight/modeling/mixins/meta.py:38`) accepts `rename=` and forwards it to `NNsight.__init__`. So `LanguageModel("...", rename={...})` is the standard form.

## Related

- `docs/usage/trace.md`
- `docs/usage/access-and-modify.md`
- `docs/usage/cache.md`
