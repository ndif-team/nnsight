---
title: Rename Modules
one_liner: Alias module paths via `rename={...}` at construction; supports single-component renames, subtree mounts, and multiple aliases.
tags: [usage, models, rename, aliases]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/modeling/mixins/meta.py]
---

# Rename Modules

## What this is for

Different architectures name the same role differently (`transformer.h` vs
`model.layers` vs `gpt_neox.layers`). The `rename={...}` constructor kwarg installs
aliases so your intervention code is portable across model families.

An alias is an ordinary attribute pointing at the **same** child Envoy object — not
a copy — so the original path keeps working, iteration doesn't double-count, and
`Cache` keys resolve through aliases too.

## When to use / when not to use

- Use when writing analysis code that should work across HuggingFace architectures.
- Use to mount a deep subtree at a shorter path.
- Use to give a role a stable name across models.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    rename={
        "transformer.h": "layers",        # mount a subtree on the root
        "mlp": "my_mlp",                  # rename every MLP child
        "transformer": ["mdl", "backbone"],  # multiple aliases for one path
    },
)

with model.trace("Hello"):
    a = model.layers[0].my_mlp.output.save()      # via aliases
    b = model.transformer.h[0].mlp.output.save()  # original still works
    c = model.mdl.h[0].output.save()              # via first alias
    d = model.backbone.h[0].output.save()         # via second alias
```

## Forms of `rename` keys and values

`rename` is `dict[str, str | list[str]]`. The behavior depends on the **key**
shape:

| Key form | Behavior |
|----------|----------|
| **Single component** (`"mlp"`) | Binds wherever it resolves — every block that has an `mlp` child gets the alias. |
| **Dotted path** (`"transformer.h"`, `"transformer.h.3.mlp"`) | Mounts that subtree on the **root** envoy under the alias name. |
| **Leading dot** (`".h"`) | The dot is a no-op; the path resolves relative to each envoy, so the alias binds on whichever envoy has that child (e.g. `model.transformer.layers`, not `model.layers`). |

| Value form | Behavior |
|------------|----------|
| **String** (`"layers"`) | One alias. |
| **List** (`["mdl", "backbone"]`) | Multiple aliases for the same path. |

Verified behaviors:

```python
# single component: alias on every block
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"mlp": "my_mlp"})
g.transformer.h[0].my_mlp is g.transformer.h[0].mlp        # True

# subtree mount on the root
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"transformer.h": "layers"})
g.layers[0] is g.transformer.h[0]                          # True

# deep path mounts on the root
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"transformer.h.3.mlp": "my_mlp"})
g.my_mlp is g.transformer.h[3].mlp                          # True

# leading dot: binds where it resolves (under transformer), not on the root
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={".h": "layers"})
g.transformer.layers[0] is g.transformer.h[0]              # True
```

## Repr shows aliases

Aliases naming a direct child appear as `alias/realname` next to the module; subtree
mounts get their own line:

```python
print(model)
# ... (mdl/backbone/transformer): GPT2Model(...)   # direct-child aliases, joined with /
#     (my_mlp/mlp): GPT2MLP(...)
#     (layers): ModuleList(...)                     # subtree mount, own line
```

## Cache keys honor the rename

`tracer.cache(...)` resolves navigation against the (renamed) envoy tree:

```python
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"mlp": "my_mlp"})
with g.trace("Hello") as tracer:
    cache = tracer.cache()

cache.transformer.h[0].my_mlp.output                 # via alias
cache["model.transformer.h.0.mlp"].output            # original path — same value
```

You can also index a renamed `ModuleList` entry by its alias string:
`cache.model.h["second_layer"]` when `rename={"1": "second_layer"}`. See
[cache.md](cache.md).

## Gotchas

- **Aliases bind on every envoy where the key resolves** — a single-component key
  (`{"mlp": "my_mlp"}`) renames the `mlp` on *every* block, not just the first.
- **Pass `rename=` at construction.** Aliases are bound during `Envoy.__init__`;
  there is no post-hoc alias API.
- **Avoid alias names that collide with Envoy attributes** (`output`, `input`,
  `trace`, ...). They will shadow or be shadowed by those attributes.
- **A dotted key mounts on the root; a leading-dot key mounts where it resolves.**
  Pick the form that matches where you want to reach the alias.

## Related

- [trace.md](trace.md)
- [access-and-modify.md](access-and-modify.md)
- [cache.md](cache.md)
