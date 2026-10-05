---
title: Rename Modules
one_liner: Alias module paths via `rename={...}` at construction; supports single-component renames, subtree mounts, and multiple aliases.
tags: [usage, models, rename, aliases]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/aliasing.py, src/nnsight/intervention/envoy.py, src/nnsight/modeling/mixins/meta.py]
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

`rename` is `dict[str | type, str | list[str]]`. The behavior depends on the **key**
shape:

| Key form | Behavior |
|----------|----------|
| **Single component** (`"mlp"`) | Binds wherever it resolves — every block that has an `mlp` child gets the alias. |
| **Dotted path** (`"transformer.h"`, `"transformer.h.3.mlp"`) | Mounts that subtree on the **root** envoy under the alias name. |
| **Leading dot** (`".h"`) | The dot is a no-op; the path resolves relative to each envoy, so the alias binds on whichever envoy has that child (e.g. `model.transformer.layers`, not `model.layers`). |
| **`*` component** (`"*.h"`) | `*` stands for any one entry, so the key need not spell a name that differs by architecture: `"*.h"` mounts the `h` under whichever child of the root has one. Two matches from one envoy raise; none is skipped. |
| **Module class** (`Mamba2Mixer`) | Binds on every envoy that has exactly one direct child of that class, pointing at that child, whatever its native name. Two matching children on one envoy raise; none is skipped. |

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

# a `*` component: the backbone's own name is not spelled
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"*.h": "layers"})
g.layers[0] is g.transformer.h[0]                          # True

# a class key: the alias follows what each block holds, not what it is called
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={GPT2Attention: "self_attn"})
g.transformer.h[0].self_attn is g.transformer.h[0].attn    # True
```

A class key is for a tree that keeps different modules under one native name: a
hybrid whose every block has a `mixer` that is a state-space mixer on one block
and an attention on the next. `{Mamba2Mixer: "linear_attn", NemotronHAttention:
"self_attn"}` gives each block the standard name for what it holds, which no
name-keyed entry can express. A class that matches two direct children of one
envoy (both norms of a Llama block are `LlamaRMSNorm`) is an error rather than a
guess; key those by name.

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
- **A key that names nothing is skipped, not an error.** That is what lets one
  `rename` cover architectures that spell a module differently:
  `{"attn": "att", "self_attn": "att"}` binds whichever of the two exists.
- **Pass `rename=` at construction.** Aliases are bound during `Envoy.__init__`;
  there is no post-hoc alias API.
- **An alias that would shadow something raises at construction.** Aliases bind
  as plain attributes, so a name that is already taken — an `Envoy` attribute
  (`output`, `input`, `trace`, ...), a sibling module, an alias an earlier key
  already claimed, or a name on the **wrapped model** (`config`, `weight`,
  `eval`, and every other `nn.Module` attribute) — would leave the original
  reachable only through the tree, not by name. That is reported when the model is built rather than surfacing later as
  an intervention landing on the wrong module:

  ```
  ValueError: `rename` alias 'embed' for 'head' would shadow a child module of
  that name on `model`. ...
  ```

  Aliasing a key to its own name (`{"attn": "attn"}`) is a no-op, not a
  collision, so it stays usable alongside the cross-architecture pattern above.
- **A `*` or class key binds only where it matches one module.** An alias names
  one module, so `{"h.*": "block"}` raises on a stack of more than one block
  rather than picking one; `*` is for a component whose name varies
  (`"*.h"`), not for naming every entry. To give every entry's child a name,
  key the child (`{"attn": "self_attn"}`).
- **A dotted key mounts on the root; a leading-dot key mounts where it resolves.**
  Pick the form that matches where you want to reach the alias.

## Related

- [trace.md](trace.md)
- [access-and-modify.md](access-and-modify.md)
- [cache.md](cache.md)
