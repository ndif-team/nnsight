---
title: Source Tracing
one_liner: Hook intermediate operations inside a module's forward pass via .source.<op_name>.
tags: [usage, source, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/cache.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/intervention/source.py:359, src/nnsight/intervention/source.py:571, src/nnsight/intervention/source.py:688, src/nnsight/intervention/envoy.py:214, src/nnsight/intervention/hooks.py:438]
---

# Source Tracing

## What this is for

`module.source` exposes every call site inside a module's `forward` method as a hookable operation. nnsight rewrites the module's forward AST so each function call is wrapped in a per-call-site dispatcher; you can then read `.input` / `.output` on each operation, just like you would on a module. Useful when the value you need lives between two operations of a module's forward (e.g. attention scores before the projection) and there is no nested submodule to attach to.

## When to use / when not to use

- Use when the activation you need is computed mid-forward inside a single module (e.g. inside `GPT2Attention.forward`) and is not exposed as a child module.
- Use when you want to replace a single operation's output without rewriting the whole module.
- Skip when a child module already exposes the value you need — `model.transformer.h[0].mlp.output` is cheaper than `.source` because it does not rewrite the AST.
- Source rewriting is only triggered when `.source` has been touched on that module; the non-source path stays at zero overhead.

## Canonical pattern

Discover operations by printing `.source` on the module (works outside a trace):

```python
print(model.transformer.h[0].attn.source)
# ...
#   self_c_attn_0       -> 36     query_states, key_states, value_states = self.c_attn(hidden_states).split(...)
#   self_c_attn_0_split_0 -> 36     ...
#   attention_interface_0 -> 66     attn_output, attn_weights = attention_interface(...)
#   self_c_proj_0       -> 79     attn_output = self.c_proj(attn_output)
```

Inspect or modify a specific operation's value inside a trace:

```python
with model.trace("Hello"):
    # Read the operation's output
    attn = model.transformer.h[0].attn.source.attention_interface_0.output.save()

    # Read its inputs as (args, kwargs)
    args, kwargs = model.transformer.h[0].attn.source.attention_interface_0.inputs

    # Replace the operation's output
    model.transformer.h[0].attn.source.self_c_proj_0.output[:] = 0
```

Print a single operation to see it highlighted in context:

```python
print(model.transformer.h[0].attn.source.attention_interface_0)
# .transformer.h.0.attn.attention_interface_0:
#     ....
#     -->     attn_output, attn_weights = attention_interface( <--
#                 self,
#                 query_states,
#     ....
```

## Variations

### Recursive source

Operations whose target is itself a Python function can be re-traced. Just chain `.source` again:

```python
with model.trace("Hello"):
    sdpa = (
        model.transformer.h[0].attn
            .source.attention_interface_0
            .source.torch_nn_functional_scaled_dot_product_attention_0
            .output.save()
    )
```

The nested `SourceAccessor` is built on first access and cached on the parent `OperationAccessor`, so subsequent traces reuse it (`src/nnsight/intervention/source.py:646`).

### Operation naming

Operation names follow `<dotted_callee>_<index>`, where the index disambiguates repeated calls in the same forward. `self.c_proj(...)` becomes `self_c_proj_0`; a second call is `self_c_proj_1`. `torch.nn.functional.softmax(...)` becomes `torch_nn_functional_softmax_0`. The transformer that produces these names is `FunctionCallWrapper` (`src/nnsight/intervention/source.py:92`).

### Iteration support

Source operations participate in `tracer.iter[...]`. The iteration tracker is bumped for every operation under a `SourceAccessor` after each forward pass (`bump_source_paths`, `src/nnsight/intervention/iterator.py:75`), so `.source.<op>.output` inside `for step in tracer.iter[2]:` targets step 2.

## Gotchas

- **Don't call `.source` on a module from within another `.source`.** The recursive `.source` chain only follows plain functions; if the operation's target is a `torch.nn.Module`, nnsight raises `ValueError("Don't call .source on a module ... Call it directly with: <path>.source")` (`src/nnsight/intervention/source.py:660`). Access the submodule directly instead.
- **Decorators on the original forward are stripped** during AST rewriting, since they may fail when re-executed outside the original class context (e.g. `@auto_docstring`). The compute is identical; the decorator side effects are lost (`src/nnsight/intervention/source.py:174`).
- **Op-path tracker can miss the first hit if `.source` is built mid-iter-loop.** If the user's first ever `.source` access on a module happens at step N>0, the operation tracker starts at 0 and that one access misses. Subsequent steps work. Touch `.source` on the module before entering the iter loop to avoid this (`src/nnsight/intervention/iterator.py:125`).
- **Hooks installed by `.source` access are one-shot** and self-remove. Re-accessing `.source.<op>.output` re-registers the hook for the next forward.
- See [docs/gotchas/integrations.md](../gotchas/integrations.md) for the full set.

## Related

- [access-and-modify](access-and-modify.md) — Module-level `.output` / `.input`.
- [iter](iter.md) — Iteration semantics (operation-level paths included).
- [docs/concepts/source-tracing.md](../concepts/source-tracing.md) — Architecture deep dive.
