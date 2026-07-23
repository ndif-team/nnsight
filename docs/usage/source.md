---
title: Source Tracing
one_liner: Hook intermediate operations inside a module's forward via .source.<callable>_<n>, and drill into called functions.
tags: [usage, source, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/skip.md, docs/usage/cache.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Source Tracing

## What this is for

`module.source` exposes every call site inside a module's `forward` as a hookable
operation. nnsight rewrites the module's `forward` AST so each call
`fn(*args, **kwargs)` is bracketed by the interleaver: you can read/replace each
operation's `.input` / `.output`, or `.skip` it — the same handles a module has,
one level finer.

Use it when the value you need lives *between* two operations of a forward (e.g. an
activation inside an MLP, or attention scores) and there is no submodule to attach
to.

## When to use / when not to use

- Use when the activation you need is computed mid-forward and isn't a child module.
- Use to read or replace a single operation's output.
- Skip when a child module already exposes the value — `model...mlp.output` is
  cheaper than `.source` (no AST rewrite).
- Source-instrumentation is installed lazily on first `.source` access and is inert
  outside a trace, so normal inference is unaffected.

## Discovering operations

`print(module.source)` renders the forward with each operation labelled at its
call site (works outside a trace):

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

print(model.transformer.h[0].mlp.source)
```

```
                    * def forward(self, hidden_states: ...) -> torch.FloatTensor:
 self_c_fc_0    ->  0     hidden_states = self.c_fc(hidden_states)
 self_act_0     ->  1     hidden_states = self.act(hidden_states)
 self_c_proj_0  ->  2     hidden_states = self.c_proj(hidden_states)
 self_dropout_0 ->  3     hidden_states = self.dropout(hidden_states)
                    4     return hidden_states
```

`print(module.source.self_act_0)` zooms in on one call site:

```
model.transformer.h.0.mlp.source.self_act_0:

    def forward(self, hidden_states: ...) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
    --> hidden_states = self.act(hidden_states) <--
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```

## Operation naming

Names are `<dotted_callee>_<occurrence>`, using the **whole** attribute chain
joined with `_`:

- `self.c_fc(...)` → `self_c_fc_0`
- `self.act(...)` → `self_act_0`
- `torch.relu(...)` → `torch_relu_0`

The occurrence counter is per callable and runs in **execution order** — nested
calls run inner-first, so `f(f(x))` gives the inner call `f_0` and the outer `f_1`.
A second `relu(...)` on another line is `torch_relu_1`.

## Canonical pattern

```python
with model.trace("Hello"):
    # read an operation's output
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()

    # read its inputs as (args, kwargs)
    args, kwargs = model.transformer.h[0].mlp.source.self_c_proj_0.inputs

    # replace an operation's output for the rest of the forward
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0
```

`.input` is the first argument; `.inputs` is the full `(args, kwargs)`. Assigning to
any of them replaces the value downstream. Operations must be requested in
execution order within a forward (see Gotchas).

## Recursive source — drill into a called function

An operation whose target is a plain Python function can be re-traced. Chain
`.source` again to expose *its* operations:

```python
with model.trace("The Eiffel Tower is in"):
    attn = model.transformer.h[0].attn
    scores = (
        attn.source.attention_interface_0
            .source.attn_output_transpose_0
            .output.save()
    )
```

**Recursive `.source` only works inside a trace.** The called function is resolved
from the live value flowing through the call at run time (call targets are often
local variables), so `some_op.source` outside a trace raises `SourceNotAvailable`.

## Iteration support

Source operations participate in `tracer.iter[...]`. Operation iteration counts
**invocations**, not forward passes:

```python
# nested source across generation steps
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    saved = nnsight.save([])
    for _ in tracer.iter[:3]:
        saved.append(
            model.transformer.h[0].attn
                 .source.attention_interface_0
                 .source.attn_output_transpose_0.output
        )
# len(saved) == 3; step 0 sees the full prompt, later (KV-cached) steps one token.
```

An op that fires once per forward is indexed per generation step; an op that loops
within one forward (e.g. an MoE expert loop) is indexed per fire.

## Gotchas

- **Recursive `.source` requires a live trace** — outside one it raises
  `SourceNotAvailable("recursive '.source' is only available inside a trace")`.
- **Drilling into a submodule call is refused.** If an operation calls a
  `torch.nn.Module`, `op.source` raises `SourceNotAvailable(... "calls a submodule;
  call '.source' on that submodule directly ...")`. Access the submodule's own
  `.source` instead.
- **Builtins / C functions have no recoverable source** — drilling into e.g.
  `torch.relu` raises `SourceNotAvailable`.
- **Decorated forwards are rejected.** A `forward` (or drilled-into function) with a
  decorator raises `SourceNotAvailable("decorated callable is not supported")` — the
  decorator is load-bearing and can't be safely dropped.
- **Requesting an operation out of execution order deadlocks → `OutOfOrderError`.**
  Reading `self_fc2_0.output` then `self_fc1_0.output` (fc1 runs first) is late.
- **Skipping a whole module drops its source ops.** A skipped module's body never
  runs, so reading `skipped_module.source.<op>.output` is out of order. See
  [skip.md](skip.md).
- **Unknown operation names raise `AttributeError` listing the available names** —
  handy for fixing a mistyped or wrong-occurrence label.

## Related

- [access-and-modify.md](access-and-modify.md) — module-level `.output` / `.input`.
- [skip.md](skip.md) — `.skip()` at the module and operation level.
- [iter-all-next.md](iter-all-next.md) — iteration semantics.
