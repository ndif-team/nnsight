---
title: Source Tracing
one_liner: .source rewrites a module's forward AST so every call site becomes a location bracketed through Interleaver.handle (input / skip / output) — the same primitive modules use, one level finer. Source is the forward view; SourceEnvoy is one operation.
tags: [concept, mental-model, source-tracing]
related: [docs/concepts/envoy.md, docs/concepts/interleaver-and-hooks.md]
sources: [src/nnsight/intervention/source.py:132, src/nnsight/intervention/source.py:271, src/nnsight/intervention/source.py:362, src/nnsight/intervention/source.py:414, src/nnsight/intervention/source.py:447, src/nnsight/intervention/source.py:664]
---

# Source Tracing

> Naming changed from the old docs: the module-forward view is now **`Source`** (was `SourceEnvoy`); a single operation is now **`SourceEnvoy`** (was `OperationEnvoy`). The per-module `SourceAccessor`/`OperationAccessor` split is gone — there is one per-module `_State` and one `handle`-based primitive.

## What this is for

Module `.input`/`.output` are the only two locations the forward *hooks* surface. Everything in between — the individual operations inside a `forward` — is invisible, because it isn't a submodule with its own hook.

`.source` makes those intermediates observable, editable, and skippable. It parses the module's `forward`, rewrites every call `fn(*args, **kwargs)` into `__nnsight_op__("source.{name}_{n}", fn, *args, **kwargs)`, and re-executes the rewritten function as the forward. At run time each op is bracketed through the **same** `Interleaver.handle` primitive modules use — `.input` before, a `.skip` gate, `.output` after — just one level finer. The interleaver knows nothing about source.

## When to use / when not to use

- Use it to reach a value that is neither a module's input nor its output — an internal intermediate (an activation function, an attention call, a `torch.matmul`).
- Use recursive `.source` (`.source` on a `SourceEnvoy`) to descend into a *called function's* body.
- **Do not** drill `.source` into a call that is itself a submodule — access that submodule directly. It raises `SourceNotAvailable`.
- The forward must be a plain Python function with recoverable source and no decorators. Builtins/C functions, closures over free variables, and decorated forwards raise `SourceNotAvailable`.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

# Discover: print .source to see the forward with each op labelled.
print(model.transformer.h[0].mlp.source)

with model.trace("Hello world"):
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()   # read
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0      # edit
```

## Discovery: print .source

`print(model.transformer.h[0].mlp.source)` renders the forward with each operation labelled at its call site (verified output):

```
                    * def forward(self, hidden_states: ...) -> torch.FloatTensor:
 self_c_fc_0    ->  0     hidden_states = self.c_fc(hidden_states)
 self_act_0     ->  1     hidden_states = self.act(hidden_states)
 self_c_proj_0  ->  2     hidden_states = self.c_proj(hidden_states)
 self_dropout_0 ->  3     hidden_states = self.dropout(hidden_states)
                    4     return hidden_states
```

Operation names are the **full dotted callee** joined with `_`, plus a per-name occurrence index: `self.c_fc(...)` → `self_c_fc_0`, `torch.relu(...)` → `torch_relu_0`, a bare `dropout(...)` → `dropout_0`. Indexing runs in execution order (nested calls inner-first). Print a single op to see it in context, flagged with `-->`/`<--`:

```
model.transformer.h.0.mlp.source.self_c_fc_0:

    def forward(self, hidden_states: ...) -> torch.FloatTensor:
    --> hidden_states = self.c_fc(hidden_states) <--
        hidden_states = self.act(hidden_states)
        ...
```

Iterating a `Source` yields its `SourceEnvoy`s in execution order: `[op.name for op in model.transformer.h[0].mlp.source]`.

## How rewriting works

`Source(envoy)` (`source.py:664`) calls `install_source(envoy)` (`source.py:414`), which:

1. `_compiled(forward)` (cached per code object) parses the source, and `_Instrument` (`source.py:132`) — an `ast.NodeTransformer` — rewrites every `Call`:
   ```python
   self.c_proj(attn_output)
   # becomes
   __nnsight_op__("source.self_c_proj_0", self.c_proj, attn_output)
   ```
   It descends into arguments *before* numbering the outer call, so nested calls get lower indices (execution order).
2. The rewritten AST is compiled and materialized into a new function whose `__nnsight_op__` global is bound to an `op` closure anchored to the module (`_make_op`, `source.py:302`).
3. That instrumented function becomes the module's **body** in its `_State`; the installed **controller** forward runs it (see below).

The result (`Compiled`) carries the op labels, their line numbers, and the dedented source for the reprs.

## The per-module controller and `_State`

Installation is **lazy and permanent**. The first time a module is sourced *or* skipped, `install_controller` (`source.py:392`) replaces its `forward` with a `controller` closure (`_make_controller`, `source.py:362`) and stores a `_State` on `module.__dict__["__nnsight__"]`.

`_State` holds:

- `body`: the (unbound) forward to run — the original, or the source-instrumented one once `.source` is used.
- `interleavers`: a `WeakKeyDictionary` mapping each interleaver that instrumented the module to the path it addresses it by. `active()` picks the one whose trace is currently running.
- `sourced`: whether `body` is instrumented yet.

Each call, the controller reads the live `_State`: if no interleaver is running, it calls `body` straight through (inert outside a trace); otherwise it checks the `.skip` gate, then runs `body`. Because state is rebound per access, a module wrapped by several envoys/interleavers reports to whichever trace is currently active — and source and skip compose on one wrapper.

## `_run_op`: the per-operation bracket

When an instrumented op fires inside a trace, `_run_op` (`source.py:271`) brackets it under `{path}.{location}`:

1. `handle("{base}.input", (args, kwargs))` — report/replace the arguments.
2. `handle("{base}.skip", _NO_SKIP)` — if a skip is pending, return the replacement as this op's output; the call never runs.
3. (recursive source) if a worker asked to drill into this op, offer the raw `fn` over `{base}.fn` so the worker can hand back an instrumented copy (see below).
4. `value = fn(*args, **kwargs)` — run the call.
5. `handle("{base}.output", value)` — report/replace the return value.

Steps 1/2/5 are the exact three handles a module hook emits — the interleaver treats an op location no differently from a module location. Occurrence tagging (`.i{n}`) applies the same way (see [Interleaver and Hooks](interleaver-and-hooks.md)).

## SourceEnvoy: one operation

`SourceEnvoy` (`source.py:447`) is the operation-level analogue of an `Envoy`. Its handles are plain properties over the mediator, mirroring an `Envoy`'s:

- `.output` — the op's return value (`Mediator.value`/`swap` on `{path}.output`).
- `.input` / `.inputs` — first argument / full `(args, kwargs)` on `{path}.input`.
- `.skip(value)` — `Mediator.skip` on `{path}.skip`; the call never runs and `value` flows on.
- `.source` — drill into the called function (recursive).

## Recursive .source

To descend into an operation's called function:

```python
with model.trace("Hello world"):
    attn  = model.transformer.h[0].attn.source
    inner = attn.attention_interface_0.source          # drill into the call
    scores = inner.matmul_0.output.save()              # an op inside it
```

`SourceEnvoy.source` (`source.py:503`) is **only available inside a trace**, because the call target is resolved from the live value flowing through the call (it's often a local, e.g. an attention implementation). It:

1. Marks the op location requested in `interleaver.sourced` (a `None` placeholder), then parks on `{path}.fn` until the op fires and `_run_op` hands back the live `fn`.
2. If `fn` is a submodule, raises `SourceNotAvailable` (access that submodule directly). Otherwise `_compiled(fn)` parses it and `_build_instrumented` binds a nested `op` so *its* calls land under `{path}.source.{label}`.
3. Caches the instrumented copy in `interleaver.sourced[path]` so later fires this run (e.g. generation steps) reuse it, and returns a nested `Source`.

Verified: `attn.attention_interface_0.source` yields inner op names like `['kwargs_get_0', 'logger_warning_once_0', 'hasattr_0', 'use_gqa_in_sdpa_0', 'repeat_kv_0', 'repeat_kv_1', ...]`.

## Iteration tracking for source

Occurrence tracking is **unified** with modules — no separate counter hooks. Because an op goes through `Interleaver.handle` *every time it fires*, its per-location count (`Mediator.iterations`) advances per **fire**:

- An op inside a loop (e.g. an MoE expert loop) fires many times in one forward pass — each fire is its own `iter[...]` index.
- An op that fires once per forward (across generation steps) counts once per forward, like the module.

So `tracer.iter[i]` over a source op selects the i-th *fire*, which differs from module-level `iter` (once per forward pass) exactly when the op loops within a forward.

## Caching across forward replacement

The instrumented forward is never written onto the module's class — it lives as the `_State.body`, run by the controller. This survives `torch.compile` re-binding, accelerate's dispatch swap, and nnsight's own `_update`: `instrument` re-installs the controller on the new module. Instrumented code is memoized per original code object in `_FORWARD_CACHE`, so re-sourcing pays the parse+compile cost once.

## Gotchas

- **Op names are the full dotted callee.** `self.c_proj(x)` → `self_c_proj_0`, not `c_proj_0`. `print(module.source)` is the source of truth — an `AttributeError` on a wrong name lists the available ops.
- **Names track the source.** A `transformers` version bump that renames an internal call renames its op.
- **Forward must be plain Python.** No source, free variables, decorators, or C functions → `SourceNotAvailable` (cached, so it isn't re-parsed).
- **Don't drill `.source` into a submodule call.** Access the submodule directly; drilling raises `SourceNotAvailable`.
- **Recursive `.source` is trace-only.** The callee is resolved from the live value at run time.
- **A skipped op still reports `.output`** as the replacement — reading `.output` of a skipped op returns what you skipped it with.

## Related

- [Envoy](envoy.md) — `SourceEnvoy` mirrors an `Envoy`'s `.input`/`.output`/`.skip`.
- [Interleaver and Hooks](interleaver-and-hooks.md) — the `handle` primitive ops share with modules, and occurrence tagging.
- Source: `src/nnsight/intervention/source.py` (`Source`, `SourceEnvoy`, `_State`, `_Instrument`, `_run_op`, `install_source`, `install_controller`).
