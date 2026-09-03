---
title: Source Tracing
one_liner: .source rewrites a module's forward AST so every call site and every assignment becomes a location bracketed through Interleaver.handle (input / skip / output) — the same primitive modules use, one level finer. Source is the forward view; SourceEnvoy is one operation.
tags: [concept, mental-model, source-tracing]
related: [docs/concepts/envoy.md, docs/concepts/interleaver-and-controller.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/interleaver.py]
---

# Source Tracing


## What this is for

A module's *controller* surfaces exactly two locations: `.input` and `.output`. Everything in between — the individual operations inside a `forward` — is invisible, because an operation is not a submodule with a controller of its own.

`.source` makes those intermediates observable, editable, and skippable. It parses the module's `forward`, rewrites every call `fn(*args, **kwargs)` into `__nnsight_op__("source.{name}_{n}", fn, *args, **kwargs)` and every assignment `x = value` into `x = __nnsight_op__("source.x_{n}", __nnsight_bind__, value)` (the same bracket around an identity), and re-executes the rewritten function as the forward. At run time each op is bracketed through the **same** `Interleaver.handle` primitive modules use — `.input` before, a `.skip` gate, `.output` after — just one level finer. The interleaver knows nothing about source.

## When to use / when not to use

- Use it to reach a value that is neither a module's input nor its output — an internal intermediate (an activation function, an attention call, a `torch.matmul`).
- Use an assignment op (`scores_0`) for a value that is not a call's return — a product, a slice, a loop's running state.
- Use recursive `.source` (`.source` on a `SourceEnvoy`) to descend into a *called function's* body.
- **Do not** drill `.source` into a call that is itself a submodule — access that submodule directly. It raises `SourceNotAvailable`.
- The forward must have recoverable Python source. Builtins/C functions raise `SourceNotAvailable`; decorated forwards and closures are instrumented (see below).

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
 self_c_fc_0     ->  0     hidden_states = self.c_fc(hidden_states)
 hidden_states_0 ->  +     ...
 self_act_0      ->  1     hidden_states = self.act(hidden_states)
 hidden_states_1 ->  +     ...
 self_c_proj_0   ->  2     hidden_states = self.c_proj(hidden_states)
 hidden_states_2 ->  +     ...
 self_dropout_0  ->  3     hidden_states = self.dropout(hidden_states)
 hidden_states_3 ->  +     ...
                     4     return hidden_states
```

Each source line carries two operations: the call, and the `+` line for the
assignment that binds its result. The labels come from the library's source, so they
move with it — these are `transformers` 5.15.

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

`Source(envoy)` calls `install_source(envoy)`, which:

1. `compiled(forward)` (cached per code object) parses the source, and `Instrument` — an `ast.NodeTransformer` — rewrites every `Call`:
   ```python
   self.c_proj(attn_output)
   # becomes
   __nnsight_op__("source.self_c_proj_0", self.c_proj, attn_output)
   ```
   It descends into arguments *before* numbering the outer call, so nested calls get lower indices (execution order). An assignment is rewritten the same way around an identity:
   ```python
   attn_output = self.c_proj(attn_output)
   # becomes
   attn_output = __nnsight_op__("source.attn_output_0", __nnsight_bind__, __nnsight_op__("source.self_c_proj_0", self.c_proj, attn_output))
   ```
   so `attn_output_0.output` is the assigned value. A bare `super()` is left alone: it reads `__class__` off the frame that calls it.
2. The rewritten AST is compiled and materialized into a new function whose `__nnsight_op__` global is bound to an `op` closure anchored to the module (`make_op`). A function with free variables (a decorator's wrapper, a `super()` call) is compiled inside a shell function whose parameters are those names, and the new function is given the original closure cells, so it keeps reaching what it closed over.
3. That instrumented function becomes the module's **body** in its `State`; the installed **controller** forward runs it (see below).

Decorators are handled before step 1. `decorator_chain` parses each wrapper's own source: a wrapper that directly calls exactly one Python function it closes over (`functools.wraps` decorators, `@torch.no_grad()`, transformers' `force_accelerate_hooks`) is peeled, and `rewrap` rebuilds it around the instrumented function so its behaviour still runs. A wrapper that doesn't call what it closes over — transformers' experts wrapper hands `original_forward` to a lookup and calls the result — is the function instrumented, closure intact; its ops are the dispatch (`experts_forward_1`), and `.source` on that op drills into whichever implementation ran.

The result (`Compiled`) carries the op labels, their line numbers, and the dedented source for the reprs.

## The per-module controller and `State`

Installation is **lazy and permanent**. The first time a module is sourced *or* skipped, `install_controller` replaces its `forward` with a `controller` closure (`make_controller`) and stores a `State` on `module.__dict__["__nnsight__"]`.

`State` holds:

- `body`: the (unbound) forward to run — the original, or the source-instrumented one once `.source` is used.
- `routes`: one entry per interleaver that instrumented this module — a weakref to it, the path it addresses the module by, and the three location strings (`.input`, `.skip`, `.output`) built once rather than per call. `active()` walks the list and returns the first route whose interleaver is interleaving *and* has workers, so a run with no intervention in it (a vLLM step nobody is tracing) skips the handoffs entirely.
- `sourced` / `compiled`: whether `body` is the instrumented forward, and its `Compiled`.

The controller reads `State` live on every call. With no active route it runs the body straight through. With one, it makes the module's three handoffs itself: `.input` (report/replace the arguments), the `.skip` gate, then the body, then `.output` — the same three `run_op` makes for a single operation, one level up. A skip returns the replacement without running the body, cut down to this device's shard first on a sharded runtime. Because state is read per call rather than baked in, a module wrapped by several envoys reports to whichever trace is currently active, and source and skip compose on the one controller.

The body runs through `run_body`, which re-applies accelerate's `pre_forward`/`post_forward` when the module carries an `_hf_hook`. Accelerate installs device alignment by replacing `module.forward` in the instance `__dict__` — the same slot the controller takes — so without this the inter-module tensor moves a sharded model depends on would silently stop happening.

## `run_op`: the per-operation bracket

When an instrumented op fires inside a trace, `run_op` brackets it under `{path}.{location}`:

1. `handle("{base}.input", (args, kwargs))` — report/replace the arguments.
2. `handle("{base}.skip", NO_SKIP)` — if a skip is pending, return the replacement as this op's output; the call never runs.
3. (recursive source) if a worker asked to drill into this op, offer the raw `fn` over `{base}.fn` so the worker can hand back an instrumented copy (see below).
4. `value = fn(*args, **kwargs)` — run the call.
5. `handle("{base}.output", value)` — report/replace the return value.

Steps 1/2/5 are the exact three handoffs a module's controller makes — the interleaver treats an op location no differently from a module location. Occurrence tagging (`.i{n}`) applies the same way (see [Interleaver and Controller](interleaver-and-controller.md)).

## SourceEnvoy: one operation

`SourceEnvoy` is the operation-level analogue of an `Envoy`. Its handles are plain properties over the mediator, mirroring an `Envoy`'s:

- `.output` — the op's return value (`Mediator.value`/`swap` on `{path}.output`).
- `.input` / `.inputs` — first argument / full `(args, kwargs)` on `{path}.input`.
- `.skip(value)` — `Mediator.skip` on `{path}.skip`; the call never runs and `value` flows on.
- `.source` — drill into the called function (recursive).

## Recursive .source

To descend into an operation's called function:

```python
with model.trace("Hello world"):
    attn  = model.transformer.h[0].attn.source
    inner = attn.attention_interface_1.source          # drill into the call
    scores = inner.attn_weights_1.output.save()        # an op inside it
assert scores.shape[1] == 12                           # [batch, heads, q, k]
```

`SourceEnvoy.source` is **only available inside a trace**, because the call target is resolved from the live value flowing through the call (it's often a local, e.g. an attention implementation). It:

1. Marks the op location requested in `interleaver.sourced` (a `None` placeholder), then parks on `{path}.fn` until the op fires and `run_op` hands back the live `fn`.
2. If `fn` is a submodule, raises `SourceNotAvailable` (access that submodule directly). Otherwise `instrument(fn, make_op(...))` builds an instrumented copy whose ops land under `{path}.source.{label}`.
3. Caches the instrumented copy in `interleaver.sourced[path]` so later fires this run (e.g. generation steps) reuse it, and returns a nested `Source`.

Which function you land in is the one the model is *running*, so the inner names depend on `attn_implementation`. Under `eager` the interface is `eager_attention_forward` and its ops are `query_size_0, scaling_0, key_transpose_0, torch_matmul_0, attn_weights_0` (scaled scores), `attn_weights_1` (masked), `nn_functional_softmax_0`, `attn_weights_2` (probabilities), through `attn_output_1`. Under the default `sdpa` it is `sdpa_attention_forward`, whose ops start `kwargs_get_0, logger_warning_once_0, sdpa_kwargs_0, hasattr_0, use_gqa_in_sdpa_0, repeat_kv_0, key_0, ...` and never build a probability matrix at all. Print the drilled `Source` rather than assuming either.

## Iteration tracking for source

Occurrence tracking is **unified** with modules — no separate counter hooks. Because an op goes through `Interleaver.handle` *every time it fires*, its per-location count (`Mediator.iterations`) advances per **fire**:

- An op inside a loop (e.g. an MoE expert loop) fires many times in one forward pass — each fire is its own `iter[...]` index.
- An op that fires once per forward (across generation steps) counts once per forward, like the module.

So `tracer.iter[i]` over a source op selects the i-th *fire*, which differs from module-level `iter` (once per forward pass) exactly when the op loops within a forward.

## Caching across forward replacement

The instrumented forward is never written onto the module's class — it lives as the `State.body`, run by the controller. When the module object itself is replaced (accelerate's dispatch swap, nnsight's own `_update`), `instrument` re-installs the controller on the new module. Instrumented code is memoized per original code object in `FORWARD_CACHE`, so a second module of the same class pays no parse or compile cost. Failures are not memoized: a forward whose source cannot be recovered raises every time it is asked for.

## Gotchas

- **Op names are the full dotted callee.** `self.c_proj(x)` → `self_c_proj_0`, not `c_proj_0`. `print(module.source)` is the source of truth — an `AttributeError` on a wrong name lists the available ops.
- **Names track the source.** A `transformers` version bump that renames an internal call renames its op.
- **Forward must have Python source.** A builtin / C function → `SourceNotAvailable`. Decorators and closures are fine; a dispatching wrapper exposes its dispatch, not the body it wraps.
- **Assignment ops can't be drilled.** `x_0.source` raises `SourceNotAvailable` — there is no callee.
- **Don't drill `.source` into a submodule call.** Access the submodule directly; drilling raises `SourceNotAvailable`.
- **Recursive `.source` is trace-only.** The callee is resolved from the live value at run time.
- **A skipped op still reports `.output`** as the replacement — reading `.output` of a skipped op returns what you skipped it with.
- **The listing is static; the forward is not.** `Source` decomposes the whole `forward`, so it names operations on branches this model's config never takes. Asking for one parks a worker at a location the model never reaches, and the trace fails with `OutOfOrderError` — a message about ordering for what is really a dead label. See [source.md](../usage/source.md#operations-that-never-run).

## Related

- [Envoy](envoy.md) — `SourceEnvoy` mirrors an `Envoy`'s `.input`/`.output`/`.skip`.
- [Interleaver and Controller](interleaver-and-controller.md) — the `handle` primitive ops share with modules, and occurrence tagging.
- Source: `src/nnsight/intervention/source.py` (`Source`, `SourceEnvoy`, `State`, `Instrument`, `run_op`, `install_source`, `install_controller`).
