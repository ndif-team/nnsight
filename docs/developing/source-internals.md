---
title: Source Internals
one_liner: AST-instrumented forwards that make individual operations inside a module addressable.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/architecture-overview.md, docs/developing/extending-envoy.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Source Internals

## What this covers

Module `input`/`output` are the only two locations the forward *hooks* surface.
Everything in between — the individual operations a `forward` performs — is
invisible to them, because an operation isn't a submodule with its own hook.
`.source` makes those intermediates observable, editable, and skippable. This doc
covers the AST instrumentation (`Instrument`), the per-module controller and
`__nnsight_op__`, the `State` object keyed on interleavers, the `Source` /
`SourceEnvoy` user views, and recursive `.source`.

The whole-forward view is `Source`; the single-operation view is `SourceEnvoy`, whose
`.output`/`.input`/`.inputs` are the same `eproperty` descriptors `Envoy` uses, one level finer.

## Quick check (verified)

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)
mlp = model.transformer.h[0].mlp

# operations of the MLP's forward, in execution order
[op.name for op in mlp.source]
# -> ['self_c_fc_0', 'self_act_0', 'self_c_proj_0', 'self_dropout_0']

with model.trace("The Eiffel Tower is in"):
    act = mlp.source.self_act_0.output.save()   # inside forward
    out = mlp.output.save()                      # the module's output
tuple(act.shape), tuple(out.shape)
# -> ((1, 7, 3072), (1, 7, 768))
```

`repr(mlp.source)` prints the forward with each op labelled at its call site:

```
                    * def forward(self, hidden_states: ...) -> ...:
 self_c_fc_0    ->  0     hidden_states = self.c_fc(hidden_states)
 self_act_0     ->  1     hidden_states = self.act(hidden_states)
 self_c_proj_0  ->  2     hidden_states = self.c_proj(hidden_states)
 self_dropout_0 ->  3     hidden_states = self.dropout(hidden_states)
                    4     return hidden_states
```

## The one primitive

The interleaver runs on exactly one primitive: a location string and
`Interleaver.handle`, which serves a value to interventions and returns whatever
they wrote back. Module `input`/`output` are just the two locations the forward
hooks emit. `source` is a *client* of that primitive — it adds more locations,
mid-forward, without the interleaver knowing source exists (module docstring,
`source.py:1`).

It does this in two steps:

1. Parse the module's `forward` and rewrite every call `fn(*args, **kwargs)` into
   `__nnsight_op__("source.{name}_{n}", fn, *args, **kwargs)`.
2. At run time `__nnsight_op__` brackets the call with `handle` on its `.input`
   (before) and `.output` (after) — both readable/replaceable — plus a `.skip` gate
   that can bypass the call.
3. Rewrite every assignment `x = value` into
   `x = __nnsight_op__("source.x_{n}", __nnsight_bind__, value)` — the same
   bracket around an identity (`bind`), so a value that is not a call's return (a
   product, a loop's running state) is addressable by the name the forward gives
   it. Tuple-literal unpacking gets one op per name; chained and augmented
   assignments are left alone.

Op naming: `name` is the called function's dotted path joined with `_`
(`self.act(...)` → `self_act`, `torch.relu(...)` → `torch_relu`, `dropout(...)` →
`dropout`); `n` is a per-name counter in **execution order** — nested calls run
inner-first, so the inner call is `_0`, matching the order the interleaver serves
values. Assignments are `{target}_{n}` on the same counter (`out[:, i] = v` →
`out_n`, `self.buf = v` → `self_buf_n`), so a name bound and then called is
`_0` at the binding and `_1` at the call (`attention_interface_1` is GPT-2's
attention call).

## Instrumentation (compile time)

`Instrument(ast.NodeTransformer)` (`source.py:132`) is the AST rewriter.

- `dotted(expr)` walks an attribute chain (skipping subscripts) to build both call
  names and assignment-target names, joined with `_`; `wrap` numbers and emits the op.
- `visit_Call(node)` (`:170`) calls `generic_visit` **first** (so inner calls are
  numbered before outer ones), then rewrites the node to
  `__nnsight_op__("source.<label>", node.func, *node.args, **node.keywords)`.
  The inserted `__nnsight_op__` call isn't re-visited, so it's never counted as an op.

`compile_source(func)` drives compilation and raises `SourceNotAvailable`
only when there is nothing to parse: a builtin/C function (no `__code__`) or
unrecoverable source (`inspect.getsourcelines` fails). Source is read from the
**code object**, not the function — given a `functools.wraps` wrapper, `inspect`
follows `__wrapped__` and hands back the decorated function's source instead.

A function with free variables — a decorator's wrapper, a `forward` that calls
`super()` — is compiled inside a shell function whose parameters are the free
names, so they compile as free variables again rather than as globals, and
`instrument` attaches the original closure cells to the new function (matched
by name; the shell can order them differently). Every function goes through the
shell — with no free variables it simply has no parameters.

Decorators are handled by `instrument`, the one entry point `install_source` and
recursive `.source` share. `decorator_chain` peels wrappers with `peel_index`,
which parses the wrapper's own source for the free names it *calls directly* and
peels through the one cell holding a Python function. A wrapper that doesn't
call what it closes over is not peeled: transformers' `use_experts_implementation`
wrapper hands `original_forward` to a lookup and calls the result, and peeling it
would instrument an eager loop that never runs (the ops would sit dead and every
request would be "out of order"). Instrumented as it is, its ops are the dispatch,
and drilling into the dispatch op reaches whichever implementation ran. Two
called functions are ambiguous and get the same treatment. `rewrap` rebuilds
peeled wrappers around the instrumented function with fresh closures — the
wrapper is the class's attribute, shared by every instance in the process. A
bare `super()` is never wrapped as an op: it reads `__class__` off the calling
frame, which inside `__nnsight_op__` is the wrong one.

Each rewritten call copies the original call's source location onto its wrapper node
(`ast.copy_location`), so `ast.increment_lineno` then shifts everything to file
coordinates and an exception inside an instrumented forward reports the real line
(without the copy, the locationless wrapper would take the raw offset). It then
compiles the module and lifts the child code object whose `co_name == func.__name__`. The
result is a `Compiled` NamedTuple (`:71`): `code`, `names` (op labels in execution
order), `lines` (label → 1-based line, for the repr), and `source`.

`compiled(func)` (`source.py:241`) memoizes this keyed on `func.__code__` in
`FORWARD_CACHE`, caching failures too, so compiling the instrumented forward is a
one-time cost per forward per process.

## `State` — keyed on interleavers, not one envoy

Per-module source/skip state lives at `module.__dict__["__nnsight__"]` (the
`STATE` key) as a `State` (`source.py:80`):

```python
class State:
    __slots__ = ("interleavers", "body", "sourced")
    def __init__(self, body):
        self.interleavers = weakref.WeakKeyDictionary()  # interleaver -> path string
        self.body = body        # unbound forward to run when not skipped
        self.sourced = False    # whether body is the instrumented forward
```

- `register(interleaver, path)` (`:109`) records the path an interleaver addresses
  the module by.
- `active()` (`:113`) returns the `(interleaver, path)` whose `interleaver.interleaving`
  is currently `True` — at most one — else `(None, None)`.

Keying on a `WeakKeyDictionary` of interleaver → path is what lets a module wrapped
by several `Envoy`s/`Interleaver`s report to whichever one is currently
interleaving. The dict holds interleavers weakly so a finished local wrapper's
interleaver can drop out without a reference cycle. Skip state is **not** stored
here — it's queried live through the interleaver's `.skip` gate.

## The controller and `__nnsight_op__` (run time)

The first time a module is sourced or skipped, its `forward` is replaced —
permanently and inertly — by a controller closure.

`make_controller(module)` (`source.py:362`):

```python
@functools.wraps(original)   # keeps the signature; generate() introspects it
def controller(*args, **kwargs):
    state = module_ref().__dict__[STATE]
    interleaver, path = state.active()
    if interleaver is None:                       # outside a trace: pass through
        return state.body(module, *args, **kwargs)
    skipped = skipped(interleaver, path)         # module-level .skip gate
    if skipped is not NO_SKIP:
        return skipped
    return state.body(module, *args, **kwargs)
```

`state.body` is the **unbound** forward (original or instrumented), so the
controller passes `module` explicitly. It reads live `State` every call, so
re-wrapping and multiple wrappers just work. It's installed via `install_controller`
(`source.py:392`), which stores the controller under `module.__dict__["forward"]`
(shadowing the class method for `nn.Module.__call__`) and, on every access,
`state.register(envoy.interleaver, envoy.path)`.

`__nnsight_op__` is the name bound into each instrumented forward's globals. It's
built per module by `make_op` (`source.py:302`):

```python
def op(location, fn, *args, **kwargs):
    interleaver, path = module_ref().__dict__[STATE].active()
    if interleaver is None:
        return fn(*args, **kwargs)               # fast path: not interleaving
    return run_op(interleaver, f"{path}.{location}", fn, args, kwargs)
```

`run_op` (`source.py:271`) is the bracket:

```python
args, kwargs = interleaver.handle(f"{base}.input", (args, kwargs))
skipped = skipped(interleaver, base)
if skipped is not NO_SKIP:
    return interleaver.handle(f"{base}.output", skipped)   # skipped: don't call fn
if base in interleaver.sourced:                            # recursive .source drill-in
    interleaver.handle(f"{base}.fn", fn)
    entry = interleaver.sourced.get(base)
    if entry is not None:
        fn = entry[0]
value = fn(*args, **kwargs)
return interleaver.handle(f"{base}.output", value)
```

The full location is `f"{path}.{location}"` where `location` already carries the
`source.<label>` prefix — e.g. `model.transformer.h.0.mlp.source.self_act_0.output`.

**The fast path is essential.** Outside a trace (`active()` is `(None, None)`) both
the controller and `op` call straight through — one weakref deref, one dict read, one
`None` check. An instrumented forward left installed on an idle model costs almost
nothing.

`install_source(envoy)` (`source.py:414`) builds the instrumented forward once
(`FunctionType(compiled.code, {**globals, OP: make_op(module)}, ...)`), sets it as
`state.body`, and flips `state.sourced = True`. `install_controller(envoy)` (`source.py:437`)
just installs the controller (no source needed to skip a whole module).

## Wiring from the interleaver — `instrument`

`Interleaver.instrument(envoy)` is called from `Envoy.__init__` and again from
`Envoy._update` (on dispatch). It lets the runtime's `Fragments` record what the
module's values are at the handoff, then calls `install_controller(envoy)`, which
makes the module's `forward` the controller: the `{path}.input` handoff, the
`{path}.skip` gate, the body, the `{path}.output` handoff. Each handoff returns
the handled value, so interventions can edit input or output; with no trace
running the controller calls the body straight through. The controller is in
place before `nn.Module.__call__` binds `forward`, which is what lets a skip read
the module's own input first.

On dispatch, `_update` re-runs `instrument` against the newly loaded module, so the
real module gets its own controller and `State` — the meta module's don't carry over.

## The user views

Both are thin objects that build location strings and delegate to `Mediator`.

`Source` (`source.py:664`) — a whole forward decomposed into operations, returned
by `Envoy.source` (`envoy.py:523`, a plain `@property` that constructs `Source(self)`
and thereby source-instruments the module). `__getattr__(name)` returns the
`SourceEnvoy` for that op (or raises `AttributeError` listing the available names,
as seen above); `__iter__` yields them in execution order; `__repr__` renders the
labelled forward.

`SourceEnvoy` (`source.py:448`) — one operation. Its `.output`/`.input`/`.inputs`
are `eproperty` descriptors (`source.py:550`, `:565`, `:581`) and `.skip()` a plain
method, each keyed on `{self.path}.{output|input|skip}`. Reading an eproperty runs
`Mediator.value(location)` then the descriptor's preprocess stub; writing runs an
optional postprocess then `Mediator.swap(location, ...)`:

| Access | Location / callback |
|--------|--------------|
| `op.output` (get / set) | `eproperty` on `f"{path}.output"` — identity preprocess |
| `op.inputs` | `eproperty(key="input")` on `f"{path}.input"` — the whole `(args, kwargs)` |
| `op.input` | `eproperty` on `f"{path}.input"` — preprocess takes the first arg; `postprocess` repacks on write |
| `op.skip(replacement)` | `Mediator.skip(f"{path}.skip", replacement)` |

These are the same descriptors `Envoy` uses for its own `.input`/`.output`/`.skip`
(`envoy.py:419`, `:448`, `:483`), one level finer. `Mediator.value`/`swap`/`skip`
(`interleaver.py:270`, `:280`, `:291`) park the intervention greenlet until the
interleaver reaches that location. The `Event` protocol is
`VALUE`/`SWAP`/`SKIP`/`BARRIER` (`interleaver.py:56`).

## Recursive `.source`

`SourceEnvoy.source` (`source.py:503`) drills into an operation's *own* function —
only inside a trace (it needs the live call target):

```python
with model.trace("hi"):
    inner = model.transformer.h[0].attn.source.some_call_0.source.torch_op_0.output.save()
```

Mechanics: it marks the op requested (`interleaver.sourced[self.path] = None`), parks
via `Mediator.value(f"{self.path}.fn")` to receive the live callable from the model
side, rejects submodules (`SourceNotAvailable` — call `.source` on the submodule
directly instead), then builds an instrumented copy via `instrument` with an op located at the
drilled path (`make_op`) and caches `(instrumented, compiled)`. On the
model side, `run_op`'s `interleaver.sourced` branch swaps that instrumented copy in
so subsequent calls run the drilled-in version.

## Gotchas

- **A GPT-2 block's `.output` is a plain `Tensor`** `(batch, seq, hidden)` in
  current transformers, not a tuple. Attention submodules still return a tuple.
  Check `repr(module.source)` / the shape rather than assuming a tuple.
- **Prefer whole-value assignment over an in-place slice into a tuple element.**
  Set `module.output = x` (or an op's `.output`) rather than mutating a narrowed
  view across a barrier.
- **Builtin / C forwards can't be sourced** — you'll get `SourceNotAvailable`.
  Access `.output`/`.input` on the module (or on a real submodule) instead.
- **A dispatching wrapper shows its dispatch, not the body it wraps.** Drill into
  the dispatch op (`experts_forward_1.source`) for the implementation that ran.
- **Assignment ops have no callee.** `x_0.source` raises `SourceNotAvailable`.

## Key files

- `src/nnsight/intervention/source.py` — `Instrument` (`:132`), `Compiled` (`:71`),
  `State` (`:80`), `make_controller` (`:362`), `make_op` (`:302`), `run_op`
  (`:271`), `install_source` (`:414`), `Source` (`:659`), `SourceEnvoy` (`:448`,
  with its `.output`/`.input`/`.inputs` eproperties at `:550`/`:565`/`:581`)
- `src/nnsight/intervention/interleaver.py` — `Event` (`:56`), `Mediator` (`:93`),
  `Interleaver.instrument` (`:521`), `Interleaver.sourced` (`:482`)
- `src/nnsight/intervention/envoy.py` — `Envoy.source` (`:523`), `.input`/`.output`/`.skip`

## Related

- [interleaver-internals.md](./interleaver-internals.md) — `Mediator.handle` semantics
- [extending-envoy.md](./extending-envoy.md) — the `eproperty` descriptor and `.provide` in general
- `tests/test_source.py`, `tests/test_interleaving.py::TestSourceIteration` — reference usage
