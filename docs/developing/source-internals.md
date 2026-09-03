---
title: Source Internals
one_liner: How a forward becomes an instrumented forward — AST rewriting, occurrence numbering, the shell compile, decorator peeling, and the drill-in protocol.
tags: [internals, dev]
related: [docs/developing/controller.md, docs/developing/interleaver-internals.md, docs/developing/extending-envoy.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py]
---

# Source Internals

## What this covers

The compilation half of `.source`: how `src/nnsight/intervention/source.py` turns a
module's `forward` into an instrumented one, where operation labels come from, and
how a drill into a called function is negotiated at run time.

It assumes the other half. [Source Tracing](../concepts/source-tracing.md) is the
mental model — locations, `run_op`'s three handoffs, what a user sees.
[The Controller](controller.md) owns `State`, `active()`, the skip gate, and the
handoffs a module itself makes. Neither is repeated here.

## Where labels come from

`Instrument` is an `ast.NodeTransformer` with three visitors and one counter dict.

`dotted(expr)` builds a label out of an attribute chain and reports whether it is
rooted in a name. Subscripts are stepped over, not recorded — `x[i].y` is `x_y`,
because a subscript says *where in* the object, not what the object is called. An
unrooted chain (`(a @ b).sum`) yields only the trailing parts.

`wrap` assigns the occurrence number, appends the label to `names`, records the
node's line in `lines`, and returns the `__nnsight_op__(...)` call. It copies the
original node's source location onto the wrapper: without that, an exception raised
inside an instrumented forward would report the offset `increment_lineno` applied
rather than the real line.

`visit_Call` calls `generic_visit` **first**, so arguments — and therefore nested
calls — are numbered before the call containing them. That is what makes the counter
run in execution order, which is the order the interleaver will serve values in. The
wrapper node it returns is not re-visited, so `__nnsight_op__` never counts as an
operation itself. Zero-argument `super()` is returned untouched: it reads `__class__`
and the first argument off its calling frame, and from inside `__nnsight_op__` there
is neither.

`visit_Assign` visits the value before the targets, matching Python's own evaluation
order (`x[f(i)] = g()` runs `g`, then `x`, then `f`), then routes the value through
`bound`. `bound` wraps it in the identity `__nnsight_bind__` under the target's
label, which is what makes a value that is never a call's return — a product, a
slice, a loop's running state — addressable by the name the forward gives it.

The cases `bound` deliberately declines are as load-bearing as the ones it takes:

- **Tuple unpacking, literal RHS only.** `a, b = e1, e2` binds each name its own
  value, so each element is wrapped separately (the tuple is still built before any
  name is bound, so `a, b = b, a` still swaps). `g, h = torch.chunk(x, 2)` has one
  value for two names — there is nothing per-name to bracket, so you get
  `torch_chunk_0` and no `g_0`.
- **Unrooted targets.** `f()[0] = v` has no name to label.
- **Chained assignment.** `a = b = v` binds one value to two names; `visit_Assign`
  wraps only single-target assignments.
- **Augmented assignment.** `x += v` has no `visit_AugAssign`, so it is not an
  operation.

`visit_AnnAssign` behaves like `visit_Assign` for `x: T = v`, and does nothing for a
bare annotation, which evaluates nothing at run time.

## Compiling inside a shell

`compile_source` reads the source through `source_tree`, which parses the **code
object** rather than the function. Given a `functools.wraps` wrapper, `inspect`
follows `__wrapped__` and hands back the decorated function's source instead of the
wrapper's; going through the code object gets the text that actually belongs to the
frame being instrumented.

The rewritten definition is then compiled *nested inside a shell function* whose
parameters are the original code object's `co_freevars`. Recompiled at module level a
free variable would become a global and break; as a parameter of an enclosing
function it compiles as a free variable again, and `instrument` can attach the
original cells. Every function goes through the shell — one with no free variables
simply gets a shell with no parameters.

Two details around it:

- **Decorator lines are dropped** (`definition.decorator_list = []`). `getsourcelines`
  includes them, and the caller has already peeled them and will rebuild them, so
  leaving them would apply each decorator twice.
- **The child code object is lifted by `co_name`, not `func.__name__`.** Under
  `functools.wraps` the wrapper is *renamed* after the function it wraps while its
  `def` line is not, so matching on `__name__` finds the wrong code object or none.

`ast.increment_lineno` shifts the tree to file coordinates before compiling, so
tracebacks through an instrumented forward point into the real file.

`instrument` is the single entry point: it peels decorators, compiles the innermost
function, rebuilds the decorators around the result, and binds `__nnsight_op__` and
`__nnsight_bind__` into its globals. Closure cells are matched **by name**, not by
position — the shell can order `co_freevars` differently from the original. A bound
method is rebuilt from its function and re-bound to the same instance.

## Peeling decorators

`peel_index(wrapper)` answers "which closure cell holds the function this wrapper
decorates?" by parsing the wrapper's own source for the free names it *calls
directly*, and keeping those whose cell holds a Python function. Exactly one
candidate is the decorated function.

Neither of the other two answers is a failure:

- **None** — the wrapper does not call what it closes over. transformers' experts
  wrapper hands `original_forward` to a lookup and calls the result, running a fused
  kernel; peeling it would instrument an eager loop that never executes, leaving
  every operation dead and every request out of order. Instrumented as it is, its
  operations are the dispatch, and drilling into the dispatch reaches the
  implementation that ran.
- **Several** — ambiguous, and treated the same way.

Matching on `__wrapped__` instead would peel the dispatcher too, which is why the
test is what the wrapper *calls*.

`decorator_chain` iterates that, guarding against a closure that calls itself.
`rewrap` rebuilds the chain inside out, giving each wrapper a **fresh** closure
rather than assigning into its existing cell: a wrapper is the *class*'s attribute,
shared by every instance in the process, so mutating its cell in place would redirect
models nobody is tracing.

## Caching

`compiled(func)` memoizes `compile_source` in `FORWARD_CACHE`, keyed on
`func.__code__` — so the second `GPT2MLP` sourced in a process pays nothing, and a
`.source` on a module already sourced is a dict lookup. A callable with no `__code__`
raises `SourceNotAvailable` before the lookup.

Failures are not cached. The assignment into `FORWARD_CACHE` happens only on a
successful return, so a forward whose source cannot be recovered re-attempts (and
re-raises) on every access. That keeps the cache to things that exist, at the cost of
re-parsing a hopeless callable — which nothing in a hot path does.

## Drilling in

`SourceEnvoy.source` and `run_op` negotiate a drill across the greenlet boundary,
because the callee is a run-time value: an attention implementation is a local
variable, and which one it holds depends on the config.

The worker side marks the location requested by writing a `None` placeholder into
`interleaver.sourced`, then parks on `{path}.fn`. The model side checks membership in
`interleaver.sourced` before making the call and, if the location is there, offers
the live `fn` at that location. The worker receives it, rejects a `torch.nn.Module`
(the submodule has its own `.source`) and the `bind` identity (an assignment has no
callee), instruments what is left, and stores `(instrumented, compiled)` back under
the same key. The model side reads the entry again and calls the instrumented copy,
so *its* operations land under `{base}.source.*` — recursively, to any depth.

Two consequences worth knowing when changing this:

- The `.fn` handoff happens **before** the call runs, one step earlier than the
  operation's own `.output`. A worker that reads `op.output` and then asks for
  `op.source` is already late.
- On later fires in the same run the entry is already built and no worker is parked,
  so the `handle` on `{base}.fn` is a no-op and the cached copy is reused. That is
  what makes a drill survive generation steps.

A function that refers to its own name in its body is not drillable: the shell makes
that name a local of the shell, so the self-reference compiles as a free variable
with no matching cell, and `instrument`'s by-name lookup raises `KeyError`. Every
`torch.nn.functional` entry point does this to pass itself to the torch-function
dispatcher, so `nn_functional_softmax_0.source` raises where its `.output` works
fine.

## The user views

`Source` holds an envoy, a `Compiled`, and a path prefix; `__getattr__` turns an
operation name into a `SourceEnvoy` (raising `AttributeError` with the available
names), `__iter__` walks `Compiled.names` in execution order, and `__repr__` renders
the stored source text with `Compiled.lines`.

`SourceEnvoy` builds location strings and delegates. Its `.output`, `.input` and
`.inputs` are the same `eproperty` descriptors `Envoy` uses (see
[extending-envoy.md](extending-envoy.md)), one level finer:

| Access | Location | Callback |
|---|---|---|
| `op.output` | `{path}.output` | identity |
| `op.inputs` | `{path}.input` (`key="input"`) | identity — the whole `(args, kwargs)` |
| `op.input` | `{path}.input` | preprocess takes the first argument; postprocess re-reads the pair and replaces just that argument |
| `op.skip(v)` | `{path}.skip` | `Mediator.skip` |

## Key files

- `src/nnsight/intervention/source.py` — `Instrument`, `compile_source`, `compiled`,
  `peel_index`, `decorator_chain`, `rewrap`, `instrument`, `run_op`, `make_op`,
  `Source`, `SourceEnvoy`.
- `src/nnsight/intervention/envoy.py` — `Envoy.source`.
- `tests/test_source.py`, `tests/test_interleaving.py::TestSourceIteration`.

## Related

- [Source Tracing](../concepts/source-tracing.md) — the mental model.
- [The Controller](controller.md) — `State`, `active()`, the module's own handoffs.
- [interleaver-internals.md](interleaver-internals.md) — `handle` semantics.
