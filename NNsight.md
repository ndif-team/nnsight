# NNsight: Design and Implementation (0.8)

> A manual for people who want to understand what nnsight is, why it works the way
> it does, and how each piece fits together. Read it front to back once for the
> mental model; keep it open as a reference after that.

## Goal of this document

nnsight lets you run a neural network and, in the middle of that run, **read and
edit any internal value** — a layer's output, an attention pattern, a gradient — by
writing ordinary Python as if you already had the value in hand. The same code runs
on a model on your laptop or on a 405B-parameter model hosted remotely.

This document explains the whole system: the problem it solves, the design
principles behind it, and how the layers — tracing, interleaving, the envoy tree,
the feature set, the model wrappers, remote execution — actually work. It favors the
*why* and the mental model over exhaustive API listing; the recipe-style pages under
[`docs/`](docs/) (routed by [`CLAUDE.md`](CLAUDE.md)) are the task reference, and the
source is the final word. Where a section maps to a `docs/` page or a source file,
it says so.

This is 0.8 — the pipeline rewrite. If you knew an older nnsight, see
[What changed in 0.8](#what-changed-in-08) and
[docs/reference/version-history.md](docs/reference/version-history.md).

---

## Table of Contents

1. [Introduction](#1-introduction) — the intervention problem, prior approaches, design principles, what changed in 0.8
2. [The mental model](#2-the-mental-model) — deferred execution, the trace, interleaving, the envoy tree
3. [Tracing](#3-tracing) — capture, parse, execute
4. [Interleaving](#4-interleaving) — the Interleaver, the Mediator, greenlets, the event protocol, batching
5. [The Envoy](#5-the-envoy) — the tree, eproperties, reading and editing values, source tracing, skip, aliasing, dispatch, ad-hoc calls, navigation, `envoys=`
6. [Features](#6-features) — save, generate vs pipe, iteration, edit, skip, gradients, barriers, scan, cache, `tracer.result`, sessions
7. [Modeling](#7-modeling) — the mixins, batching, `TransformersModel`, `DiffusionModel`, `VLLM`, deprecated aliases
8. [Debugging](#8-debugging) — clean tracebacks, DEBUG / `-v`, the errors you'll hit
9. [Remote execution](#9-remote-execution) — NDIF, config, blocking / non-blocking / async, sessions, `remote="local"`
10. [Extending nnsight](#10-extending-nnsight) — subclassing, eproperties, `envoys=`, custom batchers, runtimes
11. [Performance](#11-performance) — the overhead model, best practices, profiling

---

## 1. Introduction

### The intervention problem

Interpreting and steering a neural network means getting *inside* the forward pass:
reading the residual stream at layer 12, zeroing an attention head, adding a steering
vector before the next token, taking the gradient of a loss with respect to a hidden
state. The values you want are transient — they exist for a few microseconds inside
`model(x)` and are gone.

PyTorch's own answer is the **forward hook**: register a callback on a module and it
fires with that module's inputs and outputs. Hooks work, but they don't *compose*.
A hook is a separate function with its own scope; to combine "read layer 5, use it
to edit layer 8, then read the final logits" you juggle callbacks, closures, and
mutable state, all written inside-out relative to the order things actually happen.
And a hook only sees module *boundaries* — to reach a value computed *inside* a
forward method (the attention scores before they're projected) you have to edit the
model's code. None of it survives being sent to a model you don't have locally.

### What nnsight does instead

nnsight lets you write the intervention as **straight-line code in the order it
happens**, against the values as if they were already there:

```python
from nnsight import LanguageModel  # or TransformersModel, the primary wrapper
model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    hidden = model.transformer.h[5].output[0]      # read layer 5's output
    model.transformer.h[8].output[0][:] = hidden   # write it into layer 8
    logits = model.output.logits.save()            # keep the final logits

print(logits.argmax(-1))
```

Inside the `with` block you are not running the model — you are describing what to do
when it runs. `model.transformer.h[5].output` doesn't return a tensor immediately; it
returns a stand-in that *becomes* the real tensor at the moment the model reaches
layer 5. Assigning to `.output` doesn't mutate a local — it schedules a substitution
into the forward pass. `.save()` marks a value to survive past the block. When the
`with` exits, nnsight runs the model and your code **interleaved**, so each read
blocks until the model produces that value and each write lands before the model
reads it.

This is the whole idea: **you write against the values; nnsight arranges for them to
be there.** Three things fall out of it:

- **It composes.** Interventions are ordinary Python — loops, conditionals, function
  calls, intermediate variables. The order you write is the order they run.
- **It reaches inside forwards.** [Source tracing](#54-source-tracing) (`.source`)
  exposes the individual operations of a module's `forward`, not just its boundary.
- **It runs anywhere.** The block is captured as *source code plus the values it
  references*, so the identical trace can be shipped to [NDIF](#9-remote-execution)
  and run on a model far too large for your machine — see
  [remote execution](#9-remote-execution).

### Design principles

Everything downstream follows from a few commitments:

1. **Interventions are code, not callbacks.** The primary interface is a `with`
   block of normal Python read/write against activations. No registration, no
   callback soup.
2. **The same trace runs locally or remotely.** A trace is serializable by
   construction (source + referenced values), so `remote=True` changes *where* it
   runs, not *what* you wrote.
3. **Execution is deferred and interleaved.** The block is captured, then run
   step-for-step with the model. A read parks until the model gets there; a write
   is spliced in. This is what makes straight-line intervention code possible.
4. **Model-agnostic core, batteries on top.** The engine wraps any
   `torch.nn.Module` ([`NNsight`](#7-modeling)); the HuggingFace, diffusers, and
   vLLM wrappers add loading, tokenization, and batching without changing the core.
5. **Get out of the way.** The overhead is per-value-access bookkeeping around the
   model's own compute, which dominates (see [Performance](#11-performance)). You
   pay for what you touch.

### What changed in 0.8

0.8 is a ground-up rewrite around a compile-and-interleave pipeline. If you're
coming from an earlier nnsight, the load-bearing changes:

- **Greenlets, not threads.** Each intervention block runs as a *greenlet* (a
  cooperative coroutine), not an OS thread. The event protocol is
  `VALUE` / `SWAP` / `SKIP` / `BARRIER`. See [Interleaving](#4-interleaving).
- **`TransformersModel` is the primary HuggingFace class.** `LanguageModel` /
  `VisionLanguageModel` still work but are deprecated aliases that warn.
- **`generate` returns token ids; `pipe` returns the pipeline's records.** The old
  `generate`-returns-decoded-text behavior is now `pipe`.
- **Hookable values are `eproperty` descriptors** (`.input`/`.output`/`tracer.result`
  and your own), and per-module custom envoy classes come back via `envoys=`.
- **Remote gains an async backend** (`await` / `async for` a job); the old hybrid
  `tracer.local()` streaming is gone.
- **`save()` raises outside a trace** (it used to be a silent no-op), and the idiom
  for collecting values is to **save the container**, not each element.

The full old→new delta is in
[docs/reference/version-history.md](docs/reference/version-history.md).

---

## 2. The mental model

Hold four ideas and the rest of the document is detail.

**1. A trace is captured, not run.** `with model.trace(x):` does not execute its body
line by line. nnsight grabs the block's source (as an AST), compiles it, and sets it
aside. Nothing in the body runs against real tensors yet. This is *deferred
execution* — see [Tracing](#3-tracing).

**2. The model and your block run interleaved.** On exit, nnsight starts the model's
forward pass and your captured block *at the same time*, in one thread, handing
control back and forth. Your block runs until it asks for a value the model hasn't
produced (`model.transformer.h[5].output`); it parks there. The model runs until it
reaches that module; it hands the value over and your block resumes. This is
*interleaving*, and the machinery is a shared [`Interleaver`](#41-the-interleaver)
plus one [`Mediator`](#42-the-mediator) greenlet per block. See
[Interleaving](#4-interleaving).

**3. You address the model through an envoy tree.** `model.transformer.h[5]` is not
the module — it's an [`Envoy`](#5-the-envoy), a lightweight mirror of the module tree.
Every envoy exposes `.input` / `.output` (and `.source`, `.skip`, gradients) as the
hooks into the run. Reading one parks your block; assigning to one schedules a swap.
The tree is built once when you load the model and mirrors the real modules exactly.

**4. Values leave the block only if you `save()` them.** The block runs in a scratch
namespace; when it's done, only values you marked with `.save()` (or
`nnsight.save(x)`) are pushed back to your frame. Everything else is discarded with
the trace. See [Saving values](#61-saving-values).

Put together: you *describe* interventions against an envoy tree, nnsight *captures*
that description, then *interleaves* it with the model, and hands back what you
*saved*. Local or remote, that's the shape of every nnsight program.

---
## 3. Tracing

The intervention block you write inside `with model.trace(...):` never runs where
it stands. When Python reaches that line, nnsight steps in, reads the block's
source, sets the body aside, and arranges for the body to run later — [interleaved
with the model's forward pass](#4-interleaving). *Tracing* is the machinery that
performs that sleight of hand: it turns a `with` block into a code object nnsight
can run on its own terms.

The reason it works this way is the whole premise of the library. To let you write
`hidden = model.transformer.h[5].output` as if the value were already in hand, the
line can't execute when Python gets to it — the model hasn't run yet, so there is no
value. So the body is *captured* rather than executed, then handed to a **backend**
that decides where and how it runs. The base backend runs it in place; a remote
backend ships it to [NDIF](#9-remote-execution). The tracer doesn't know the
difference — swapping the backend is the whole seam between a local run and a remote
one.

The pipeline is deliberately small and layered. Everything in this section lives in
`src/nnsight/tracing/` — a self-contained "capture a `with` block and run it through
a backend" library that imports no torch and knows nothing of models or
interventions. The intervention-specific behavior (running the body against a live
model) is one overridden method, `execute`, in
`src/nnsight/intervention/tracer.py`. The base `Tracer`
(`src/nnsight/tracing/tracer.py`) does five steps, each overridable: **capture**,
**parse**, **build**, **compile**, **execute**. Below they group into three: capture
(grabbing the block), parse (turning it into code, via build and compile), and
execute (running it through the backend). For the recipe-level view see
[docs/concepts/deferred-execution.md](docs/concepts/deferred-execution.md) and
[docs/developing/tracing-pipeline.md](docs/developing/tracing-pipeline.md).

### 3.1 Capture

`model.trace("The Eiffel Tower is in")` constructs a tracer and stores the call
arguments; nothing is read yet. The work starts when the `with` block *enters*.
`Tracer.__enter__` calls `capture()`, which looks two frames up (`sys._getframe(2)`,
past `__enter__`) to find the user's frame — the one that ran the `with` statement.
From that frame it has everything it needs: the filename, the line number of the
`with`, and the live globals and locals the block was written against.

Capture reads the source text of that file (`Tracer.source`) and parses out the
`with` node at that line, then compiles the block's body into a standalone code
object. The parse and compile are pure functions of *where the trace sits in the
code*, so their results are memoized per call site in a process-wide cache, `BLOCKS`
(`src/nnsight/tracing/globals.py`), keyed by
`(filename, lineno, co_name, co_firstlineno)`:

```python
key = (code.co_filename, frame.f_lineno, code.co_name, code.co_firstlineno)
if key not in BLOCKS:
    source = self.source(code.co_filename)
    node = self.parse(source, frame.f_lineno)
    compiled = None if node is None else self.compile(self.build(node), frame)
    BLOCKS[key] = (node, compiled)
```

A `with model.trace(...)` inside a Python loop is therefore read, parsed, and
compiled exactly once — every later iteration is a dict lookup. A site that turns
out *not* to be a `with` block caches `(None, None)`, so even that negative verdict
isn't re-derived on each call (it lets `capture` raise `WithBlockNotFoundError`, the
signal a plain `envoy.method(...)` call uses to fall back to running normally).
Source text is cached separately in `SOURCES`, read once per file and never
re-validated — a file edited mid-run is traced as it was first seen, and since the
cache key includes the line number, ordinary edits invalidate on reload anyway.

Source lookup handles three contexts: ordinary files and IPython/Jupyter cells
(both via `linecache` — IPython registers each cell there under the frame's
filename), and `python -c "<code>"` programs (recovered from `sys.orig_argv`).
Anything with no reachable source — a raw `exec` string, a heredoc, `<stdin>` —
yields empty text, which fails the parse and raises `WithBlockNotFoundError`. The
practical consequence: `with model.trace(...):` reads *its own source*, so it can't
be run from a one-liner or an `exec`'d string that isn't itself the program. Write
snippets to a real `.py` file.

**Skipping the body inline.** Capturing the source isn't enough; the body must also
be prevented from running where it sits. Still in `__enter__`, if the block has real
code (`skippable` — a body that is only `pass`, a docstring, or `...` is left to run
harmlessly), `skip_context` arms a per-frame trace hook (`frame.f_trace`) that
raises `ExitTracingException` the instant execution reaches the body's first line.
`ExitTracingException` is a control-flow signal, not an error — it's caught in
`__exit__` and never surfaces to you. Because Python delivers trace events per
*line*, one constraint falls out: the body must start on its own line. A one-liner
like `with model.trace(x): y = model.output.save()` is refused with a clear
`ValueError`, because a body written on the `with` line would run where it stands
*and* again through the backend. A multi-line `with` header (arguments spread over
several lines) is fine — the hook stays armed until it reaches the body, so the `as`
target binds before the block is skipped.

### 3.2 Parse

Parsing finds the `ast.With` (or `ast.AsyncWith`) node that starts on the trace line
and turns its body into a compilable code object. Parsing a whole source file is
`O(its AST)` and dominates a cold capture, so `parse` first tries `_parse_block`,
which slices *just* the block out of the source — the (possibly multi-line) header
and its body, bounded by indentation and open-bracket depth — dedents it to column
0, parses that alone, and shifts the line numbers back so tracebacks and the skip
hook still point at the real file. Getting the slice bound wrong is safe by
construction: collecting too much just parses a few trailing statements (the `with`
is still `body[0]`); collecting too little makes the slice unparseable, which
returns `None` and falls back to parsing the whole file. It never returns a wrong
node.

From the node, two more steps produce the code object. `build` wraps the block's
*body* — not the `with` line itself — in an `ast.Module` and backfills line/column
info with `fix_missing_locations`. `compile` compiles that module under the original
frame's filename and sets its `co_name` to the frame's name, so a traceback from
inside the body reads as if it ran where it was written. The `(node, compiled)` pair
is what lands in the `BLOCKS` cache. The node is kept because it's needed again for
[remote serialization](#9-remote-execution): a block that rides to a server is
reduced to *source plus the variables it references*, cross-version-safe, rather
than shipping a compiled code object.

### 3.3 Execute

When Python unwinds the `with`, `__exit__` runs. For the expected
`ExitTracingException` (the body was skipped) — or a clean exit, when there was
nothing to skip — it invokes the backend, which is the one line that says what "run
the block" means:

```python
class Backend:
    def __call__(self, tracer: Tracer) -> None:
        tracer.execute(tracer.info.code)
```

So the backend is just a dispatch seam; the behavior lives in `execute`. The base
`Tracer.execute` runs the compiled body against a scratch namespace and pushes the
results back:

```python
scope = Scope(dict(frame.f_locals), frame.f_locals, frame.f_globals)
exec(code, scope)
push_result(frame, scope)
```

The block runs in a `Scope` (`src/nnsight/tracing/util.py`), not directly in your
frame, because it runs later than the line it was written on — and, once
interleaving, in a greenlet somewhere else entirely. A `Scope` resolves names from
three places in order: a *snapshot* of the frame's locals taken at capture (so a
`for prompt in prompts:` loop variable means what it meant where the block was
written, even though the loop has moved on by the time the block runs), the *live*
locals of the frame (so a name bound by a sibling `tracer.invoke(...)` block is
visible), and the globals by fallback. Assignments made in the block land in the
scope, so they survive to be pushed back; `push_result` then filters them — see
[Saving values](#61-saving-values) for why only `.save()`-marked values leave the
outermost trace, while a nested trace pushes everything up to its enclosing block.

`InterleavingTracer.execute` (`src/nnsight/intervention/tracer.py`) overrides this
to run the body *interleaved with the model's forward pass* instead of plainly. It
builds a [`Batcher`](#44-batching-narrow-and-widen) and one or more
[`Mediator`](#42-the-mediator) workers, then hands them to `Envoy.interleave` to run
alongside `fn(*input)` (the model's `__call__`, or `generate`, ...). Which shape it
takes depends on whether `trace()` itself got input:

- **Direct-input mode** (`trace(x)`) — the whole block is one implicit invoke over
  `x`. `execute` builds a single `Mediator` for the entire body, gives it the
  input's batch group, and runs it.
- **Invoke mode** (`trace()` with no input) — the block is run *now*, once, purely
  to collect its `with tracer.invoke(...)` sub-blocks. Each invoke captures its own
  body and registers its own input and worker (see
  [Batching](#44-batching-narrow-and-widen) and [§7.2](#72-batching)); their inputs
  are then combined into one batched forward. If the body registers no invokes and
  `trace()` had no input, that's an error — `trace()` needs an input or at least one
  invoke block.

`ScanningTracer` (behind `model.scan(...)`) is the same machinery with one wrapper:
it defers to `InterleavingTracer.execute` inside a `FakeTensorMode`, so the forward
propagates only tensor *metadata* — shape, dtype, device — with no real compute and
no weights loaded. This lets you check activation shapes on an undispatched model
without running it for real. The values read inside a scan are fake tensors, valid
only within the block: read their `.shape`/`.dtype` there, but a fake tensor saved
out of the scan is unusable once the fake mode exits. Scan and trace share the same
`_batch_size`/`_batch` preprocessing, so a string prompt is tokenized and invokes
are batched exactly as in a real trace. See [docs/usage/scan.md](docs/usage/scan.md).

Whichever tracer runs, `execute` ends by pushing each worker's saved values back
into your frame and clearing the interleaver so the next run starts clean. If the
backend raises, `__exit__` re-raises with a cleaned traceback that drops nnsight's
own frames, leaving your code — see [Debugging](#8-debugging). Any exception from the
body that *isn't* the skip signal propagates normally.

## 4. Interleaving

Interleaving is the heart of nnsight — the part that makes reading and editing
activations from straight-line code actually work. The problem it solves: your
intervention code and the model's forward pass have to run *in lockstep*. Your code
pauses whenever it asks for a value the model hasn't produced yet; the model runs
until it reaches that value, hands it over, and your code resumes — possibly editing
the value on the way back in.

nnsight implements this with **greenlets** (cooperative, single-threaded
coroutines), not OS threads. A worker and the model take strict turns on one thread,
so there are no locks, no queues, and no races — only control handed back and forth
by explicit switches. This is why the mental model is so clean: at any instant
exactly one of {the model, one worker} is running, and a worker only ever runs
between the two model events it cares about. Everything in this section lives in
`src/nnsight/intervention/interleaver.py` (plus `batching.py` and `barrier.py`); the
concept pages are
[docs/concepts/interleaver-and-hooks.md](docs/concepts/interleaver-and-hooks.md) and
[docs/concepts/threading-and-mediators.md](docs/concepts/threading-and-mediators.md).

### 4.1 The Interleaver

An `Interleaver` drives the model side. One is shared across an entire
[`Envoy`](#5-the-envoy) tree, so every module in the model reports into the same set
of workers. It owns two things: the per-module controllers that turn a forward
pass into a stream of events, and the list of `Mediator` workers those events feed.

**One controller per module.** When an envoy is built, the interleaver's
`instrument` method installs a *controller* as that module's `forward` — once, at
construction time, for the model's lifetime. On every call the controller hands
the module's `(args, kwargs)` through `handle("{path}.input", ...)`, consults the
`.skip` gate, runs the real forward, and hands the output through
`handle("{path}.output", ...)`. Because both handoffs *return* the value they
handle, an intervention can edit the input or the output in place. `.inputs`
exposes the full argument pair, `.input` the first argument. There are no PyTorch
forward hooks — that is deliberate: a module with no hooks is called on PyTorch's
fast path, and the controller costs one frame and one check when no trace is
running. Under transformers tensor parallelism the controller runs inside the
model's own collective hooks, so it sees a partial sum or a shard there, which
`TPFragments` makes whole only when a worker is waiting for it.

The single most important property of these hooks is that they **pass through when
idle**. The first line of each is `if not self.interleaving: return None`, and
`interleaving` is a flag flipped on only between the interleaver's `__enter__` and
`__exit__`. Outside a trace the hooks return `None` — PyTorch's signal for "no
change" — and the module runs exactly as if they weren't there. So an instrumented
model runs at normal speed whenever you aren't tracing; the cost when idle is one
short-circuiting `if` per hook, nothing more. There is no lazy install, no sentinel,
no removing and re-adding hooks around each run: the hooks are permanent and gated on
a flag.

`instrument` also installs the per-module **source/skip controller** (see
[Source tracing](#54-source-tracing) and
[docs/developing/hook-system.md](docs/developing/hook-system.md)), which replaces the
module's `forward` to add the `.skip` gate and, on demand, operation-level access.
The controller is registered up front so it's in place before `nn.Module.__call__`
binds `forward` — necessary because a skip's replacement can be read from the
module's own input first. `instrument` runs again on dispatch (`Envoy._update`, when
meta weights are swapped for real ones): it drops the old module's hooks and
re-installs on the new one.

**`handle`: one call, every worker.** Everything the model side does routes through
`Interleaver.handle(provider, value)`. It offers `value` at that location to every
mediator in `mediators` order (so if two invokes both edit the same location, invoke
0's edit lands before invoke 1's — definition order); each worker either reads it,
edits it, or ignores it, and the possibly-edited value threads through to the next.
Afterward, the post-intervention value is offered to any active
[`tracer.cache()`](#6-features) observers, narrowed to each cache's own batch rows,
so a cache records exactly what interventions produced. The edited value returns to
the hook, which substitutes it back into the forward.

That one primitive — a location string plus `handle` — carries *everything*, not
just module boundaries. The model's return value isn't produced by any module hook,
so `Envoy.interleave` calls `handle("result", result)` after the forward to serve
anything parked on [`tracer.result`](#6-features). The `.skip` gate is
`handle("{path}.skip", ...)`. Source operations inside a forward are
`handle("{path}.source.<op>.output", ...)`. A custom runtime that computes a value
PyTorch never surfaces (vLLM's logits) plumbs it in the same way. There is no
separate event type for any of these — they're all the one primitive, which is why
[extending nnsight](#10-extending-nnsight) with a new hookable value means adding a
*location*, not a hook.

One interleaver persists across many runs. `__enter__` flips `interleaving` on and
starts every not-yet-started worker; `__exit__` flips it back off (swallowing an
intentional `EarlyStopException` from `tracer.stop()`); `cancel` then clears the
workers and the batcher so the next run starts clean. The hooks themselves are never
touched — a server reuses the same interleaver, request after request.

### 4.2 The Mediator

Each block of intervention code becomes one `Mediator` — the body of a direct-input
`trace(...)`, or one `with tracer.invoke(...)`, or one registered
[edit](#6-features). A mediator wraps the captured block and runs it inside its own
greenlet, the **worker**. The worker drives the interaction: it runs the block until
the code asks for a value, then *parks* — recording what it's waiting for in
`pending` and switching control back to the parent greenlet (the model side). The
parent later resumes it once the model reaches the awaited location.

The mediator carries the block's compiled `code` and the `Scope` (`lcls`) it runs
against — its capture-time names, the frame it shares with sibling blocks, and the
globals behind them. This scope doubles as what `push_result` reads the block's
saved values back out of when the run finishes. A mediator built for an
[edit](#6-features) runs with `copy=True`, so it execs against a fresh copy of its
scope on every replay and doesn't accumulate the last run's names.

`start` creates the worker greenlet, stashes a weakref back to the mediator on it
(so intervention code can find *its own* mediator via `getcurrent().mediator()` — the
mechanism behind `tracer.iter` and `tracer.barrier()`), and switches in, running the
block up to its first park. A worker's whole life is visible through `alive`, which
is `True` only while the worker exists and still has code left to run — `False`
before `start` and, crucially, `False` after the block finishes, because a greenlet
is falsy once it has run to completion. There is no "worker done" event; a finished
worker is just an `alive == False` mediator.

Two methods move control across the greenlet boundary, and they are exact
counterparts. On the worker side, `event` switches to the parent handing over an
event tuple and blocks until a value is switched back — this is what every park call
bottoms out in. On the parent side, `switch` resumes the worker with a value and
returns whatever the worker parks on next, or `None` when the worker finishes. If
the worker raises, `switch` stashes a clean, intervention-only traceback on the
exception (as `__intervention_tb__`) *before* the re-raise unwinds the model and hook
frames on top of it, so [Debugging](#8-debugging) can show you your own code rather
than nnsight's plumbing — then the exception propagates and halts the run.

### 4.3 The event protocol

A worker parks by switching a tuple `(Event, location, ...)` to its parent. There
are exactly four kinds of event, and their one-line contracts are the whole
vocabulary of interleaving:

| Event | Raised by | Means |
|---|---|---|
| `VALUE` | `Mediator.value(loc)` | Read the value at `loc`; park until the model reaches it. |
| `SWAP` | `Mediator.swap(loc, v)` | Replace the value at `loc` with `v` on the model's way past. |
| `SKIP` | `Mediator.skip(loc, v)` | Skip the computation gated at `loc`, using `v` as its result. |
| `BARRIER` | `Mediator.barrier()` | Wait for the other blocks; names no location. |

The [`Envoy`](#5-the-envoy) properties are thin wrappers over these: reading
`envoy.output` is `Mediator.value("{path}.output")`; `envoy.output = x` is
`Mediator.swap("{path}.output", x)`; `envoy.skip(v)` is `Mediator.skip`. `BARRIER`
is the odd one out — it names nothing the model produces, so the model side never
serves it; another worker does, on its way past the same barrier.

There is no `END` event and no `EXCEPTION` event. A worker finishing is just its
greenlet running to completion (`switch` returns `None`); an error is just an
exception propagating out of `switch`. The protocol is only these four requests.

**Park and switch, concretely.** A read illustrates the full cycle. The worker runs
to `hidden = model.transformer.h[5].output`, which calls `Mediator.value(...)`; that
switches `(Event.VALUE, "model.transformer.h.5.output.i0")` to the parent and
blocks. Control is now on the model side, which runs the forward until layer 5's
controller hands its output to `Interleaver.handle`, which calls the worker's `handle` for that
location. The worker's pending event matches, so `handle` switches the value into
the worker, which resumes with the tensor in `hidden`, runs to its next park (or
finishes), and hands control back. A write is the same shape: `swap` parks the same
way, but when the model reaches the location `handle` substitutes the worker's value
for what the model produced before resuming the forward. A worker can read *then*
write the same location — `handle` loops while the worker keeps parking on the same
location, so a read followed by an assignment to `.output` both drain on one visit
before the model moves on.

**Occurrence tags (`.i{n}`).** A location can be reached many times in one run —
every step of a generation loop revisits every module. Each park carries the
occurrence the worker wants, appended to the location as `.i{n}`. With no
`tracer.iter`, `n` is always `0`, so every request binds to the *first* visit — the
plain single-forward behavior. `tracer.iter[k]` pins the worker to occurrence `k`;
its request tagged `.i{k}` simply doesn't string-match earlier visits and waits
while they pass by, binding on the k-th. `handle` tracks per-location visit counts
(`iterations`) and tags each visit accordingly, so matching is a single string
comparison with no numeric bookkeeping in the hot path. (After the first hit of a
pinned non-zero step, the worker *relaxes* so the rest of that step's requests follow
the model sequentially rather than re-forcing the index.) Because a source operation
also goes through `handle` every time it fires, an op inside a loop advances its own
occurrence counter per fire while a module advances once per forward — the iteration
model falls out of the one primitive, with no extra counters. See
[Features](#6-features) for `tracer.iter` / `tracer.all()`.

**Errors and dangling workers.** Within one worker, requests happen in execution
order and the model runs in forward order, so asking for an *earlier* location after
a *later* one is impossible to satisfy — the earlier module has already run past, and
its next visit will never come. This surfaces as `OutOfOrderError`. It's caught after
the model returns, by `check_dangling_mediators`, which inspects any worker still
parked:

- A plain request (`iteration == 0`) for a location the model ran past or never
  reached is a real error: `OutOfOrderError` is thrown *into* the worker, so the
  traceback points at the exact line that was waiting.
- A request inside an open-ended `tracer.iter[:]` loop that outran the model's steps
  is *expected*: the error is thrown to unwind the worker (running its `finally`
  blocks), then caught and turned into a warning. Values from steps that did run are
  already saved. This is why an unbounded `iter[:]` / `all()` discards everything
  after the loop — bound the loop (`iter[:N]`) to keep trailing code.
- A `BARRIER` still pending means fewer blocks reached the barrier than its count —
  a `ValueError` points at the waiting line.

The common practical version: read a module's `.input` *before* its `.output`, and
to access modules in a different order use a separate invoke (a separate worker over
the same forward). `tracer.barrier(n)` is the tool for the cross-worker case — a
meeting point `n` blocks agree on, where the last to arrive releases the rest, so a
value read in one invoke is guaranteed written into another only after the read. See
[Barriers in Features](#6-features) and [§7.2](#72-batching).

### 4.4 Batching: narrow and widen

A single `with model.trace() as tracer:` can hold several `with tracer.invoke(x):`
blocks, whose inputs are combined into one batched forward while each block's
interventions see only *its* rows of every activation. Each invoke is one worker,
scoped to a `batch_group` — a `[start, size]` row range in the combined batch. The
per-trace `Batcher` (`src/nnsight/intervention/batching.py`) collects the invokes,
assigns each its group, and does the row math at run time.

The scoping is two mirror operations, driven from the worker's `handle`:

- **On a read**, `Batcher.narrow(value, group)` slices every batched tensor down to
  the worker's rows before serving it — so an invoke over one prompt sees
  `output.shape[0] == 1` even though the real forward ran a batch of three. A tensor
  counts as batched only when its leading dim equals the combined batch size, so
  activations whose dim 0 is sequence length or hidden size pass through untouched.
- **On a write**, `Batcher.widen(full, group, edited)` splices the worker's edited
  rows back into the full batch — via `torch.cat` rather than in-place assignment, to
  keep autograd correct for leaf and view tensors and to avoid aliasing when the
  edit is itself a narrowed view of the full tensor.

Narrowing and widening only actually happen when **two or more** non-empty invokes
contribute rows (`Batcher.batching`). A lone invoke *is* the whole batch, so single-
input traces pay no slicing overhead. An **empty** invoke (`tracer.invoke()` with no
args) has no batch group and sees the whole combined batch — its own worker over
every row, useful for logic that spans all invokes.

The row math here is dim-0 only; equalizing everything else (padding shorter prompts
to a common sequence length, building a combined attention mask) is the model's job,
in `_batch`, when it assembles the combined input. That model side — `_batch_size`
and `_batch`, the `TransformersModel` implementation, and custom batch layouts like
diffusion's classifier-free-guidance doubling or vLLM's flat-token axis — is covered
in [Batching](#72-batching) under [Modeling](#7-modeling); the concept page is
[docs/concepts/batching-and-invokers.md](docs/concepts/batching-and-invokers.md) and
the internals are in
[docs/developing/batching-internals.md](docs/developing/batching-internals.md).

---

## 5. The Envoy

An `Envoy` is how you *point at* a module. When you write
`model.transformer.h[5].mlp`, you are not holding the `torch.nn.Module` — you are
holding a lightweight mirror of it whose job is to give that module a stable
address (`"model.transformer.h.5.mlp"`) and to expose its live values during a
run. Everything you do inside a trace goes through an envoy: reading `.output`,
overwriting `.input`, reaching an operation inside the forward with `.source`,
skipping the module, taking a gradient. The envoy is the surface; the machinery
underneath is the [Interleaver and the Mediator](#4-interleaving).

The design is deliberately thin. An envoy owns almost no state of its own — a
path, a reference to the wrapped module, a shared interleaver, and a list of
children. The interesting behavior lives in a handful of *descriptors*
([eproperties](#52-eproperties-how-values-are-hooked)) that turn attribute access
into interleaver traffic, and in a per-module controller that
[source tracing](#54-source-tracing) and [skipping](#55-skipping-a-module) share.
Read this section to understand what the tree is, how `.input`/`.output` actually
hook a value, and the smaller powers — source, skip, rename, dispatch, ad-hoc
calls — that hang off the same tree.

Source: `src/nnsight/intervention/envoy.py`, `src/nnsight/intervention/eproperty.py`,
`src/nnsight/intervention/source.py`. Concept pages:
[docs/concepts/envoy.md](docs/concepts/envoy.md),
[docs/concepts/source-tracing.md](docs/concepts/source-tracing.md).

### 5.1 The envoy tree

An `Envoy` wraps one `torch.nn.Module` and mirrors its submodule tree. When you
construct one — directly, or (far more often) through `NNsight(module)` or a wrapper
like `TransformersModel` — it walks `module.named_children()` and builds a child
envoy for each submodule, recursively. Every module in the model therefore has a
matching envoy reachable by the *same attribute path* you'd use on the module
itself:

```python
from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True)

model.transformer.h[0].mlp        # the Envoy for that GPT-2 block's MLP
model.transformer.h[0].mlp.path   # 'model.transformer.h.0.mlp'
```

The **root envoy is the model**. `NNsight(module)` is nothing more than a root
`Envoy` given the conventional name `"model"` for its path; `TransformersModel`,
`DiffusionModel`, and `VLLM` are `Envoy` subclasses that add loading and
tokenization on top of the identical tree. So `model` itself is an envoy, `model.output`
is the whole model's output, and the children fan out below it.

Each envoy holds the module it mirrors as `._module`. This is the escape hatch to
the real PyTorch object — its parameters, its `state_dict`, its class — and it is
what attribute access falls through to: if you ask an envoy for something it doesn't
define, it looks on `._module` (and, if that yields a submodule not yet mirrored,
wraps it as a child on the spot). A `ModuleList` mirrors as an indexable, iterable
envoy, so `model.transformer.h[0]` and `for block in model.transformer.h:` both
work.

Children are stored in `._children` in module order, and the tree is built once, at
construction. It does not rebuild on every trace — it is a fixed structure that the
run flows *through*. (One exception: reassigning a module through the envoy, e.g.
`model.transformer.h[0].adapter = MyAdapter()`, registers the new module and wraps
it as a child; see [Ad-hoc module calls](#58-ad-hoc-module-calls).)

### 5.2 eproperties: how values are hooked

`.input`, `.inputs`, and `.output` look like plain attributes, but they are
**eproperties** — a small subclass of Python's `property`
(`src/nnsight/intervention/eproperty.py`). This one descriptor is the read/write
plumbing behind essentially every value nnsight exposes: module input and output,
a [source op's](#54-source-tracing) input and output, the run's
[`tracer.result`](#6-features), a runtime's `.logits`. Understand the eproperty and
you understand how any hookable value works.

An eproperty is bound to a **location string**, `"{host.path}.{key}"` — so
`model.transformer.h[0].output` reads the location
`"model.transformer.h.0.output"`. Reading and writing the attribute are translated
into interleaver traffic at that location:

- **Reading** calls `Mediator.value(location)`. Your block's greenlet parks until
  the model reaches that location and produces its value (see
  [the Mediator](#42-the-mediator)).
- **Writing** calls `Mediator.swap(location, value)` — the model, when it reaches
  the location, substitutes your value and continues with it.

Around that raw traffic, the eproperty runs up to three callbacks, and knowing which
is which is the whole mental model:

- **preprocess** — *the decorated stub itself.* It takes the raw value the
  interleaver served and returns what you read. The base `.output` is an identity
  view (`def output(self, value): return value`); `.input` is a preprocess that
  digs the first argument out of the served `(args, kwargs)`.
- **`.postprocess`** — the write side, run on the value you assign *before* it is
  swapped in. `.input` uses it to repack your lone first argument back into the full
  `(args, kwargs)` the module expects.
- **`.transform`** — the write-back for an *edited view*. When a preprocess hands
  back a reshaped or sliced view of the served value and you edit that view in
  place, whether the edit is visible to the model depends on the kind of view. An
  **aliasing** view — one that shares storage with the original, like
  `.view(...).transpose(...)` — propagates in-place edits with no transform needed.
  A **computed or copying** view — a per-head split that reshapes into new storage,
  say — does not: your edits land on a copy the model never sees, so the eproperty
  needs a `.transform` that maps the edited view back to the model's layout. It
  fires once, after the block is done with that read, and its result is spliced in
  as if you had swapped it.
- **`.provide`** — the *serve* side, used from the model/driver rather than the
  block. It hands a value to the interleaver at the eproperty's location so a worker
  parked there resumes with it. This is how values that aren't module outputs get
  served (the tracer serving `result`, a runtime serving `logits`); see
  [Extending nnsight](#10-extending-nnsight).

Here are the actual definitions, verbatim in spirit, of the three you use daily:

```python
@eproperty(key="input")
def inputs(self, value):                 # identity view of the whole (args, kwargs)
    return value

@eproperty
def input(self, value):                  # first-argument view of the same location
    args, kwargs = value
    return first_input(args, kwargs)

@input.postprocess                       # write side: repack the lone first arg
def input(self, value):
    args, kwargs = Mediator.value(f"{self.path}.input")
    return replace_first_input(args, kwargs, value)

@eproperty
def output(self, value):                 # identity view of the module's output
    return value
```

Note that `.input` and `.inputs` deliberately **share the key `"input"`** — both
address `"{path}.input"`. They are two views of the same served value: `.inputs`
gives you the full `(args, kwargs)`, `.input` the first argument. The `key=`
argument exists precisely so several eproperties can offer different views of one
location.

The `description=` argument does one thing: it surfaces the eproperty as its own
line in the `Envoy` repr, so a special hookable value (a model's `.logits`) shows up
in the printed tree. The plain `.input`/`.output` carry no description and stay
hidden. Accessing any eproperty **outside a trace** raises, because the mediator has
nothing to serve:

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
```

Full treatment of defining your own is in
[Extending nnsight](#10-extending-nnsight) and
[docs/developing/extending-envoy.md](docs/developing/extending-envoy.md).

### 5.3 Reading and editing values

Reading an envoy's `.output` inside a trace parks your block until the model reaches
that module, then hands you the *real* runtime object — not a proxy. There is
nothing to unwrap: `print`, `.shape`, `.mean()`, `.clone()` all work on it directly,
because the worker greenlet genuinely receives that tensor (see
[Interleaving](#4-interleaving)). To keep a value past the block, mark it with
`.save()` — and when you're collecting many values, save the *container* and append
raw reads into it:

```python
with model.trace("Hello world"):
    hidden = model.transformer.h[-1].output.save()   # keep one value

    acts = nnsight.save([])                           # save the container
    for block in model.transformer.h:
        acts.append(block.output)                     # append raw reads
```

See [Saving values](#61-saving-values) for why the container idiom is the correct
one and where a bare `.save()` silently drops a value.

There are two ways to change a value, and the distinction matters:

```python
# IN-PLACE: mutate the tensor the model is holding. Later reads of the same
# location see the change.
model.transformer.h[0].output[:] = 0

# REPLACEMENT: hand the model a new object (fires the eproperty setter -> a SWAP).
# Downstream computation continues with the new value.
model.transformer.h[0].output = my_new_tensor
```

In-place editing works because the base `.output` preprocess returns the live
tensor and the controller returns whatever the worker left it as; replacement
works because assigning to the eproperty fires its setter, which issues a swap.

**Know the shape of what you're editing.** A module's `.output` is exactly the object
its `forward` returns. In current `transformers`, a GPT-2 *block*'s `.output` is a
plain `Tensor` of shape `[batch, seq, hidden]` — read and write the whole tensor, no
`[0]` indexing. An *attention* submodule, by contrast, returns a **tuple**
`(attn_out, ...)`. Don't assume; check with `print(module.source)` (which shows the
forward and what it returns) or a saved `.shape`:

```python
with model.trace("Hello world"):
    block = model.transformer.h[0].output          # a Tensor [1, 2, 768]
    attn  = model.transformer.h[0].attn.output     # a tuple; attn[0] is the tensor
    attn_out = attn[0].save()

    model.transformer.h[0].output[:] = 0           # edit the block tensor in place
    model.transformer.h[0].attn.output[0][:] = 0   # edit the first tuple element
```

`.input` gives the module's first forward input (first positional, else first
keyword); `.inputs` gives the full `(args, kwargs)` pair. Writing `.input` repacks
correctly into the `(args, kwargs)` the model wants.

**Gotcha — tuple-element views across a barrier can segfault.** Reaching into a
tuple element and mutating a narrowed view of it across a
[barrier](#6-features) can crash. The safe move is to assign a modified tensor back
rather than relying on the in-place edit of a shared view: build the tensor you want
and set `.output` (or the tuple) to it. Reading `.output` again after an in-place
edit returns the *modified* value, so `.clone()` first if you need a pre-edit copy.

Finally, order matters: within one invoke you must read locations in **execution
order**. Asking for an earlier module's output after a later one has already run
raises `OutOfOrderError` once the model finishes. To touch modules out of order,
use separate invokes. Full detail in
[docs/usage/access-and-modify.md](docs/usage/access-and-modify.md).

### 5.4 Source tracing

Module `.input`/`.output` are the only two locations the forward *hooks* surface.
The individual operations *inside* a `forward` — an activation function, a
`torch.matmul`, an attention call — are invisible to hooks, because an operation
isn't a submodule. `.source` makes them observable, editable, and skippable.

`envoy.source` returns a `Source`: the module's `forward` decomposed into its
operations. Under the hood, the first time you touch `.source` the module's forward
is AST-instrumented — every call `fn(*args, **kwargs)` is rewritten to run through
the same interleaver `handle` that modules use, one level finer. The instrumentation
is lazy and permanent, and completely inert outside a trace, so ordinary inference
is unaffected. (The mechanism is described in [Source tracing](#54-source-tracing)'s
companion internals doc; see below.)

You discover operations by printing `.source`, which works outside a trace and shows
the forward with each op labelled at its call site:

```python
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

Operation names are the **full dotted callee joined with `_`**, plus a per-name
occurrence index in execution order: `self.c_fc(...)` → `self_c_fc_0`,
`torch.relu(...)` → `torch_relu_0`, a bare `dropout(...)` → `dropout_0`. Nested calls
run inner-first, so `f(f(x))` numbers the inner `f` as `f_0`. You rarely memorize
names — `print(module.source)` is the map, and a wrong name raises an
`AttributeError` listing the available ops.

Inside a trace, each operation exposes the same handles a module does:

```python
with model.trace("Hello world"):
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()  # read
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0     # edit
```

Two classes back this. `Source` is the whole-forward view (`envoy.source`); indexing
it by an op name yields a `SourceEnvoy`, the single-operation view. A `SourceEnvoy`
is the operation-level analogue of an `Envoy`: its `.output`, `.input`, and
`.inputs` are the *same* eproperties ([§5.2](#52-eproperties-how-values-are-hooked)),
keyed on the op's path, and it has its own `.skip(replacement)`.

**Recursive `.source`** drills into the function an operation *calls*, exposing its
operations one level deeper — and works **only inside a trace**, because the call
target (often a local, like an attention implementation) is resolved from the live
value flowing through the call:

```python
with model.trace("Hello world"):
    attn  = model.transformer.h[0].attn.source
    inner = attn.attention_interface_0.source            # drill into the call
    scores = inner.attn_output_transpose_0.output.save() # an op inside it
```

Drilling into a call that is itself a *submodule* is refused (call `.source` on that
submodule directly); so are builtins/C functions, closures, and decorated forwards
— all raise `SourceNotAvailable`. See [docs/usage/source.md](docs/usage/source.md)
and [docs/developing/source-internals.md](docs/developing/source-internals.md), and
`src/nnsight/intervention/source.py`.

### 5.5 Skipping a module

`module.skip(replacement)` tells the interleaver: when the model is about to run
this module, **don't** — use `replacement` as its output instead. The forward body
never runs; `replacement` flows on in its place. It is the tool for ablating a
module, routing around a layer, or splicing in a reconstructed activation (an SAE
output, say):

```python
with model.trace("Hello world"):
    # Pass layer 0's input straight through as its output (a residual bypass).
    model.transformer.h[0].skip(model.transformer.h[0].input)
    output = model.output.save()
```

The skip gate is offered *before* the module's input is read, so reading the
module's own `.input` and handing it back as the replacement — the pass-through
above — works on the very first trace. The replacement must match the shape the
module would normally return (a plain tensor for a GPT-2 block, a tuple for an
attention submodule). Source operations expose the same `.skip(replacement)`.

Skip shares the per-module controller with [source tracing](#54-source-tracing), and
the two compose on one wrapper. For the full mechanics — how skip and early
stopping relate, batched skips that must cover every row, and persistent skips via
`edit(inplace=True)` — see [Skipping and early stopping](#65-skipping-and-early-stopping)
and [docs/usage/skip.md](docs/usage/skip.md).

### 5.6 Aliasing modules (rename)

Different architectures name the same role differently — `transformer.h` here,
`model.layers` there, `gpt_neox.layers` elsewhere. The `rename={...}` constructor
argument installs **aliases** so your intervention code reads the same across
families. An alias is an ordinary attribute pointing at the *same* child envoy
object — not a copy — so the original path keeps working, iteration doesn't
double-count, and the alias survives a [dispatch](#57-dispatch-and-update) re-point
with no rebuild.

The key shape decides where an alias binds, and every resolving envoy in the tree
runs the binding:

- A **single-component key** (`"mlp"`) binds wherever it resolves — *every* block
  that has an `mlp` child gets the alias.
- A **dotted key** (`"transformer.h"`) mounts that subtree on the **root** envoy
  under the alias name.
- A **leading-dot key** (`".h"`) binds relative to *each* envoy — the alias lands on
  whichever envoy actually has that child (e.g. `model.transformer.layers`, not
  `model.layers`).

A value may be one alias or a list of them:

```python
model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    rename={
        "transformer.h": "layers",           # subtree mounted on the root
        "mlp": "my_mlp",                      # every block's mlp aliased
        "transformer": ["mdl", "backbone"],  # two aliases for one path
    },
)

with model.trace("Hello world"):
    a = model.layers[0].my_mlp.output.save()     # via aliases
    b = model.transformer.h[0].mlp.output.save() # original still works
    c = model.mdl.h[0].output.save()             # via the first alias
```

Pass `rename=` at construction — aliases bind during `Envoy.__init__`; there is no
post-hoc API. Avoid alias names that collide with existing `Envoy` attributes
(`output`, `input`, `trace`). See
[docs/usage/rename-modules.md](docs/usage/rename-modules.md).

### 5.7 Dispatch and update

Model wrappers build in two phases. When you construct `TransformersModel("...")`
without `dispatch=True`, nnsight builds the model's **structure on the meta device**
— every module is created, shapes and dtypes are known, but no weights are loaded
and nothing sits in memory. The envoy tree is built over this weightless skeleton,
so you can inspect it, print `.source`, and set up `rename`/`envoys` immediately.

The real weights are loaded on demand. `.dispatch()` — triggered automatically on
the first real (non-scan) interleave, or callable directly to load eagerly — loads
the weights via the wrapper's `_load` and hands the new module to `_update`, which
**re-points the existing envoy tree** at it in place. `_update` walks the tree by
name, swapping each envoy's `._module` for the real one and re-instrumenting it (the
new module gets its own controller and hooks; the meta module's don't carry over).
Passing a ready `nn.Module`, or `dispatch=True`, skips the meta phase and loads
eagerly.

Because `_update` re-points the *same* envoy objects rather than rebuilding the
tree, everything that referenced them stays valid: **aliases survive dispatch** (they
point at the same child objects `_update` re-points), and standalone children added
to the tree that aren't part of the loaded module — `TransformersModel`'s
`generator`, for instance — are left in place, keeping their own module and hooks.
This is why `rename=` and `envoys=` are threaded to the envoy but kept out of the
replayed load arguments: the tree carries them across the swap for free. See
`src/nnsight/modeling/mixins/meta.py`.

### 5.8 Ad-hoc module calls

Inside a trace you can *call* any attached module directly, feeding it whatever input
you like, to compute with it out of its place in the forward pass. The canonical use
is the logit lens — running `lm_head` on an intermediate hidden state:

```python
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    logits = model.lm_head(model.transformer.ln_f(hidden))
    tok = logits[0, -1].argmax(-1).save()
# model.tokenizer.decode(tok) -> ' Paris'
```

By default, while interleaving, `envoy(...)` calls `module.forward(...)` **directly**,
skipping PyTorch's hook dispatch. That is deliberate: it keeps the ad-hoc call from
re-firing the interleaver's own hooks (which would try to switch into the very worker
making the call) and leaves the module's real place in the forward pass untouched.
Outside a trace it's just an ordinary module call.

Pass `hook=True` to force the full `module(...)` path so the module's own hooks *do*
fire and its submodules become addressable at `.submodule.output`. Use it for a
module you've **attached** to the tree that isn't part of the real forward — an
adapter, LoRA, or SAE applied in an [edit](#6-features) — so its internals become
observable:

```python
model.transformer.h[0].adapter = MyAdapter()          # attach; auto-wrapped as a child
with model.edit() as (tracer, edited):
    acts = edited.transformer.h[0].output
    edited.transformer.h[0].output[:] = \
        edited.transformer.h[0].adapter(acts, hook=True)

with edited.trace("Hello world"):
    inner = edited.transformer.h[0].adapter.inner.output.save()  # now hookable
```

Applying the attached module inside an `edit` runs it on every trace; to apply it on
*every step* of a generation loop, put the passthrough under the edit tracer's
`iter`. See [Extending nnsight](#10-extending-nnsight).

### 5.9 Navigating the tree

Beyond attribute access, an envoy offers programmatic ways to walk and address the
tree:

- **`.modules(include_fn=None, names=False)`** flattens the whole subtree (children
  first, then self) into a list, optionally filtered by a predicate; with
  `names=True` it yields `(path, envoy)` tuples. `.named_modules()` is the same with
  names on.
- **Iteration** — `for block in model.transformer.h:` yields *direct* children in
  order (not recursive); `model.transformer.h[0]` indexes them, and `len(envoy)`
  gives a `ModuleList`'s length.
- **`.get(path)`** resolves a dotted path from an envoy — `model.get("transformer.h.0.mlp")`
  — which is the programmatic alternative to attribute access when the path is built
  at runtime. Outside a trace it returns the descendant envoy; inside one, a trailing
  `.output`/`.input` resolves through to the live value.
- **`.path`** is the envoy's dotted location (`"model.transformer.h.0.mlp"`), the
  string every interleaver location is derived from.
- **`._module`** is the wrapped `torch.nn.Module` — the escape hatch to real
  parameters, `state_dict`, class, and so on.
- **`.device`** is the device of the module's first parameter (or `None` if it has
  none); `.devices` is the set of devices its parameters live on. `.to()`, `.cpu()`,
  `.cuda()` move the wrapped module in place and return the envoy for chaining.

```python
for path, envoy in model.named_modules():
    if envoy.device is not None:
        print(path, envoy.device)

mlp = model.get("transformer.h.0.mlp")
```

If a submodule's name shadows an `Envoy` attribute (BERT names a submodule
`output`, for instance), the submodule keeps the name and nnsight's attribute moves
to `nns_<name>`, with a warning.

### 5.10 Custom envoy classes (envoys=)

By default every node in the tree — root and all children — is the base `Envoy`.
Sometimes you want a *particular* module to be a richer envoy: an attention module
that exposes a per-head `.heads` view via its own [eproperty](#52-eproperties-how-values-are-hooked),
say. The `envoys=` constructor argument maps a **module type** (matched against the
module's MRO) or a **dotted path suffix** (`"attn"`, `"transformer.h"`) to a custom
`Envoy` subclass. When a child's module or path matches, it is wrapped with that
subclass instead of the base `Envoy`; non-matching modules stay base `Envoy`. The map
is inherited all the way down the tree, so it applies wherever the match occurs.

```python
class AttnEnvoy(Envoy):
    @eproperty(key="output")
    def heads(self, value):                # a per-head view of the attn output
        ...

model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    envoys={"attn": AttnEnvoy},            # every attn module gets the subclass
)
```

This is the wiring that lets a custom eproperty live on the *right* module rather
than on the model class. The full treatment — defining the eproperty, its
`.transform` for a reshaping view, and the alternatives (a subclass of the model,
`tracer.result`, a runtime's `.logits`) — is in
[Extending nnsight](#10-extending-nnsight) and
[docs/developing/extending-envoy.md](docs/developing/extending-envoy.md).

---

## 6. Features

Everything up to here has been about the shape of a trace: you open a `with` block, the model runs, and [interleaving](#4-interleaving) pauses the model wherever your code asks for a value. This section is the working vocabulary that lives inside that block — the verbs and nouns you reach for once the basic idea is in hand. Each subsection stands on its own; read the one that matches your task.

A few of these are so fundamental that the rest assume them. Saving (§6.1) is how any value survives the block at all. Generation (§6.2) and iteration (§6.3) are inseparable — a generation loop is the whole reason a location can be reached more than once, and iteration is how you target which reach. The others — editing, skipping, gradients, barriers, scanning, caching — are each a single idea layered onto the trace you already understand.

### 6.1 Saving values

The trace body does not run in your frame. It is captured, compiled, and executed against a *copy* of your locals (see [Interleaving](#4-interleaving) and `src/nnsight/tracing/tracer.py`), so by default nothing it computes flows back to you. A variable you bind inside the block simply vanishes when the block exits — read it afterward and you get `UnboundLocalError`. **Saving is how you name the exceptions.** `nnsight.save(x)` (or, equivalently, `x.save()`) marks a value to survive past the trace; on exit, `push_result` writes back only the marked values that are also bound to a name.

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output.save()
    logits = nnsight.save(model.output.logits)

print(hidden.shape, logits.shape)   # readable after the block
```

Two properties of `save` explain everything that can go wrong with it. First, **it marks the concrete object by identity and returns it unchanged**, so you save the value you bind — `h = x.save()`, never `(x.save() * 2)`, which marks `x` and returns the (unsaved) product. Write `(x * 2).save()` instead. Second, **a marked value only comes back if it is bound to a name** the body can push back: a bare `model.output.logits.save()` on its own line marks the tensor but leaves no local to return it under, so it silently never appears. This is invisible locally but bites hard on vLLM and [remote execution](#9-remote-execution), where results are read back by name.

**`save` raises outside a trace.** With no trace running there is nowhere to hand the value back *from*, and its mark would be cleared before anything could read it — so calling it with no trace active is a `ValueError`, not a silent no-op:

```python
xs = nnsight.save([])          # ValueError: save() was called outside a trace
with model.trace("Hello"):
    ...
```

This is the single most common structural mistake, because it collides with the idiom for gathering values. **The rule for collections is: save the container, store raw values in it.** Create the list (or dict) *inside* the trace, save the container itself, and append unmarked values:

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    xs = nnsight.save([])                                  # save the container
    for step in tracer.iter[:3]:
        xs.append(model.transformer.h[-1].output[:, -1, :])   # append raw values
    final = tracer.result.save()
# xs holds the 3 collected values; final is the generated ids
```

Two ways this goes wrong, both worth internalizing:

- **Saving the elements** — `xs.append(x.save())`, or a `[b.output.save() for b in ...]` comprehension — marks values with no name to return them under. It happens to work locally, because the appends mutate a list in a frame you still hold, but on a remote trace the appends happen server-side and *nothing comes back*.
- **Leaving the container unsaved** — `xs = []` inside the trace with no `save`, or a comprehension bound to an unsaved name — never pushes the container back, so it is `UnboundLocalError` after the block.

A comprehension follows the same one rule: `hiddens = nnsight.save([b.output for b in model.transformer.h])` — save the whole list, keep the elements raw.

**Two forms, and when to prefer each.** `x.save()` is mounted on *every* Python object by an optional C extension gated by `CONFIG.APP.PYMOUNT` (default on). Tensors read from `.output`/`.input` always carry `.save()`; but for plain Python values (ints, lists, `torch.Size`) the method exists only if the extension built, so `nnsight.save(x)` — a plain function with no mount dependency — is the safe choice there. When in doubt, use `nnsight.save(x)`. (`docs/usage/save.md`, `docs/gotchas/save.md`.)

Saving nests cleanly. Only the *outermost* trace filters to saved values; an inner trace pushes all its locals up to the enclosing block. This is what makes [sessions](#611-sessions) and [gradient blocks](#66-gradients) let values flow without a `save` at every boundary.

### 6.2 Generate vs pipe

A single [trace](#4-interleaving) runs one forward pass. Real language work needs more than one: autoregressive decoding runs the model once per generated token. nnsight gives you two doors into that, and they return fundamentally different things because they run different code.

**`model.generate(input, max_new_tokens=N, ...)` runs the model's own `generate`** and returns the **token ids** on `tracer.result` — a `[batch, seq]` tensor of the prompt plus completion. It uses the checkpoint's own generation settings, so it is **greedy by default** (it does not apply the sampling a task pipeline would); ask for sampling explicitly with `do_sample=True`. Every kwarg (`generation_config=`, `num_return_sequences=`, ...) is forwarded to the model's `generate`.

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.decode(ids[0]))     # ...the city of Paris, and
```

The finished ids also pass through a `Generator` module, so per-step token access is available at `model.generator.streamer.output` — the prompt arrives as one block, then one new token per step. (Reading the *finished* ids at `model.generator.output` still works but is deprecated; use `tracer.result`.)

**`model.pipe(input, ...)` runs the whole `transformers.pipeline`** end to end and returns what it **postprocesses to** — decoded-text records for text-generation (`[{"generated_text": ...}]`), label/score dicts for a classifier, and so on. Because the pipeline applies the checkpoint's `task_specific_params`, pipe output is **sampled by default** for gpt2; pass `do_sample=False` for reproducible output.

```python
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()
print(records[0]["generated_text"])
```

Both are ordinary tracing contexts — your interventions fire on every forward the decode loop makes. The choice is about the return value and the defaults: reach for `generate` when you want token ids and per-step interventions on the model's own settings; reach for `pipe` when you want the pipeline's decoded records and its preprocessing. (`docs/usage/generate.md`, `docs/usage/pipe.md`.)

### 6.3 Iteration

In a single forward pass a module is reached exactly once. In a generation loop it is reached once per decoded step, so `model.transformer.h[0].output` names a *different occurrence* each step. Iteration is how you say which occurrences a stretch of trace body targets.

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:                     # steps 0, 1, 2
        toks = ...  # reads here bind to the current step
```

`tracer.iter` accepts a slice, an int, or a list: `tracer.iter[:3]` is steps 0–2, `tracer.iter[2]` is just step 2, `tracer.iter[[0, 2, 4]]` is those steps only. `tracer.all()` is exactly `tracer.iter[:]`. `step` is the real integer index, so a plain Python `if step == 2:` works inside the loop. Under the hood, looping over `tracer.iter` walks the running mediator's `iteration` pointer across the selected occurrences (`src/nnsight/intervention/iterator.py`), and restores it on exit, so loops nest.

**The gotcha that defines correct usage: an unbounded `iter[:]` (or `all()`) drops every line after the loop.** An open-ended selection keeps handing out step indices until the model stops generating; the final over-run request — for a step the model never runs — is left parked, and the interleaver throws `OutOfOrderError` into that worker, which is caught and *warned*, not raised. But that unwinding tears down the loop **and every statement after it in the same block**. So a `tracer.result.save()` placed after an unbounded loop never runs.

The fix is to use a **bounded** `iter[:N]` matching `max_new_tokens` — then the loop ends normally and trailing code executes:

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    xs = nnsight.save([])
    for step in tracer.iter[:3]:
        xs.append(model.lm_head.output[0, -1].argmax(dim=-1))
    ids = tracer.result.save()                       # runs, because the loop was bounded
```

`max_new_tokens` is a cap, not a guarantee — if the model stops early (EOS), the steps that didn't happen warn but the reached steps' saved values are kept. Negative indices raise `ValueError` (there is no "last step" shorthand), and there is **no `tracer.next()`** — the old manual-stepping API is gone. The `with tracer.iter[...]:` block form still works but is deprecated in favor of the `for` loop. (`docs/usage/iter-all-next.md`.)

### 6.4 Editing a model

An intervention written inside a trace applies once, to that trace. An *edit* makes it permanent: `model.edit(...)` captures the same intervention DSL but, instead of running it against a live forward, **stores it on the envoy** to be replayed on every later trace. This is how you install always-on transforms — zero a head, add a steering vector, swap in an SAE — without rewriting each trace.

```python
with model.edit() as (tracer, edited):
    edited.transformer.h[0].output[0][:] = 0

with edited.trace("Hello world"):
    out = edited.transformer.h[0].output[0].save()   # zeros — edit applied
with model.trace("Hello world"):
    orig = model.output.save()                        # original model, untouched
```

The default `model.edit()` stores the edit on a **shallow copy** of the envoy (its module, interleaver, and children are shared — no weights are duplicated; only the `_edits` list is independent), leaving the original clean, and binds `(tracer, edited)`. `model.edit(inplace=True)` stores it on the envoy itself and binds only `tracer`. Clear stored edits with `model.clear_edits()`.

Stored edits live in `envoy._edits` and run **first** on every later trace — `Envoy.interleave` prepends them to the run's mediators (`src/nnsight/intervention/envoy.py`), so an edit's swap lands before a same-trace intervention reads that location, and their effects are visible to your code. Multiple edits stack in registration order. A plain edit applies at the *first* occurrence of a location; to re-apply it every step of a generation loop, put the passthrough under the edit tracer's `iter` (see [Iteration](#63-iteration)). Because `_edits` serializes by value, edits ride with the model to a [remote server](#9-remote-execution). (`docs/usage/edit.md`.)

### 6.5 Skipping and early stopping

Two ways to *not run* part of the model, at two different scales.

**`module.skip(replacement)` bypasses one module.** When the model is about to run that module, its forward is not executed — `replacement` is used as its output instead. A skip gate is installed on every module up front (via the source/skip controller), so it works even when the replacement is read from the module's own `.input` — turning the module into a pass-through:

```python
with model.trace(x):
    model.transformer.h[1].skip(model.transformer.h[1].input)   # layer 1 passes through
    out = model.output.save()
```

The replacement must match the shape and type the module would normally return — for a GPT-2 block that is a plain tensor, for some attention submodules a tuple. In a batched trace a `.skip()` must cover **every** row: skip the module in every invoke or none, because a shared forward can't run for only the unskipped rows (otherwise `ValueError: A batched .skip() has to cover every row`). A skip is one-shot per module call; across generation steps each step needs its own, via `tracer.iter[...]` or a persistent [edit](#64-editing-a-model).

**`tracer.stop()` aborts the whole run** at the point the worker is parked — everything captured before it is kept, nothing after it in the model runs. Save what you need *before* stopping, because code after `tracer.stop()` in the same block never executes (Python raises `EarlyStopException` at the call, which the interleaver treats as a clean early exit and swallows):

```python
with model.trace("Hello world") as tracer:
    h0 = model.transformer.h[0].output.save()   # save first
    tracer.stop()                                # layers 1..N never run
```

Requesting a location the run never reached — a module after the stop, or the inner submodules of a skipped module — raises `OutOfOrderError`. (`docs/usage/skip.md`, `docs/usage/stop-and-early-exit.md`.)

### 6.6 Gradients

Reading activations is the forward story; `with tensor.backward():` is the backward one. It runs the real backward pass **interleaved** with the body of its block — a nested interleaving session in its own right — so the block can read and replace the `.grad` of any tensor as the gradient reaches it. nnsight patches `torch.Tensor.backward` at import; a bare `tensor.backward()` with no `with` falls through to vanilla PyTorch.

A backward block is almost always nested inside a forward trace, so the tensors whose gradients you want are the real ones the run produced. **Capture those forward tensors before the backward block** — the forward pass is over by the time autograd runs, so `.output`/`.input` are unreachable inside it; only `.grad` is meaningful there.

```python
with model.trace("Hello world"):
    hs   = model.transformer.h[-1].output      # capture the forward tensor first
    loss = model.output.logits.sum()
    with loss.backward():                       # real backward, interleaved
        g = hs.grad.clone().save()              # gradient flowing into hs
        hs.grad = hs.grad * 2                   # ...and replace it downstream
```

Reading `t.grad` parks the block until autograd produces that gradient (a self-removing hook is registered on `t`); assigning `t.grad = v` swaps a replacement into the same channel. Because gradients flow backward, **request `.grad` in reverse-forward order** — later layers first — or hit `OutOfOrderError`. Request it on the tensor you captured directly, not on a slice or index of it (an indexing view is a new tensor whose gradient isn't the one autograd delivers). `retain_graph=True` supports multiple backward passes over one graph. As a nested trace, a backward block pushes its locals up, so a value saved inside reaches you through the outer trace's boundary. (`docs/usage/backward-and-grad.md`.)

### 6.7 Barriers

Inside a single trace with several `with tracer.invoke(x):` blocks (see [Batching](#72-batching)), each block runs as its own worker, and workers resume **in the order the model reaches what each asked for**, not the order they were written. That is exactly what makes them a batch rather than a sequence — but it means a value one invoke reads and another writes has no guaranteed ordering, and neither worker can see the other's progress. A value produced in one invoke is not visible in another until the ordering is pinned.

`tracer.barrier(n)` is that meeting point. Every block that participates calls the returned barrier; each waits, and the last of the `n` to arrive releases them all — so everything written *above* a barrier has happened before anything written *below* one.

```python
with model.pipe(max_new_tokens=3, do_sample=False) as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()                                    # signal: embeddings read
        result = tracer.result.save()
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()                                    # wait for the read
        model.transformer.wte.output = embeddings    # then hand it across
```

Both invokes touch `wte.output`; without the barrier the second worker would try to swap in `embeddings` before the first had bound the name — `NameError`. The barrier is called, not entered (it is not a context manager), works for any `n`, and is reusable (it empties its waiting list on release). Reach for it whenever two or more invokes hand a value across the *same* module; when they touch different modules, shared invoke scope already handles it. If fewer than `n` blocks call it, it never releases and the run ends reporting the unmet count. (`docs/usage/barrier.md`.)

### 6.8 Scanning

`model.scan(input)` runs the forward under PyTorch's `FakeTensorMode`: tensors carry shape, dtype, and device but no real data, no kernels run, and — crucially — **the model is not dispatched**. It is a full tracing context (a `ScanningTracer`, subclass of the interleaving tracer), so every primitive works — `.output`, `.input`, `.save()`, `tracer.invoke`, `tracer.cache` — but nothing computes. Use it to inspect activation shapes or validate shape-dependent code (slicing, reshapes, intervention indexing) without paying to load weights or run the model.

```python
model = TransformersModel("openai-community/gpt2", task="text-generation")
print(model.dispatched)   # False — architecture on meta, no real weights

with model.scan("The Eiffel Tower is in"):
    dim = nnsight.save(model.transformer.h[0].output.shape[-1])   # int
    hs  = model.transformer.h[-1].output.save()                   # a FakeTensor
print(dim, tuple(hs.shape))   # 768 (1, 7, 768)
print(model.dispatched)       # still False
```

Because scan is a tracing context, **`save` is still required** — the same exit filter applies. The values it hands back are fake tensors: read their `.shape`/`.dtype`/`.device` inside the block, but a fake tensor is valid only within the scan (it cannot be used once the fake mode exits), and shapes come back as `torch.Size`/`int`, so prefer `nnsight.save(...)` for those. Shapes seen in a scan match a real forward. Some ops lack a fake/meta kernel and will raise inside scan even when they run for real. (`docs/usage/scan.md`.)

### 6.9 Caching

Reading one location with `.output` is the retail path; `tracer.cache(...)` is the wholesale one. It records the activations of *many* modules at once across the whole run — every selected layer, and in a generation loop every step. Because the interleaver already funnels every module input/output through `handle` (applying interventions first), the cache is just a **post-intervention observer** — it needs no per-module hooks (`src/nnsight/intervention/cache.py`).

```python
with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache()                          # every module's output
cache["model.transformer.h.0"].output               # by path
cache.transformer.h[0].output                       # or by tree navigation
```

The returned `CacheView` fills in as the run proceeds and is already saved, so it survives the trace. Read a module's captured value with `.output`, `.inputs`, or `.input` after selecting it — by absolute path (`cache["model.transformer.h.0"]`) or by navigating the envoy tree (`cache.transformer.h[0]`), which resolves renames and `ModuleList` indices the same way the model does. Select a subset with `modules=[...]` (envoys or path strings), capture inputs with `include_inputs=True`, and control storage with `device` (default CPU), `dtype`, and `detach`.

Two things shape correct use. **Only modules reached *after* the `tracer.cache(...)` call are captured**, so call it early. And **a module visited more than once accumulates one entry per visit**: `cache[path].output` unwraps a single visit to the value directly but returns a *list* for multiple visits (a generation loop), with `len(cache[path])` the visit count. A cache opened inside an invoke records that invoke's rows only. (`docs/usage/cache.md`.)

### 6.10 The trace result

`tracer.result` is the value the traced call returned — the model's output for `trace`, the token ids for `generate`, the pipeline's records for `pipe`. It is an `eproperty` (`src/nnsight/intervention/tracer.py`): reading it *serves* the return value to a worker parked on it, once the model has produced it. Like any traced value it must be saved to survive the block:

```python
with model.generate("Madison Square Garden is in", max_new_tokens=3) as tracer:
    ids = tracer.result.save()
```

The subtlety is [batching](#72-batching): served through the interleaver's `handle`, `tracer.result` read *inside* an invoke is narrowed to that invoke's rows, so each invoke sees its own slice of the combined output. Read at the trace level it is the whole result. This is why in the [barrier](#67-barriers) example each invoke can save its own `tracer.result` and get back only its prompt's continuation.

### 6.11 Sessions

Each `with model.trace(...)` is its own boundary: values don't cross from one trace to the next, and everything you want to keep needs a `save`. A **session** removes that per-trace boundary. `with model.session():` encloses several traces so a value read in one is available in a later one **without** a `save`, because the *session* — not each trace — is the boundary back to your code.

```python
with model.session():
    with model.trace("Madison Square Garden is in the city of"):
        hs = model.transformer.h[5].output[:, -1, :]     # no .save() needed
    with model.trace("_ _ _ _ _ _ _"):
        model.transformer.h[5].output[:, -1, :] = hs     # flows in
        patched = model.output.logits.argmax(dim=-1).save()   # SAVE — leaves the session
print(patched)
```

This is a direct consequence of the save-nesting rule from §6.1: the save filter runs only at the *outermost* boundary, and a session is that boundary — each inner trace pushes all its locals up. So `hs` needs no save (it stays inside the session), but `patched` does (it crosses back to plain Python). The session body is real Python: loops, conditionals, and building lists all run natively around the nested traces, which execute as they are reached.

A session is also how **multiple traces batch into a single [remote](#9-remote-execution) job** — `remote=True` on the *session* (not the inner traces) ships the whole block as one job, and the inner traces run against the server's model when it executes the body. Mechanically there is no separate session state: `model.session()` returns a plain `Tracer` that captures the block, execs it as real Python, and gates saves at its own outermost boundary (`src/nnsight/intervention/envoy.py`). (`docs/usage/session.md`.)

---

## 7. Modeling

Everything so far — the trace, [interleaving](#4-interleaving), [the envoy tree](#5-the-envoy) — works on any `torch.nn.Module`. What the *modeling* layer adds is everything that surrounds a real model: building it from a repo id, deferring its weights until you actually run, tokenizing a prompt, batching several prompts into one forward, and carrying an identity a remote server can reconstruct. None of that is intervention machinery; it is the plumbing that lets you write `TransformersModel("openai-community/gpt2")` instead of hand-assembling a pipeline and feeding it tensors.

The design keeps that plumbing out of the intervention core. `NNsight(module)` is the whole contract the rest of nnsight depends on — a root [Envoy](#5-the-envoy) over a module tree. The model wrappers are Envoy *subclasses* that layer loading, tokenization, and remote identity on top, each concern a separate mixin so a wrapper takes only the ones it needs. Model classes are exposed lazily from the root package (a module-level `__getattr__`), so `import nnsight` never drags in `transformers`, `diffusers`, or `vllm` — an optional dependency errors only when its model is actually used.

Source lives under `src/nnsight/modeling/`. The routing doc is [docs/models/index.md](docs/models/index.md), with a page per class.

### 7.1 The mixin architecture

The base of every model is `NNsight(torch.nn.Module)` — which is to say, `NNsight` *is* an `Envoy` (`modeling/base.py`). Wrap any module and you get the envoy tree plus `.trace()`, `.scan()`, `.edit()`, `.session()`, `.cache()` for free. `NNsight` adds nothing but a conventional name for "wrap a whole model"; it is a thin named `Envoy`. Use it directly for a custom net or any non-HuggingFace module.

On top of that base sits a short chain of mixins, each earning its place by adding one behavior. Reading up from `Envoy`:

- **`Loadable`** (`mixins/loadable.py`) — an Envoy that loads its *own* module. Every construction routes through `_load`, which returns the module to wrap. The base `_load` returns a ready `torch.nn.Module` as-is, so `Loadable(mod)` wraps it directly; anything else is `NotImplementedError` until a subclass overrides it. This is the single decision point where a subclass decides *what a pre-loaded module means* — `TransformersModel` wraps it in a pipeline, `DiffusionModel` treats it as a component. The `rename`/`envoys` arguments are Envoy concerns, so they are threaded to `Envoy.__init__` and kept out of `_load`.

- **`Meta`** (`mixins/meta.py`) — the lazy build. Loading a large model's weights is slow and memory-hungry, but planning a trace only needs the model's *structure* — the module tree and the shapes flowing through it — which is fixed by config, not weights. So `Meta` does a two-phase build: `_load_meta` constructs a weightless skeleton on the *meta* device up front (a `MetaDevice` torch-function mode forces every tensor onto meta however it is created), and `dispatch()` loads real weights via `_load` and swaps them into the existing envoy tree the first time the model actually runs. `scan()` runs the forward under fake tensors and never dispatches; only a real `interleave()` triggers `dispatch()`. Passing a ready module, or `dispatch=True`, skips the meta phase and loads eagerly.

- **`Remotable`** (`mixins/remotable.py`) — remote identity. A remote run does not ship the model; the server already has it loaded. What travels is a **model key** of the form `"import.path.ClassName:model_key"`: the import path names the wrapper class to reconstruct, the suffix names the checkpoint. `Remotable` adds `to_model_key()`/`from_model_key()`, routes `remote=` on `.trace()`/`.session()` through the remote (or local-simulation) backend, and gives subclasses per-request environment hooks (`_remoteable_get_env`/`_remoteable_set_env`) and a persistent-object map so tokenizers and modules resolve to the server's live objects rather than being serialized. See [Remote execution](#9-remote-execution) for the full flow.

From `Remotable` the tree forks. `HuggingFaceModel(Remotable)` (`modeling/huggingface.py`) is the shared HuggingFace base — it builds the architecture on meta from the repo's config (`AutoConfig` + `from_config`) and loads real weights on dispatch (`from_pretrained`), with the auto class configurable through `AUTO_CLASS`. Its `_remoteable_model_key` canonicalizes the repo id via the Hub so different spellings of the same model produce the same key. `TransformersModel` and `DiffusionModel` extend it. `VLLM(Remotable)` branches off directly, since a vLLM engine loads and runs nothing like a HuggingFace module.

**The load flow, end to end.** You construct a wrapper; unless you passed a ready module or `dispatch=True`, `Meta.__init__` opens a `MetaDevice` context and calls `_load_meta` to build the weightless skeleton, then `Envoy.__init__` mirrors it as the envoy tree. You can `scan()` for shapes at this point with no weights in memory. The first real `.trace()`/`.generate()` calls `interleave`, which sees the model is undispatched and calls `dispatch()` → `_load` → `_update` (real weights swapped into the same envoy objects, so any aliases you hold stay valid). From then on the model is loaded.

```python
model = TransformersModel("openai-community/gpt2")  # meta build, no weights
model.scan("Hello")                                 # inspect shapes, still no weights
with model.trace("Hello"):                          # dispatches real weights on first run
    hidden = model.transformer.h[5].output.save()
```

Because the chain has no ABCs, extension points are just underscore-prefixed methods with working defaults (`_load`, `_load_meta`, `_batch_size`, `_batch`). Subclassing is covered in [Extending nnsight](#10-extending-nnsight); the reference override to read is whichever of these methods your model needs to change.

### 7.2 Batching

A single `with model.trace(input):` runs one forward over one input. But several `with tracer.invoke(x):` blocks inside one trace combine into a *single* batched forward, each block's interventions scoped to only its rows of every activation. Batching is what makes multi-prompt comparison a single efficient pass rather than a Python loop. The mechanism lives in `intervention/batching.py`; the user-facing side is [docs/concepts/batching-and-invokers.md](docs/concepts/batching-and-invokers.md), the internals in [docs/developing/batching-internals.md](docs/developing/batching-internals.md).

The split of responsibility is clean. The **model** knows how to turn inputs into a combined forward; the **Batcher** knows the row bookkeeping. Two model methods carry the model's half:

- `_batch_size(*inputs, **kwargs)` — how many batch rows an invoke contributes (`0` means params-only, e.g. an invoke that just sets `max_new_tokens=` and expects the actual data in other invokes). The base default counts any input as one row; batching models report the true row count of a prompt / list / tensor / encoding.
- `_batch(invokes, fn)` — assemble the collected invokes into the combined `(args, kwargs)` the run will use. This is where sequence lengths are equalized (padding), because the Batcher's row math is dim-0 only.

The `Batcher` (one per trace) does the rest. Each `add` records an invoke and assigns it a `batch_group` — a `[start, size]` row range in the combined batch. At run time `narrow` slices a full batched activation down to a block's rows when it reads, and `widen` splices an edit back into the full tensor. A tensor counts as batched only when its leading dim equals the combined `total`, so non-batched tensors pass through untouched. Crucially, narrowing only kicks in with two or more non-empty invokes — a lone invoke *is* the whole batch and sees every row untouched.

A model picks its batcher through the `_batcher_class` class attribute (default `Batcher`). A model whose batch layout is not a plain dim-0 stack overrides `_narrow_tensor`/`_widen_tensor`. `DiffusionModel` does exactly this with `DiffusionBatcher`: a denoiser sees each prompt repeated `num_images_per_prompt` times and, under classifier-free guidance, the whole thing doubled (unconditional half then conditional half), so its batcher maps each invoke's plain `[start, size]` onto that expanded layout — reading and writing exactly the invoke's rows across both halves — by picking the case from the tensor's leading dim at run time.

`TransformersModel` supplies the most substantial batching (see 7.3): it batches text, token ids, and encodings, left-padding causal decoders and correcting `position_ids`, while refusing to batch inputs that cannot be padded together (a raw feature tensor, a multimodal encoding) rather than silently mangling them.

Note that values produced inside one invoke are not visible in another; sharing across invokes needs `tracer.barrier(n)` (a Features concern, not a batching one).

### 7.3 TransformersModel

`TransformersModel` (`modeling/transformers.py`) is **the** primary HuggingFace class — one wrapper for any task, not one class per modality. It is backed by a `transformers.pipeline` chosen by `task=` (inferred from the checkpoint when unset). The reason to lean on a pipeline rather than re-derive preprocessing: a `transformers.pipeline` already knows which preprocessors a task loads, how to turn its inputs into model inputs, and how to collate them — and all of that varies per task, per checkpoint, and per release of `transformers`. Reusing it is the "use the upstream primitive" principle in practice.

**Three ways to run it, and the difference matters:**

- `trace` runs **one forward**. Its input is assembled here, so it accepts anything the model accepts — text, token ids, a tensor, or a pre-tokenized encoding.
- `generate` generates **through the model** and returns token ids on `tracer.result`. It takes the same inputs a forward does and generates with the checkpoint's own settings (not the `task_specific_params` a pipeline would fold in).
- `pipe` runs **the whole pipeline** — it tokenizes and collates its own input and returns what the pipeline postprocesses to (decoded text, labels, scores).

`generate` versus `pipe` is the distinction covered in [Generate vs pipe](#62-generate-vs-pipe): reach for `generate` when you want the ids (and per-step access), `pipe` when you want the pipeline's finished records.

**Attributes.** The pipeline and its preprocessors are exposed directly: `pipeline`, `tokenizer`, `processor`, `image_processor`, `feature_extractor`. Which of them a task loads varies — a text task has a `tokenizer` and no `image_processor`, a multimodal one has a `processor` — so any of them may be `None`, and the one that will actually be used is `model.tokenizer`. Passing one in at construction adopts it instead of loading it. There is also `generator`, a standalone passthrough module that generation output flows through: reading the finished ids at `model.generator.output` is deprecated in favor of `tracer.result`, but `model.generator.streamer.output` gives per-step token access that `tracer.result` has no equivalent for. (This standalone child survives dispatch and PEFT rebinds, which only rebuild the tree from the HF module's own children.)

It works across tasks — text-generation, fill-mask, text-classification, image-classification, image-text-to-text, feature-extraction, and more — and accepts either a repo id **or** a pre-loaded module. From a repo id, the pipeline loads the model and infers every preprocessor. From a pre-loaded module the factory can't infer, so the task is inferred from the architecture (`_infer_task`: a generative model is text-generation, otherwise the class-name suffix decides) or taken from `task=`, and preprocessors are sourced from what you passed or the model's `name_or_path`. Other construction options: `peft=<repo_id>` applies a LoRA adapter at load time (and can be swapped per request server-side via the remote env hooks); `rename=`, `envoys=`, and `dispatch=` are the standard Envoy/Meta arguments.

**Batching specifics.** `_batch_size`/`_num_rows` classify every input format — a string is one row, a list of strings one per prompt, a flat token-id list one row, a 2-D tensor or list-of-sequences one per leading entry, a chat conversation one row (not one per message). `_batch` dispatches on which run mode is active: `pipe` hands prompts to the pipeline with `batch_size`; `trace`/`generate` assemble model inputs here — each invoke's text goes through the task's own `preprocess`, and the per-invoke encodings are padded together by the pipeline's `pad_collate_fn`, while pre-tokenized ids and raw feature tensors bypass preprocessing. Padding side is the model's business, not the task's: causal decoders left-pad (so `output[:, -1]` is every row's real last token) and get mask-derived `position_ids` so an absolute-position model doesn't mispredict a short prompt padded up to a longer one; encoders keep right padding. Inputs that can't be padded into an `input_ids` batch — a raw feature tensor, a multimodal encoding — are carried straight to the model as a lone invoke, and asking to batch several of them raises rather than silently mangling them.

Full page: [docs/models/transformers-model.md](docs/models/transformers-model.md).

### 7.4 DiffusionModel

`DiffusionModel` (`modeling/diffusion.py`) wraps a `diffusers.DiffusionPipeline`. A diffusion pipeline is not itself a module — it orchestrates several (unet/transformer, vae, text_encoder, scheduler, ...) around a denoising loop — so nnsight wraps it in a `_PipelineModule` that registers each *module* component as a child and forwards a call to the pipeline's denoising loop. The result: each module component is an envoy (`model.unet` or `model.transformer`, `model.vae`, `model.text_encoder`, ...), and `model.output` / `tracer.result` is the pipeline's own image output object (read the images off its `.images`).

Both `trace` and `generate` run the *whole* pipeline, interventions firing on every component the denoising loop invokes; they differ only in the default step count. `trace` defaults to `num_inference_steps=1` — a fast one-step pass for inspecting or editing activations — while `generate` uses the pipeline's own default. Use `tracer.iter` to target a particular inference step.

```python
model = DiffusionModel("stabilityai/sdxl-turbo")
with model.generate("a photo of a cat", num_inference_steps=20):
    latents = model.unet.output[0].save()   # per denoising step under tracer.iter
    images = model.output.save()
images.images[0]  # a PIL image
```

Reproducibility goes through `seed=`: passed to `generate`, an int seed becomes a reproducible `torch.Generator` — a per-image list for a batch, so each image is independently reproducible — while passing `generator=` directly overrides it. Multi-prompt invokes batch via the `DiffusionBatcher` described in 7.2, which handles the classifier-free-guidance doubling and `num_images_per_prompt` expansion. To run one component's forward on its own rather than the whole pipeline, trace that envoy directly: `with model.unet.trace(sample, timestep, encoder_hidden_states=...):`.

Loading follows the same lazy pattern as the HuggingFace base, but the meta build is assembled component-by-component: a pipeline can't be loaded weightless, so each module component is built from its config on meta while the light components (schedulers, tokenizers, processors) load normally on a real device, and a meta pipeline of the same shape is assembled from them. Resolving each component's class handles the `[library, class_name]` spec in `model_index.json`, including Flax/TF class names (mapped to their PyTorch equivalents) and diffusers pipeline-subpackage components (e.g. a safety checker). Requires the optional `diffusers` package. Full page: [docs/models/diffusion-model.md](docs/models/diffusion-model.md).

### 7.5 VLLM

`VLLM` (`modeling/vllm/vllm.py`) is the high-throughput runtime — PagedAttention, continuous batching, tensor parallelism, and optional async streaming, with arbitrary Python interventions written exactly as for any other model. It is a large subsystem; this is the orientation, and [docs/models/vllm.md](docs/models/vllm.md) plus [docs/developing/vllm-integration.md](docs/developing/vllm-integration.md) are the reference.

The defining constraint is that vLLM runs the model in its own worker process. This client process holds only a meta-device copy of the module tree, with no weights to hook, so a trace cannot simply run alongside the forward the way it does locally. Instead the intervention travels *to* the model: each invoke's worker is serialized into its request's `SamplingParams.extra_args`, rides vLLM's own request pipeline into the worker, is deserialized there, run against the real module, and its saved values shipped back. Two things follow. Interventions are scoped to a *request*, so each `tracer.invoke(...)` carries exactly one prompt — several prompts means several invoke blocks, not a list. And sampling settings (`temperature`, `max_tokens`, `top_p`, ...) are passed to `trace`/`invoke` rather than configured on the model, since each invoke is its own vLLM request.

You read generated tokens through `model.logits` and `model.samples` — `eproperty` hookable values (the same descriptor behind `.output`/`.input`; see [Extending nnsight](#10-extending-nnsight)) exposing this step's pre-sampling logits and the token ids drawn from them — not `tracer.result`, which is not served here. `mode="sync"` (default) builds a `vllm.LLM`; `mode="async"` builds vLLM's streaming `AsyncLLM`, and a trace then yields outputs as they generate via `async for output in tracer.backend`, with saves arriving on the finished output's `.saves[...]`. Tensor parallelism is handled by a VLLM-specific batcher that maps each worker onto its own tokens within the flat `[total_tokens, hidden]` slab the scheduler packs. Needs the optional `vllm` extra.

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", dispatch=True)
with model.trace("The Eiffel Tower is in", temperature=0.0):
    model.transformer.h[8].output[:] = 0
    logits = model.logits.save()
```

### 7.6 Deprecated aliases

Two names remain for backwards compatibility and warn (`DeprecationWarning`) on construction:

- **`LanguageModel`** (`modeling/language.py`) — a `TransformersModel` pinned to `task="text-generation"`. It adds nothing of its own beyond accepting `tokenizer_kwargs` at load (apply the same settings to `model.tokenizer` yourself — `padding_side` is the usual one). Use `TransformersModel(repo_id, task="text-generation")`.
- **`VisionLanguageModel`** (`modeling/vlm.py`) — a `TransformersModel` pinned to `task="image-text-to-text"`, taking a prompt and images by keyword (`text=`, `images=`) and running the processor over them before the model's own `generate`. Use `TransformersModel(repo_id, task="image-text-to-text")`.

Both share `TransformersModel`'s remote key (via `_remoteable_class`), so a model deployed as a `TransformersModel` is reachable whether a client wraps it as the base class or either alias. The diffusion class is `DiffusionModel` — there is no separate `DiffusersModel` to migrate from. Pages: [docs/models/language-model.md](docs/models/language-model.md), [docs/models/vision-language-model.md](docs/models/vision-language-model.md).

---

## 8. Debugging

Intervention code is deferred and interleaved: nnsight captures the body of your
`with model.trace(...):` block, compiles it, and runs it in a greenlet worker
that trades control back and forth with the model's forward pass (see
[Interleaving](#4-interleaving)). That indirection is what makes debugging feel
different from ordinary Python — the line that raises did not run where you wrote
it, and by the time an exception surfaces it has passed through nnsight's hooks
and the model's own frames. The three subsections below cover how nnsight keeps
that machinery out of your way when something goes wrong: tracebacks that point
at your code, a single switch that puts the plumbing back when you need to see
it, and the handful of errors you will actually hit.

### 8.1 Clean tracebacks

The most important thing to know is that **an exception raised inside a trace
body is the real exception**. There is no wrapper class, no `.original`
attribute, no dynamically synthesized `NNsightException`. If layer indexing
raises `IndexError`, you catch `IndexError`; if your arithmetic raises
`ValueError`, that is what propagates. The type is preserved end to end:

```python
try:
    with model.trace("Hello"):
        h = model.transformer.h[100].output.save()   # only 12 layers -> IndexError
except IndexError as error:
    print(type(error).__name__)   # IndexError
```

What nnsight does adjust is the *traceback*, not the exception. A raw traceback
from a worker is buried under nnsight's own frames — the interleaver, the
mediator, the module controllers — plus the model's forward stack. So when a trace
body raises, `clean_traceback` (`src/nnsight/tracing/util.py`) rebuilds the
traceback keeping only the frames whose source file lives *outside* the nnsight
package, leaving your own frames across whatever files your intervention code
spans. The plumbing is stripped by default; the error points at the line you
wrote.

nnsight also works to point at the *right* line. When a worker raises, the
mediator stashes an intervention-only traceback on the exception (as
`__intervention_tb__`) before the model and hook frames pile on during
unwinding, so the surfaced trace can name the exact intervention line rather than
the deepest model frame. This matters most for the deferred-worker path (remote
and vLLM), where the error is reduced to a wire-safe dict in one process and
re-raised in another (see `src/nnsight/intervention/errors.py`): a deferred
worker error comes back as a `RuntimeError` carrying the original type name,
message, and that intervention traceback, because reconstructing the original
exception class across a process boundary is brittle.

### 8.2 DEBUG mode and `-v`

Sometimes the bug *is* in the plumbing — or you want to see every frame anyway.
`CONFIG.APP.DEBUG` (in `src/nnsight/schema/config.py`) is the single switch that
turns clean tracebacks off, and it does two things:

1. **Full tracebacks.** With `DEBUG` on, `clean_traceback` strips nothing: the
   whole stack, nnsight internals included, is shown. Turn it on when you suspect
   the fault is in nnsight rather than your intervention code.
2. **Verbose remote logging.** `RemoteBackend` sets `self.verbose = verbose or
   CONFIG.APP.DEBUG`, so remote runs log payload and result byte sizes and print
   each status update on its own line instead of collapsing into one in-place
   spinner (see [Remote execution](#9-remote-execution)).

There are four ways to enable it, matching how you tend to run code:

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True          # this process
```

```bash
NNSIGHT_DEBUG=1 python your_script.py    # environment variable
python your_script.py -v                 # or --verbose
```

The command-line form is a plain `sys.argv` scan performed once at import
(`Config._from_cli`), checking for `-v` or `--verbose` anywhere in the launching
command. It is deliberately dumb: any launcher that happens to pass `-v` turns
debug mode on too, so a run under `pytest -v` executes with full tracebacks and
verbose remote logging.

To make it stick, persist it to the user config file:

```python
nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()                    # writes ~/.config/nnsight/config.yaml
```

Debug output is noisy — payload sizes and a per-status timeline on every remote
run — so turn it back off for clean output once you are done. A related switch,
`CONFIG.APP.REMOTE_LOGGING` (default `True`), controls the status display
independently of `DEBUG`; see [Configuration](#92-configuration). Full settings
reference: `docs/reference/config.md`; the traceback details:
`docs/errors/debug-mode.md`.

### 8.3 Common errors

These are the errors you will actually meet, each with its cause and its fix. The
full map lives in `docs/errors/index.md`; the exception type is always preserved,
so you can `except` on the real class.

**`OutOfOrderError` — "`'<location>'` was requested but the model already ran
past it."** Each block of intervention code runs in its own worker that is served
locations *in the order the model reaches them*, holding one pending request at a
time. Ask for layer 1's output after layer 5's and layer 1 has already fired and
gone; the worker is left parked, and at the end of the run
`Interleaver.check_dangling_mediators` throws `OutOfOrderError` into it so the
traceback lands on the exact waiting line. Two flavors share this one class:

- *Wrong order.* Reading modules out of forward-pass order within one block. Fix:
  lay your reads out top-to-bottom in the order modules run (the order in
  `print(model)`), or split the out-of-order reads across separate invokes —
  each invoke is its own worker with an independent access order.
- *Never reached.* The model finished with a worker still waiting for a location
  that never fired — a module skipped under `model.eval()`, a branch not taken, a
  submodule of a `.skip()`-ped module, an `iter` loop that outran the model.
  Confirm a module actually fires with `model.scan(...)` before reading it.

The `.i<n>` suffix on the location is the occurrence tag — which visit of that
location the request targets; `.i0` outside iteration, counting up per step
inside a generation loop. Import it from `nnsight.intervention.interleaver` to
catch it. (`docs/errors/out-of-order-error.md`,
`docs/errors/value-was-not-provided.md`.)

**`Cannot access '<location>' outside of interleaving`** (a `ValueError`).
Reading or writing an Envoy value — `.output`, `.input`, `.inputs`, `.source` —
when no trace is running. Envoy properties resolve through the worker driving the
current intervention, and intervention code only runs *while interleaving*, so no
worker means there is nothing to park on and nothing to answer with. Assigning to
one gives the same message, since a swap goes through the same check. You hit this
by reading a value after the block exited without saving it, or from a closure
that captures an Envoy and runs later. It also fires inside the body of an
invoke-mode trace, which runs inline only to collect its invokes — the reads
belong inside a `tracer.invoke(...)` block. Fix: read inside the trace and
`.save()` what you need afterward (see [Saving values](#61-saving-values)).
(`docs/errors/cannot-access-outside-interleaving.md`.)

**`trace() needs an input, or at least one 'with tracer.invoke(...)' block`** (a
`ValueError`). A `with model.trace() as tracer:` with no direct input *and* no
`tracer.invoke(...)` block has no batch to run on. Give `trace()` an input
directly, or add at least one invoke.
(`docs/errors/cannot-access-outside-interleaving.md`.)

**`save() was called outside a trace`** (a `ValueError`). `.save()` and
`nnsight.save(x)` mark a value to be returned when the outermost trace exits,
which only means something inside a trace. Calling it before the block —
`acts = nnsight.save([])` on the line *above* `with model.trace(...)` — marks
into a saved set that is cleared before anything reads it, so it is an explicit
error rather than a silent no-op. Move the save inside the block, and build any
accumulator there:

```python
with model.trace("Hello"):
    acts = nnsight.save([])                          # accumulator, saved inside
    acts.append(model.transformer.h[0].output)
```

(`docs/errors/save-outside-trace.md`; the internal `mark()` is the same mechanism
without the guard, for backends recording a finished request's values.)

**The unbounded `iter[:]` drops-trailing-code warning.** An open-ended
`for step in tracer.iter[:]:` (or `tracer.all()`) that outruns the model leaves
the final over-run request dangling; because the worker is inside an iteration
loop, nnsight *warns* rather than raising, unwinding the loop to run its
`finally` blocks and keeping values from the steps that were reached. The catch
is that unwinding drops the loop *and every line after it* — so a
`tracer.result.save()` placed after an open-ended loop never runs. Prefer a
bounded `iter[:N]` when you need code to execute after the loop; full treatment
in [Iteration](#63-iteration).

## 9. Remote execution

Everything you have written so far runs the same trace whether the weights sit on
your GPU or on someone else's. That is the whole idea behind remote execution:
you build the model locally on the meta device — its architecture constructed so
`model.transformer.h[0].output` is a real Envoy path, but no weights allocated —
write ordinary intervention code against it, and ship the *serialized trace* to a
server that holds the real weights. The model never travels; your code does. This
section covers where it runs (NDIF), how to configure it, and the four ways to
wait for a job, ending with the in-process dry run that proves your trace will
ship.

### 9.1 NDIF

NDIF — the National Deep Inference Fabric — is a hosted service that runs nnsight
intervention code on shared GPU pods. It exists so you can trace models that will
never fit on your hardware: Llama-3.1-70B and 405B, DeepSeek, and the like. You
instantiate the wrapper locally (on meta, no GPU, no download), and NDIF
deserializes your trace on a server that holds the weights, runs the forward pass
with your interventions spliced in, and streams the results back.

What crosses the wire is the trace, serialized *source and all*: the captured
block, every function and class it references, and any registered local modules
are reduced to source plus their referenced globals and locals and pickled
(zstd-compressed when `CONFIG.API.COMPRESS` is on). The model itself is never
serialized — it is named by a `model_key` and must already be deployed on NDIF.
Because the whole thing ships as source, anything your block touches must be
importable on the server or shipped by value; local-only modules are registered
automatically (see [remote="local"](#96-remotelocal) and
`docs/remote/register-local-modules.md`).

A submitted job moves through a lifecycle you watch as status updates:
`RECEIVED` (validated and accepted) → `QUEUED` (waiting in the model's queue) →
`PROVISIONING`/`DEPLOYING` (capacity coming up) → `DISPATCHED` (handed to a
deployment) → `RUNNING` (forward pass on the GPU) → `COMPLETED` (saves ready to
download) or `ERROR` (a server-side exception, surfaced locally as `RemoteError`
with the remote traceback). A `LOG` update carries a `print(...)` from inside
your block — a transient message, not a lifecycle stage. Only values you
`.save()` come back; everything else is local to the server run and discarded.
(`docs/remote/ndif-overview.md`.)

### 9.2 Configuration

Every remote request is keyed against an NDIF API key, and both the key and the
host live on the `CONFIG` singleton (`src/nnsight/schema/config.py`). The key
sits at `CONFIG.API.APIKEY`, the base URL at `CONFIG.API.HOST` (default
`https://api.ndif.us`; the websocket URL is derived, `https://` → `wss://`).

The canonical way to set the key is once per machine:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("YOUR_KEY")   # sets APIKEY and persists it
```

`set_default_api_key` assigns the key and calls `CONFIG.save()`. There are two
other paths: the `NDIF_API_KEY` environment variable (read at import, overriding
the on-disk value), and — in Colab — a Userdata secret named `NDIF_API_KEY`,
read as a fallback when neither the env var nor a file key is set.

Config is layered, later winning: **shipped defaults < user config file <
environment**. The user file lives at `$XDG_CONFIG_HOME/nnsight/config.yaml`
(default `~/.config/nnsight/config.yaml`), or wherever `$NNSIGHT_CONFIG` points.
Keeping it under `~/.config`, separate from the package's shipped
`config.yaml`, is deliberate: upgrading nnsight cannot clobber your saved key.
`CONFIG.save()` writes the current values back to the *user* file, never the
shipped one.

Two `CONFIG.APP` switches shape remote runs beyond `DEBUG` (see
[DEBUG mode and `-v`](#82-debug-mode-and--v)):

- `CONFIG.APP.PYMOUNT` (default `True`) mounts `.save()` onto every object so
  `value.save()` works in a trace. When it is `False` — or the optional C
  extension that mounts it did not build — use `nnsight.save(value)` instead.
  Mounting adds `.save` to all objects process-wide, so anything checking
  `hasattr(x, "save")` will see it.
- `CONFIG.APP.REMOTE_LOGGING` (default `True`) shows the live status display and
  the download progress bar. Set it `False` for silent runs.

To target a different deployment, set `CONFIG.API.HOST` (persistently, or via the
`NDIF_HOST` env var), or override per call by passing the host URL as `remote=`;
the URL must start with `http://` or `https://`. (`docs/remote/api-key-and-config.md`,
`docs/reference/config.md`.)

### 9.3 Blocking and non-blocking

`model.trace(input, remote=True)` is the simplest remote run. It is the same
`trace` you call locally — the block is captured on `__exit__`, serialized, and
handed to a `RemoteBackend`. In the default **blocking** mode the client holds one
`/subscribe` websocket open: it takes a session id, POSTs the payload to
`/request`, reads status updates off the socket until `COMPLETED`, downloads the
result from a presigned URL, and pushes the saved values back into your frame so
your `h = ...save()` variables populate — exactly as a local trace would.

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")
model = TransformersModel("meta-llama/Llama-3.1-70B")   # meta device

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

For a job you do not want to block on, pass `blocking=False`. This swaps to
fire-and-poll: the trace's `__exit__` submits the request over plain HTTP (no
websocket — the server records each status to its object store) and returns
immediately, leaving the backend holding a `job_id`. Each later call to the
backend polls `GET /response/{job_id}`, returning `None` while the job is still
running and the saves dict on `COMPLETED`:

```python
import time

with model.trace("Hello", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()

backend = tracer.backend        # the RemoteBackend, now holding the job id
while True:
    result = backend()          # None until COMPLETED, then the saves dict
    if result is not None:
        break
    time.sleep(1)

print(result["output"].shape)
```

The result dict is keyed by the **saved variable's name** in your trace — in the
non-blocking and async paths the trace has long since exited, so nothing is
pushed into a frame; you read `result["output"]`. Because there is no background
polling thread, each `backend()` call fetches only whatever status the server
last recorded — poll once after a long wait and you may jump straight from
`RECEIVED` to the result, observing no intermediate states. If you stored a
`job_id`, you can construct a poll-only backend later
(`RemoteBackend(model.to_model_key(), blocking=False, job_id="...")`) and fetch
the result without resubmitting. (`docs/remote/remote-trace.md`,
`docs/remote/non-blocking-jobs.md`.)

### 9.4 Async

`AsyncRemoteBackend` waits for a job on an asyncio event loop rather than blocking
a thread or polling by hand. Submission is still synchronous — the backend
subscribes, takes the session id, and POSTs the payload inside the trace's
`__exit__` — but only the *waiting* is async, so the loop stays free while the job
runs. Construct one and pass it as the trace's `backend`, then `await` it for the
saves dict:

```python
import asyncio
from nnsight import TransformersModel
from nnsight.intervention.backends.remote import AsyncRemoteBackend

model = TransformersModel("meta-llama/Llama-3.1-70B")

async def main():
    backend = AsyncRemoteBackend(model.to_model_key())
    with model.trace("The Eiffel Tower is in the city of", backend=backend):
        logit = model.lm_head.output[0][-1].argmax(dim=-1).save()
    result = await backend                       # wait for COMPLETED, get the saves
    print(model.tokenizer.decode(result["logit"]))

asyncio.run(main())
```

`await backend` renders the status display and raises `RemoteError` on a server
error, just like the blocking parent; the result is a dict keyed by your saved
variable names. Because the websocket `recv` is blocking, it runs through
`asyncio.to_thread`, so several backends awaited together with `asyncio.gather`
all make progress at once.

The other form, `async for update in backend`, hands you each raw `ResponseModel`
status update as it lands and then the **saves dict as the final item**. This
form does *not* touch the display and does *not* raise on `ERROR` — an `ERROR`
simply ends the stream, and you inspect it and raise yourself if you want. Tell
the two apart by type: every status update is a `ResponseModel`, the single final
item is a plain `dict`:

```python
async for update in backend:
    if isinstance(update, dict):
        result = update                          # the saves dict, yielded last
    else:
        print(update.status, update.description) # raw status update
```

The connection closes automatically when the await resolves or the iterator
finishes. Construct the backend and enter the trace on the same thread that later
awaits it. (`docs/remote/remote-async.md`.)

### 9.5 Sessions

A session bundles several traces into a **single** NDIF job. The whole session
block serializes as one request, queues once, executes contiguously on the
server, and returns its saved values together — one queue wait instead of three
for a three-step experiment. `remote=True` goes on `model.session(...)`, **not**
on the inner `model.trace(...)` calls: the session already provides the remote
backend, and the inner traces run inside it.

```python
with model.session(remote=True):
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[:, -1, :]          # captured, not saved

    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[:, -1, :] = hs          # reused directly
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()
```

The reason to reach for a session over a run of separate remote traces is that
**values flow across traces without a round trip**. `hs` above is captured in the
first trace and reused in the third; it never leaves the server, so no `.save()`
and no result download. You call `.save()` only on the values you want returned
to your process — cross-trace sharing inside the session is free, cross-process
(server → you) is not. To carry a collection built across traces back, create it
as a saved accumulator at session scope and append to it:

```python
import nnsight

with model.session(remote=True):
    means = nnsight.save([])                      # saved accumulator
    for i in range(12):
        with model.trace("Hello"):
            means.append(model.transformer.h[i].output.mean())

print(len(means))   # 12
```

Sessions cut queue and transport overhead, not GPU time; a five-minute session is
still five minutes of compute. Variables defined outside the session cannot be
referenced inside it — build everything from scratch in the block. And a session
is all-or-nothing: if any inner trace raises, the job aborts and no further
traces run, so structure fault-tolerant pipelines as separate jobs. Sessions
support `blocking=False` too, polled through `session.backend()` exactly like a
non-blocking trace. This all rests on the same `Remotable` mixin that powers
`trace(remote=True)` — see [The mixin architecture](#71-the-mixin-architecture)
and the local [Sessions](#611-sessions). (`docs/remote/remote-session.md`.)

### 9.6 remote="local"

`remote="local"` runs the entire serialize → deserialize → execute round trip
**in-process** — no server, no network, no key. It is a dry run of a real NDIF
request: `LocalSimulationBackend`
(`src/nnsight/intervention/backends/local.py`) serializes the trace exactly as
`RemoteBackend` would, then deserializes it *with your non-installed modules
hidden* — mimicking a server whose environment does not contain your own source
files — and runs the deserialized block against your real, dispatched local
model. Results land back in your frame like an ordinary local trace.

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

The point is the hidden-modules step. If your block references a local function
or class that was not shipped by value, the deserialize raises
`ModuleNotFoundError` — exactly as it would on the server — so a passing
`remote="local"` run is strong evidence a real `remote=True` run will work. It is
the recommended way to validate a remote script offline before spending a queue
slot on it. (`docs/remote/ndif-overview.md`,
`docs/remote/register-local-modules.md`.)

---

## 10. Extending nnsight

Everything nnsight does to a model — mirroring its tree, hooking its values,
batching several prompts into one forward, loading it lazily, shipping it to
NDIF — is assembled from a handful of small, overridable pieces. You extend
nnsight by subclassing one of those pieces and filling in a method or two, not
by reaching into the interleaver. The extension surface is deliberately narrow:
underscore-prefixed methods with working defaults, a descriptor for hookable
values, a class attribute for the batch layout, and a one-method `Backend`.
This section walks each in turn.

The mental model to carry in: `NNsight` *is* an `Envoy` (see [The Envoy](#5-the-envoy)),
the higher-level model classes are envoys with a loading/execution mixin chain
stacked on top (see [The mixin architecture](#71-the-mixin-architecture)), and
every value you can read or write is an `eproperty` over one location string
(see [eproperties: how values are hooked](#52-eproperties-how-values-are-hooked)).
Extending nnsight means adding to one of those three layers. The recipe-style
reference for all of this is `docs/usage/extending.md`; the developer deep-dives
are `docs/developing/extending-envoy.md`, `docs/developing/adding-a-new-runtime.md`,
and `docs/developing/adding-a-new-backend.md`.

### 10.1 Subclassing NNsight and Envoy

The simplest extension adds behavior to a whole model. `NNsight` is a thin,
named `Envoy` (`class NNsight(Envoy)` with an empty body in
`src/nnsight/modeling/base.py`), so a subclass of it is a normal Python class
that also happens to wrap a module tree. Add methods that run inside a trace,
add per-instance configuration in `__init__`, override `_batch_size`/`_batch`
to support batched invokes (see [Batching](#72-batching)) — nothing about the
subclass is special until you override a hook.

```python
from nnsight import NNsight

class MyModel(NNsight):
    def logit_lens(self, hidden):
        return self[1](hidden)      # run a later module ad hoc, inside the trace
```

Calling a child envoy directly (`self[1](hidden)`) runs that module's forward
out of execution order without re-firing the interleaver's hooks — the logit-lens
idiom, and the reason `Envoy.__call__` exists. Because a class-level attribute on
an `Envoy`/`NNsight` subclass is shared across every instance, keep mutable
per-model config (a head count, a device map) in `__init__`, not at class scope.

**When you need loading, lazy build, or remote, subclass the mixin chain instead
of `NNsight`.** `NNsight` wraps an already-instantiated `nn.Module`; it has no
`_load`, no `dispatch`, no `scan`. Those capabilities live in a short chain of
mixins over `Envoy`, and you inherit exactly as many as you need
(`src/nnsight/modeling/mixins/`, and see [The mixin architecture](#71-the-mixin-architecture)):

```
Envoy               the tree; hooks; trace / interleave / __call__
 └─ Loadable        _load(...): construct the module from a spec, not a passed one
     └─ Meta        meta-device build up front; dispatch() swaps in real weights; scan()
         └─ Remotable   remote model key + per-request env; remote & local backends
             └─ HuggingFaceModel   from_pretrained loading by repo id
```

Pick the lowest base that gives you what you need. If you already hold the
`nn.Module`, `NNsight(module)` is enough. If you construct it from a repo id or
a config, start at `Loadable` and override `_load`:

```python
from nnsight.modeling.mixins.loadable import Loadable

class MyLoadable(Loadable):
    def _load(self, repo_id, **kwargs):
        return build_module_from(repo_id, **kwargs)   # your loader; returns an nn.Module

model = MyLoadable("my-org/my-model")   # _load runs, and its result is wrapped
```

`Loadable.__init__` routes every construction through `_load`
(`src/nnsight/modeling/mixins/loadable.py`). The base `_load` returns a ready
`torch.nn.Module` as-is — so `Loadable(mod)` still wraps a live module directly —
and raises `NotImplementedError` for anything else. Your override decides what a
pre-loaded module means for your runtime (`TransformersModel`, for instance,
wraps a passed module in a `transformers.pipeline`). `rename` and `envoys` are
`Envoy` concerns, not load arguments, so the mixin keeps them out of the `_load`
signature.

If you also want a **meta-device tree built up front** — so users can build the
envoy tree, inspect shapes, and call `scan()` without paying for weights — add
the `Meta` mixin and override `_load_meta` alongside `_load`. `Meta.__init__`
runs `_load_meta` inside `with MetaDevice():`, which forces every tensor onto the
meta device; `dispatch()` later calls `_load` and re-points the tree at real
weights, and it fires automatically on the first `interleave` if not already
dispatched. `DiffusionModel` is the worked example
(`src/nnsight/modeling/diffusion.py`): `_load` builds a real pipeline with real
weights, and `_load_meta` assembles a same-shape pipeline component-by-component
from configs on the meta device, loading only the light components (schedulers,
tokenizers) for real. Override `_load`, not `dispatch`, when loading needs
preconditions.

**Attaching a standalone module to the tree.** Submodules of the wrapped module
are mirrored as envoys automatically. To expose a module that is *not* part of
the wrapped module — a streamer, a sampler, a generated-id passthrough — build an
`Envoy` over it with the model's own interleaver and append it to `_children`.
This is exactly how `TransformersModel` exposes `model.generator`
(`src/nnsight/modeling/transformers.py`):

```python
self.generator = Envoy(
    Generator(), path=f"{self.path}.generator", interleaver=self.interleaver
)
self._children.append(self.generator)
```

Two constraints fall out. First, for the standalone module's `.output` to be
*readable*, the module has to actually be called during the run — you pass values
through it in your `trace`/`generate` override with `self.generator(output, hook=True)`,
which fires its hooks so `model.generator.output` receives the value (and can
edit it). Second, standalone children survive a model-environment rebind — lazy
dispatch swapping in real weights, or a PEFT adapter rebind — because they keep
their own module and hooks rather than being rebuilt from the wrapped tree.

### 10.2 Custom hookable values (eproperty)

`.output`, `.input`, `.inputs`, and `tracer.result` are not special-cased in the
interleaver — they are all instances of one descriptor, `eproperty`
(`src/nnsight/intervention/eproperty.py`), and you can define your own. An
`eproperty` turns a plain attribute into a hook into the run: reading it parks
the worker until the model reaches that location and hands back the value there;
writing it swaps a new value in. The location is `"{obj.path}.{key}"` — or just
`key` when the host has no `path`, as for `tracer.result`. Because the descriptor
reads and writes through the `Mediator`, an `eproperty` accessed outside a trace
raises rather than returning stale state (see [eproperties: how values are
hooked](#52-eproperties-how-values-are-hooked) and
[Interleaving](#4-interleaving)).

You define an `eproperty` for a value the model produces *outside* an ordinary
module hook — an engine's logits, a per-head reshaping of an activation, a
telemetry read — that you still want to read and edit like any other location.
The whole mechanism is one primitive on the interleaver:
`handle(location, value)` offers a produced value to every worker parked on that
location and returns whatever they wrote back. An `eproperty` is the reusable
read/write wrapper over one such location, with two ends.

**The read side — the API a user writes.** Decorate a stub with `@eproperty`
(bare) or `@eproperty(key=..., description=...)`. **The decorated stub *is* the
preprocess**: it takes the raw value the interleaver served and returns what the
user reads, so an identity view is just `return value`.

```python
from nnsight.intervention.eproperty import eproperty

class MyModel(NNsight):
    @eproperty                       # key defaults to the stub's name, "telemetry"
    def telemetry(self, value):      # preprocess: served value -> what you read
        return value
```

Three keyword and callback refinements shape the descriptor:

- **`key`** — the location suffix appended to the host's path. It defaults to the
  stub's name. Several eproperties may share a key to give different *views* of the
  same location: `.inputs` uses `@eproperty(key="input")` so it and `.input`
  address the same served value.
- **`description=`** — a short label, used only in the repr. An `eproperty` with a
  description surfaces in the Envoy repr tree as `(name): description`; the plain
  built-in views (`.output`/`.input`) carry none and stay hidden. Give a runtime
  value like `.logits` a description so it shows up.
- **`.postprocess(self, value)`** — runs on a *written* value before it is swapped
  in. `Envoy.input` uses it to repack a lone first argument back into the full
  `(args, kwargs)` the model expects.
- **`.transform(self, value)`** — the write-back half of a *reshaping* preprocess.
  When the preprocess hands back a reshaped or sliced view, the user's in-place
  edits to that view are invisible to the model, which still holds the original.
  A transform maps the edited view back to the model's layout; it fires once,
  after the block is done with the read, and its result is spliced in as if
  swapped. Note the asymmetry: an *aliasing* view (a `.view()` / `.transpose()`
  that shares storage) propagates edits without a transform, so you only need one
  when the preprocess produced a copy or the write path is a reshape.

**The produce side — where the value exists.** `eproperty.provide(obj, value)`
calls `obj.interleaver.handle(location, value)`, serving the value to a parked
worker and returning it — edited if the worker wrote back. Call it from your
runtime wherever the value is computed, inside an open interleaver context. This
is how vLLM feeds `.logits` and `.samples` from its model runner:
`type(model).logits.provide(model, original)` serves the value at
`"model.logits"`, the exact location a user's `logits = model.logits.save()` is
parked on. There is no registration table — a descriptor and a `provide`, and the
two sides cannot drift out of sync because they name the same location.

The canonical reshaping example is a per-head view of an attention (or MLP)
output. The preprocess reshapes `[B, S, H]` into `[B, n_heads, S, head_dim]`;
the transform writes an edited view back into the module's real layout (verified
in `tests/test_language.py`, class `Heads`):

```python
from nnsight import Envoy
from nnsight.intervention.eproperty import eproperty

class Heads(Envoy):
    n_heads = 12

    @eproperty(key="output")
    def heads(self, value):                       # preprocess: [B,S,H] -> per-head
        b, s, h = value.shape
        return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

    @heads.transform
    def heads(self, value):                       # write the edited heads back
        b, nh, s, hd = value.shape
        return value.transpose(1, 2).reshape(b, s, nh * hd)

with model.trace(prompt):
    model.attn.heads[:, 5] = 0                    # zero head 5; transform reshapes it back
```

Here `key="output"` deliberately shares the module's `.output` location, so
`.heads` is a second, reshaped view of the very value `.output` serves. Because
the preprocess returns a reshaped view and the write is a reshape, the transform
is required for the edit to reach the model. The full recipe — including the
non-module `tracer.result` case and the vLLM runtime values — is in
`docs/developing/extending-envoy.md`.

### 10.3 Custom envoy classes (envoys=)

A custom `eproperty` only takes effect on a class that is actually used as an
envoy. Child modules default to the base `Envoy`, so a `.heads` accessor defined
on an `Envoy` subclass reaches a specific submodule only if that submodule is
wrapped with the subclass. The `envoys=` argument is how you make that happen.

`envoys=` is a map, threaded down the whole tree, from a module *type* or a
dotted *path suffix* to a custom `Envoy` subclass. When each child is wrapped,
`_resolve_envoy_class` (`src/nnsight/intervention/envoy.py`) consults the map:
a `torch.nn.Module` subclass key is matched against the module's MRO (tried
first, so a base class matches every subclass), and a string key matches a
dotted path suffix (`"mlp"`, `"transformer.h"`). Anything that matches nothing
stays the base `Envoy`. The map is inherited by children, so a single spec at the
root applies at every depth it resolves.

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={GPT2MLP: Heads}, dispatch=True,        # or {"mlp": Heads} by path suffix
)

model.transformer.h[0].mlp        # a Heads envoy, with a .heads eproperty
model.transformer.h[0].attn       # untouched: still the base Envoy
```

The `Heads` class above reshapes a bare `[B, S, H]` tensor, so it fits a module
whose `.output` *is* that tensor (a GPT-2 MLP). A module whose `.output` is a
tuple — a GPT-2 attention block — needs a preprocess that indexes `value[0]`
first; `docs/patterns/per-head-attention.md` carries that tuple-output variant.
The point of `envoys=` is precisely this targeting: you attach the reshaping
view to exactly the modules whose output layout it understands, and leave every
other module as the plain `Envoy`. When you have no specific module to target —
a run-level value like a runtime's logits — put the `eproperty` on the
model/runtime subclass (or the tracer) directly instead, and skip `envoys=`
(see [Custom hookable values (eproperty)](#102-custom-hookable-values-eproperty)).

### 10.4 Custom batching

The batcher (see [Batching](#72-batching)) assumes every batched activation is a
plain dim-0 stack: the combined forward's leading dimension is the concatenation
of each invoke's rows, so narrowing an activation to a block means slicing
`[start, start + size)` on dim 0, and widening an edit back means splicing those
rows in with a `cat`. That assumption is wrong for some runtimes. A diffusion
denoiser repeats each prompt `num_images_per_prompt` times and, under
classifier-free guidance, doubles the whole batch into an unconditional half
followed by a conditional half; vLLM flattens tokens onto a single axis. When the
batch axis isn't a plain dim-0 stack, you subclass `Batcher`.

The base `Batcher` (`src/nnsight/intervention/batching.py`) does the container
walk for you — recursing through tuples, lists, dicts, and HF `ModelOutput`s to
find the tensors — and delegates the per-tensor row math to two overridable
methods:

- **`_narrow_tensor(tensor, group) -> tensor`** — slice one batched tensor down
  to a group's rows.
- **`_widen_tensor(full, group, edited) -> tensor`** — write an edited block's
  rows back into the full tensor.

Override those two (and, if your layout needs bookkeeping beyond the plain
`[start, size]` group, `add`) and you get correct narrow/widen for any layout,
because the parallel container walk and the `SkipParts` machinery are unchanged.
Point your model at the subclass with the `_batcher_class` class attribute; the
standard tracer instantiates `self.envoy._batcher_class(self.envoy, self.kwargs)`
and hands it to `interleave(batcher=...)`, so setting the attribute is all the
wiring required.

`DiffusionBatcher` (`src/nnsight/modeling/diffusion.py`) is the worked example.
It reads `num_images_per_prompt` off the trace's forward kwargs, and in `add` it
records, alongside each invoke's plain group, an *image-expanded* group in the
repeated batch. Its `_narrow_tensor` then picks the case by the tensor's leading
dim at run time — an un-expanded activation (`rows == total`) narrows on the
plain group; an image-repeated one (`rows == image_total`) narrows on the image
group; a guidance-doubled one (`rows == image_total * 2`) narrows *both* halves
and concatenates them, so an intervention on `model.unet` reads exactly its
invoke's rows across the unconditional and conditional passes alike.
`_widen_tensor` inverts each case, chunking a doubled edit back into its two
halves. `DiffusionModel` then wires it in with a single line, `_batcher_class = DiffusionBatcher`.
`docs/developing/batching-internals.md` covers the contract in full.

### 10.5 New runtimes and backends

The two words name two different extension points, and knowing which you want
saves a lot of work.

A **backend** decides *what is done with the captured block* — run it here, ship
it to a server, serialize and replay it, log it. A **runtime** is a model type
with its own loading, batching, or execution model — a new inference engine. If
you want the same trace to run somewhere else, you want a backend; if you want a
different kind of model, you want a runtime.

**Backends.** A `Backend` is a one-method callable
(`src/nnsight/tracing/backend.py`):

```python
class Backend:
    def __call__(self, tracer):
        tracer.execute(tracer.info.code)
```

By the time your `__call__` runs, the block is already captured and compiled:
`tracer.info.code` is the compiled block body, `tracer.info.frame` is the
caller's live frame. You decide whether to run it (`tracer.execute(code)` — for
an `InterleavingTracer`, that stands up the interleaver and runs the model),
transform it, ship it, or store it. The base runs it in place; `RemoteBackend`
skips local execution and serializes the tracer to NDIF; `LocalSimulationBackend`
round-trips it through serialization to validate. You wire a backend in with
`model.trace(..., backend=MyBackend())`, or, for a remote-style backend keyed off
`remote=`, by adding a branch in your model's `Remotable._remote_backend`. The
full recipe — including the async dual-call pattern and how saved values are
pushed back — is `docs/developing/adding-a-new-backend.md`.

**Runtimes.** A new runtime subclasses the mixin chain from
[Subclassing NNsight and Envoy](#101-subclassing-nnsight-and-envoy) and fills in
the underscore-prefixed extension points that make it a real model:

- `_load` / `_load_meta` — construct the module (and its meta-device shape).
- `_batch_size` / `_batch` — report each invoke's row count and combine invokes
  into one call (see [Batching](#72-batching)); a `_batcher_class` if the batch
  layout is exotic (see [Custom batching](#104-custom-batching)).
- a `trace`/`_call` override to point the run at your engine's method, and an
  `interleave` override only if your runtime doesn't run a local forward.
- `eproperty` values for engine-internal outputs, served with `.provide` (see
  [Custom hookable values (eproperty)](#102-custom-hookable-values-eproperty)).
- the `Remotable` hooks (`_remoteable_model_key`, `_remoteable_persistent_objects`,
  ...) if it should run on NDIF.

`VLLM` (`src/nnsight/modeling/vllm/vllm.py`) is the deepest reference — a
non-PyTorch engine, two processes, async and serve backends — and shows every
one of these filled in: it starts no local workers in `interleave` (they're
serialized onto the engine's requests), and it surfaces `.logits`/`.samples` as
eproperties served from the model runner. `TransformersModel` and `DiffusionModel`
are the pipeline-backed references. The full walkthrough is
`docs/developing/adding-a-new-runtime.md`.

## 11. Performance

### 11.1 The overhead model

nnsight's cost is per-value-access bookkeeping wrapped *around* the model's own
compute, and the model's compute dominates by orders of magnitude. When you read
or write a location, the machinery parks the intervention greenlet, switches to
the model, hands off when the location is reached, narrows the
activation to the invoke's rows, hands it to your code, takes back whatever you
wrote, widens it back into the full batch, and switches back. That is real work —
a greenlet switch, a hook call, a narrow/widen pair, an `eproperty`'s
preprocess — but it is measured in microseconds and, crucially, it is **constant
in model size**. It does not scale with parameter count.

Set that against what the model does between two accesses: a single
`torch.nn.Linear` on a real hidden size is a matmul that dwarfs thousands of
descriptor dispatches. For any real model, where a forward pass takes
milliseconds to seconds, the interleaving pipeline is a negligible fraction of
wall-clock time — the overwhelming majority of a profiled trace is the model's
own matmuls, and the interleaver/eproperty/batcher machinery is a small,
fixed-cost sliver. The overhead only becomes visible in the opposite regime:
tight loops over *tiny* models, where the forward itself is a handful of trivial
matmuls and the constant pipeline cost is no longer hidden beneath it. That
regime is exactly what the benchmark harness under `tests/performance/`
isolates, by wrapping a stack of small linear layers so what remains in the
timings is the pipeline rather than the model.

One consequence worth internalizing: the per-module controller nnsight installs
is persistent, but it **no-ops when you are not interleaving** — and, because it
is the module's `forward` rather than a hook, the module stays on PyTorch's fast
call path. A model that has been traced still carries its controllers; outside a
`with model.trace()` block they cost one frame and one check per module call, so a
model isn't "slowed down" by having been used with nnsight. The cost is paid per
trace, and inside a trace it is paid per value you actually touch.

### 11.2 Best practices

The single largest win is **consolidating traces**. Each trace pays its setup
cost — building the interleaver, scoping the invokes, starting and parking the
worker greenlets — once, regardless of how much happens inside. So loop *inside*
one trace, not many traces in a loop:

```python
import nnsight

# Bad — one trace per layer, pays interleaver setup N times
hiddens = []
for layer in model.transformer.h:
    with model.trace(prompt):
        h = layer.output.save()
    hiddens.append(h)

# Good — one trace, setup paid once
with model.trace(prompt):
    hiddens = nnsight.save([layer.output for layer in model.transformer.h])
```

The good version also shows the **save-the-container idiom**: build the list of
locations you want inside the trace and call `.save()` on the *container*, rather
than saving each element separately. One saved name carries the whole list back.

A few more habits keep a trace cheap and correct:

- **Bound your iteration.** An unbounded `iter[:]` or `all()` runs the loop for
  the whole generation and drops everything sequenced after it, so trailing code
  never runs. Use a bounded `iter[:N]` when you need per-step values *and* a final
  result afterward (see [Interleaving](#4-interleaving)).
- **Capture forward tensors before a backward block.** A value read during the
  forward is available inside the trace; reaching for it again after a `backward()`
  block has run risks reading a location the model has already passed. Save it
  while it's live.
- **Read only what you need.** Every location you touch is a park/switch/serve
  round trip. Saving a handful of specific outputs is cheaper than caching every
  module — reach for `tracer.cache(...)` when you genuinely want breadth, not as a
  default.
- **Prefer `TransformersModel` / `NNsight`** over the deprecated aliases; that's a
  construction-warning concern, not a runtime one, but it keeps you on the
  supported path.

The reference for all of this is `docs/developing/performance.md`.

### 11.3 Profiling

Because the overhead model tells you the model's own ops should dominate, the
useful question when a trace feels slow is *where the time actually goes* — model
compute or machinery — and a plain `cProfile` around a realistic workload
answers it directly. Profile a function that runs the trace, not a single
statement, and look at the cumulative time attributed to the model's forward
versus the nnsight frames (`interleave`, `handle`, `narrow`/`widen`, the
`eproperty` `__get__`/`__set__`).

```python
import cProfile, pstats

def workload():
    with model.trace(prompt):
        hiddens = nnsight.save([layer.output for layer in model.transformer.h])
    return hiddens

workload()   # warm up: the first trace at a site pays block capture + compile once

profiler = cProfile.Profile()
profiler.enable()
for _ in range(20):
    workload()
profiler.disable()
pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)
```

Two things make a profile trustworthy. **Warm up first** — the first trace at a
given source location parses and compiles the block (memoized thereafter), and
`.source`'s first access compiles the instrumented forward; both are one-time
costs that will distort a cold profile. And, on GPU, call
`torch.cuda.synchronize()` before and after the timed region, or the model's
async kernels will be misattributed and the machinery will look artificially
expensive. One more practical constraint: block capture reads the block's source
via `inspect`, so define the trace-using function at module level in a real
file — it will not work from `python -c "..."` or a heredoc.

For comparing two nnsight trees rather than profiling your own workload, the
harness under `tests/performance/` (`interleave_bench.py` plus `compare.py`) is
the source of truth. It isolates each cost — capture warm vs cold, per-invoke
slope, per-intervention slope, `.source` steady state — and reports medians in
microseconds. Take **ratios**, not absolutes: the numbers are machine- and
version-dependent, so a ratio near 1.0 means "the same," and any small difference
is worth re-running before you trust it. Full documentation of the harness and
what each benchmark isolates is in `docs/developing/performance.md`.
