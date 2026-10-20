---
title: Tracing Pipeline
one_liner: Capture, parse, build, compile, execute — how the body of a with block becomes a code object nnsight runs on its own terms.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/interleaver-internals.md, docs/developing/backends.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/tracing/util.py, src/nnsight/tracing/globals.py, src/nnsight/tracing/backend.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/iterator.py, src/nnsight/intervention/editing.py]
---

# Tracing Pipeline

## What this covers

The pipeline that turns user code inside a `with model.trace(...): ...` block into
a code object the interleaver can run in a greenlet worker. The base `Tracer`
(`src/nnsight/tracing/tracer.py`) is model-agnostic — it imports no torch — and
does five small steps, each overridable by a subclass:

1. **Capture** — find the calling frame and its source.
2. **Parse** — locate the `with` node at the trace line.
3. **Build** — wrap the block *body* in a compilable module.
4. **Compile** — compile the body to a code object.
5. **Execute** — run the code object and push results back into the caller's frame.

Between compile and execute, `__enter__` arms a skip hook so the body does not run
inline; `__exit__` hands the compiled body to the backend.

## Architecture

Capture happens on `__enter__` (`tracer.py`). The tracer object is constructed
earlier when the user wrote `model.trace(...)`, but nothing is read until the
`with` actually enters. `capture()` is idempotent, so callers that want to sniff
whether a `with` block exists can call it early and again on entry — this is how
`traceable` (`envoy.py`) distinguishes `with envoy.method(...):` from a plain
`envoy.method(...)` call.

### Tracer.Info

`Tracer.Info` (`tracer.py`) is tiny — just what `execute` needs:

- `frame: FrameType` — the caller's live frame (its globals/locals, and its
  `co_filename`/lineno/name for tracebacks).
- `code: CodeType` — the compiled block body.

On serialization (`Info.__getstate__`) the live frame is replaced by a
`SerializedFrame` (`src/nnsight/tracing/util.py`) that keeps only the code
metadata; the real code object and captured scope are rebuilt on the remote side
from the source-reduced interventions payload (see
[serialization.md](./serialization.md)). The AST node lives on the tracer
(`tracer.node`), not in `Info`.

### The per-site cache

`capture()` builds a key from the calling frame and memoizes the parse+compile:

```python
key = (code.co_filename, frame.f_lineno, code.co_name, code.co_firstlineno)
if key not in BLOCKS:
    source = self.source(code.co_filename)
    node = self.parse(source, frame.f_lineno)
    compiled = None if node is None else self.compile(self.build(node), frame)
    BLOCKS[key] = (node, compiled)
```

`BLOCKS` (`src/nnsight/tracing/globals.py`) maps the site key to `(node, code)`. A
`with model.trace(...)` inside a Python loop is read, parsed, and compiled exactly
once. A site that turns out **not** to be a `with` block caches `(None, None)`, so
the negative verdict isn't re-derived every call —
`capture` raises `WithBlockNotFoundError` (`tracer.py`) on that case.

`SOURCES` (`globals.py`) caches raw file text keyed by filename. Neither cache
is evicted or re-validated against on-disk changes — a file edited mid-run is
traced as it was first seen. Both are process-wide.

### Finding the caller's frame

`capture()` looks two frames up: `sys._getframe(2)` (`tracer.py`), i.e. past
`__enter__` to the user's frame — a fixed depth, not a stack walk. Subclasses that
enter at a different depth compensate directly: `EditingTracer.__enter__` (`editing.py`)
mirrors `Tracer.__enter__` inline rather than calling `super().__enter__()` so
capture sees the user's frame at the same depth.

### Source extraction

`Tracer.source(filename)` (`tracer.py`) returns the file text, read at most
once and cached in `SOURCES`. It handles three contexts:

1. **Files and IPython/Jupyter cells** — both come from
   `linecache.getlines(filename)`. IPython registers each cell in `linecache` under
   the frame's filename with no mtime, so the exact compiled source is available
   and line numbers line up. `checkcache` is deliberately *not* called, so a file
   saved mid-run doesn't shift line numbers under a running trace.
2. **`python -c "<code>"`** — no file on disk; the program compiles under filename
   `<string>`, so the literal source is recovered from `sys.orig_argv` (gated on
   the exact `<string>` filename).
3. Anything else with no source (e.g. a raw `exec` string, `<stdin>`) yields an
   empty string, which fails the parse and raises `WithBlockNotFoundError`. Its
   message is built by `_not_found_message` (`tracer.py`), which branches on
   whether any source was recovered and whether the filename is a file still on
   disk, since the way out differs for each.

> **Running examples:** because the block is captured from source, a trace does
> not work from a heredoc, `exec` string or anything else with no registered
> source. Write snippets to a `.py` file and run the file.

### Parse

`Tracer.parse(source, lineno)` (`tracer.py`) finds the `ast.With` /
`ast.AsyncWith` node entered from `lineno`: the `with` line on Python 3.10/3.11,
the line the tracer's own item starts on from 3.12, which in a parenthesized or
backslash-continued header sits below the `with`. It first tries `_parse_block`
(`tracer.py`), which slices just the block out by indentation and bracket depth,
dedents it, parses it, and shifts line numbers back; an unparseable slice falls
back to parsing the whole file. The one dangerous error is a slice that ends early
yet still parses (a `with` missing part of its body), which is why blank lines and
comments never end the slice. `parse` returns `None` if there is no `with` at that
line.

### Build + compile

`build(node)` (`tracer.py`) wraps the block's **body** (not the `with` line) in
an `ast.Module` and `fix_missing_locations`. `compile(module, frame)`
(`tracer.py`) compiles it under the original frame's filename and sets
`co_name` to the frame's, so tracebacks read as if the body ran where it was
written.

### The skip hook

If the block has real code (`skippable`, `tracer.py` — a body that is only
`pass`/docstring/`...` is left to run harmlessly), `__enter__` calls `skip_context`
(`tracer.py`). It sets a per-frame `f_trace` hook that raises
`ExitTracingException` the moment execution reaches the body's first line, and sets
a no-op global `sys.settrace` so Python actually delivers per-frame events.

Two constraints fall out of this line-based mechanism:

- The body must start on its own line. A body written on the `with` line would run
  where it stands and again through the backend, so `skip_context` raises a
  `ValueError` refusing it.
- A multi-line `with` header (arguments over several lines) is tolerated: the hook
  stays armed while `frame.f_lineno < body_lineno` and only fires at the body, so
  the `as` target still binds before the block is skipped.

### Execute (varies by tracer type)

`Tracer.execute(code)` (`tracer.py`, the base) runs the block against a `Scope`
built from the caller's locals, then `push_result` writes results back:

```python
scope = Scope(dict(frame.f_locals), frame.f_locals, frame.f_globals)
exec(code, scope)
push_result(frame, scope)
```

Subclasses override `execute` to change *how* the block runs:

- **`InterleavingTracer.execute`** (`src/nnsight/intervention/tracer.py`) —
  builds a `Batcher`, makes the worker(s), and hands them to `Envoy.interleave` to
  run alongside the model's forward. Two shapes: direct input (`trace(x)` → one
  implicit invoke over the whole block) vs invoke mode (`trace()` → the body is run
  now to collect `tracer.invoke(...)` sub-blocks, each a worker scoped to its rows).
- **`ScanningTracer.execute`** (`tracer.py`) — defers to
  `InterleavingTracer.execute` inside a `FakeTensorMode`, so the forward propagates
  only shapes/dtypes and needs no real weights.
- **`Invoker.execute`** (`tracer.py`) — registers its input as a batch group
  and its body as a worker on the parent tracer's interleaver; the parent runs them.
- **`EditingTracer.execute`** (`editing.py`) — stores the block as a
  `Mediator` on `envoy._edits` (with `copy=True`) instead of running it; the edit
  is replayed on every later trace.
- **`Iterations.execute`** (`iterator.py`) — the deprecated
  `with tracer.iter[...]:` form; re-runs the block once per pinned occurrence. The
  `for step in tracer.iter[...]:` form is a plain loop and does not go through
  `execute` at all.

### Scope: how a captured block reaches names

`Scope` (`src/nnsight/tracing/util.py`) is the namespace a captured block runs
in. Because the block runs later than the line it was written on — and, once
interleaving, in a greenlet — the names it reaches come from three places, in
order:

1. **A snapshot** of the frame's locals taken at capture (so a `for prompt in
   prompts:` variable means what it meant *where the block was written*).
2. `shared` — the live locals of the frame the block was written in, so a name
   bound by a *sibling* block (an earlier `tracer.invoke(...)`) is visible.
3. `glbls` — the frame's globals, by fallback via `__missing__`.

`Scope` is passed as `exec`'s *globals* (not just locals), so a `lambda` or nested
`def` inside the block reaches the block's own names (their free variables compile
to `LOAD_GLOBAL`, which honors `__missing__` on a dict subclass). Writes land in
the scope *and* in `shared`, so a block sees its own assignments and so do the
blocks written beside it.

### push_result and saving

`push_result(frame, variables)` (`tracer.py`) writes a block's variables back
into the caller's frame. `save` (`tracer.py`, exposed as `nnsight.save` and
behind `.save()`) marks a value by identity in a per-thread set; the **outermost**
trace filters `push_result` to only the marked values, while a nested trace pushes
everything up to its enclosing block. Depth is tracked around the backend call in
`__exit__` (`inc()`/`dec()`, `tracer.py`). An interleaving trace pushes only what
each block bound (`Scope.writes()`, `util.py`): every invoke's scope starts from the
same snapshot of the frame, so pushing a scope whole would put the old value back
over a sibling's rebind. `push` marks what a nested trace writes into a block's
scope as bound by that block, so it rides on up too.

`save()` **raises** if called with no trace running (`tracer.py`):

```text
save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to
return from the enclosing `with model.trace(...):` block, so it only works inside
one — move the save into the trace block.
```

`mark(value)` (`tracer.py`) is the same marking without the guard, for internal
callers that record return values outside a running trace (e.g. a remote backend
recording a finished request's saves).

### Frame write-back (push)

`push(frame, variables)` (`src/nnsight/tracing/util.py`) copies results back
into the live frame. On Python < 3.13 it calls the C-API `PyFrame_LocalsToFast` to
flush the change through the fast-locals array; from 3.13 on `f_locals` is a live
write-through mapping (PEP 667) and a plain `update` suffices.

## Key files / classes

- `src/nnsight/tracing/tracer.py` — `Tracer` (`Info`, `capture`, `source`, `parse`,
  `build`, `compile`, `execute`), `skip_context`, `save`, `mark`, `push_result`.
- `src/nnsight/tracing/util.py` — `Scope`, `push`, `SerializedFrame`.
- `src/nnsight/tracing/globals.py` — `SOURCES`, `BLOCKS`.
- `src/nnsight/intervention/tracer.py` — `InterleavingTracer`, `ScanningTracer`, `Invoker`.
- `src/nnsight/intervention/iterator.py` — `Iterations`; `editing.py` — `EditingTracer`.

The end-to-end sequence for one trace is in
[architecture-overview.md](./architecture-overview.md#lifecycle--sequence); new
tracer types and backends are in its
[extension points](./architecture-overview.md#extension-points). Two extension
points are specific to this page: override `build`/`compile` only for a different
wrapping of the body (no nnsight subclass does), and add a branch in
`Tracer.source` to capture source from a new filename shape (e.g. a hosted-notebook
cell store).

## Related

- [serialization.md](./serialization.md) — how `Info` and the block survive remote transmission.
- [deferred-execution.md](../concepts/deferred-execution.md) — the mental-model version.
