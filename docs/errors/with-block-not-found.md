---
title: With Block Not Found
one_liner: "WithBlockNotFoundError: the AST parser could not locate a `with` statement at the line where the tracer was created — usually means the source isn't recoverable for that frame."
tags: [error, tracing, ast]
related: [docs/errors/index.md, docs/errors/debug-mode.md, docs/concepts/deferred-execution.md, docs/developing/tracing-pipeline.md]
sources: [src/nnsight/intervention/tracing/base.py:36, src/nnsight/intervention/tracing/base.py:432, src/nnsight/intervention/tracing/base.py:262, src/nnsight/intervention/tracing/base.py:300]
---

# With Block Not Found

## Symptom

```
nnsight.intervention.tracing.base.WithBlockNotFoundError: With block not found at line <N>
We looked here:

  ...
  some surrounding source lines  <--- HERE
  ...
```

The error always carries the line number nnsight was looking for and a five-line context window with `<--- HERE` marking the offending line.

## Cause

`Tracer.capture()` (`src/nnsight/intervention/tracing/base.py:200`-`356`) reconstructs the source of the `with` block so it can be AST-parsed, compiled, and run on a worker thread. To do that it has to find a `with`/`async with` statement at the exact line where `Tracer.__enter__` was called.

The flow is:

1. Identify which "case" the caller lives in — IPython cell, regular `.py` file, `python -c` string, or the `<nnsight-console>` REPL (`base.py:255`-`310`).
2. Pull the source lines for that frame.
3. Parse them with `ast.parse`, walk for a `With`/`AsyncWith` node whose `context_expr.lineno == start_line` (`base.py:382`-`413`).
4. If no such node is found, raise `WithBlockNotFoundError` (`base.py:432`).

In short: nnsight could read the source, but the line number it expected to be a `with` statement turned out not to be one. This almost always means **the source nnsight is reading is not the source the line number was generated against** — they're out of sync.

## Common triggers

- **Dynamically-built source** — code constructed via `exec()` / `compile()` on a string that wasn't registered with `linecache`. nnsight has no way to recover the original `with` line.
- **Edited files mid-run** — the source on disk has changed since the interpreter loaded it. `inspect.getsourcelines` reads from disk; the line numbers come from the already-loaded code object. nnsight installs a `linecache.checkcache` no-op (`base.py:283`) to mitigate this for the duration of a trace, but if the file changed earlier the cached lines are still stale.
- **Notebook cell rewrites** — IPython cell content was replaced (e.g. by an autoformatter or a magic) between the cell being submitted and the trace running. nnsight reads `_ih[-1]` (`base.py:268`); if that doesn't match the actual line numbers, it fails here.
- **REPL with multi-line edits** — the standard Python REPL has `<stdin>` as filename; nnsight has no `linecache` entry for it. The `<nnsight-console>` REPL works because nnsight populates `__INTERACTIVE_CONSOLE__.buffer` itself (`base.py:300`-`309`).
- **Very narrow edge case: line numbers from a different source generation** — rewriting trace bodies via decorators, source-rewriting tools, or transpilers can produce code whose `co_firstlineno` doesn't correspond to any real `with` in the user-visible source.

## Fix

Most of the time, fix the source-recovery path:

- **Re-run from a fresh interpreter** if you've edited the file mid-process. The `linecache` will reload from disk on next invocation.
- **Save and run as a `.py` file** instead of using `exec()` on a string of code that contains a `with model.trace(...)`.
- **Move dynamic-source generation up one level**: build the trace function as a real def in a real file (or via `make_function` / `__source__` attribute, see `docs/developing/serialization.md`), then call it.
- **In Jupyter**, re-run the cell. If the cell was edited and submitted multiple times in quick succession the input history may be desynchronized — restart the kernel if "re-run" doesn't help.

If the error is reproducible from a clean state, this is likely an nnsight bug — open an issue with a minimal reproducer.

## Mitigation / how to avoid

- Avoid generating intervention code via `exec()` / `eval()` on strings — use real function defs and pass the function in.
- Don't edit the `.py` containing a `with model.trace(...)` while a long-running script is mid-trace.
- For programmatic/dynamic intervention, look at `nnsight.intervention.serialization.make_function` which is the supported path for "I have source as a string and want to register it for tracing."

## Related

- [debug-mode.md](debug-mode.md) — turn on `CONFIG.APP.DEBUG = True` to see the internal `Tracer.capture` frames if the message alone isn't enough.
- [docs/concepts/deferred-execution.md](../concepts/deferred-execution.md) — why nnsight needs the source in the first place.
- [docs/developing/tracing-pipeline.md](../developing/tracing-pipeline.md) — full description of `capture` -> `parse` -> `compile` -> `execute`.
