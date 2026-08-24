---
title: With Block Not Found
one_liner: "WithBlockNotFoundError: the tracer call isn't used as a `with` block, or its source line couldn't be recovered as one."
tags: [error, tracing, ast]
related: [docs/errors/index.md, docs/errors/debug-mode.md, docs/concepts/deferred-execution.md]
sources: [src/nnsight/tracing/tracer.py:51, src/nnsight/tracing/tracer.py:270, src/nnsight/tracing/tracer.py:306]
---

# With Block Not Found

## Symptom

```
nnsight.tracing.tracer.WithBlockNotFoundError
```

The exception carries **no message** — the class docstring is the
explanation: "The tracer call isn't used as a `with` block, so there's nothing to
trace."

> Older nnsight raised `WithBlockNotFoundError: With block not found at line <N>`
> with a source-context window. That message text is gone.

## Cause

`Tracer.capture()` (`src/nnsight/tracing/tracer.py:270`) reads the source of the
frame that entered the tracer, parses it, and looks for a `with` / `async with`
node at the exact line the tracer was created on. If there is no such node there,
it raises `WithBlockNotFoundError` (`src/nnsight/tracing/tracer.py:306`).

The block's *body* is what nnsight actually runs (interleaved with the model), and
it can only get that body by finding the `with` statement in the source. No `with`
at that line means there is nothing to capture. The verdict (found / not found) is
memoized per call site, so a non-`with` site fails the same way every time without
re-parsing.

## Common triggers

- **Not used as a `with` block.** Calling `model.trace(x).__enter__()`, or storing
  the tracer and entering it by hand, so the entering line isn't a `with`.
- **Dynamically-built source.** Code `exec`/`compile`-d from a string that isn't
  registered in `linecache`, so the recovered source doesn't match the line
  numbers — the line nnsight looks at isn't the `with` it expected. (Under a
  `python -c` launch, an *unrelated* dynamic `exec` deliberately surfaces this
  rather than mis-tracing the wrong body.)
- **Edited-file / stale-source edge cases.** nnsight reads each file's source once
  and never re-validates it (`Tracer.source`), so if the recovered source is out of
  sync with the running code object, the expected line may not be a `with`.

## Fix

```python
# WRONG — tracer entered by hand, not as a `with`
t = model.trace("Hello")
t.__enter__()                       # WithBlockNotFoundError
```

```python
# FIXED — use it as a context manager
with model.trace("Hello"):
    out = model.lm_head.output.save()
```

- **Run from a real `.py` file** (or a notebook cell) instead of `exec()`-ing a
  string that contains `with model.trace(...)`.
- **For programmatic intervention**, build the trace body as a real function in a
  real file and call it, rather than generating source at runtime.
- If the error reproduces from a clean state with an ordinary `with` block, it is
  likely a bug — open an issue with a minimal reproducer.

## Related

- [docs/concepts/deferred-execution.md](../concepts/deferred-execution.md) — why nnsight needs the block's source in the first place.
- [debug-mode.md](debug-mode.md) — traceback handling.
