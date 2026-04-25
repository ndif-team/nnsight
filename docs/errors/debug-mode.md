---
title: Debug Mode and Traceback Reconstruction
one_liner: "Set CONFIG.APP.DEBUG = True to keep nnsight internal frames in tracebacks (instead of the user-facing reconstructed view)."
tags: [error, debug, traceback]
related: [docs/errors/index.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/__init__.py:79, src/nnsight/intervention/tracing/util.py:94, src/nnsight/intervention/tracing/util.py:156, src/nnsight/intervention/tracing/util.py:218]
---

# Debug Mode and Traceback Reconstruction

## How nnsight tracebacks normally look

Intervention code is captured by AST, compiled into a function with a synthetic filename like `<nnsight_...>` and run on a worker thread. By default, an exception thrown from that worker is wrapped in `ExceptionWrapper` (`src/nnsight/intervention/tracing/util.py:94`) and presented with a traceback that:

1. Skips frames whose filename contains `nnsight/` (i.e., library internals).
2. Rewrites `<nnsight...>` synthetic frames back to the original user file and line numbers.
3. Shows the original exception type and message at the bottom.

The exception type is preserved — `isinstance(e, ValueError)` still works — because `wrap_exception` builds a dynamic subclass that inherits from both `ExceptionWrapper` and the original exception type (`src/nnsight/intervention/tracing/util.py:308`).

## What DEBUG mode changes

`CONFIG.APP.DEBUG = True` flips the filtering in `ExceptionWrapper._collect_frames` (`src/nnsight/intervention/tracing/util.py:156` and `:218`):

- Outer frames whose filename contains `nnsight/` are kept instead of skipped.
- Internal nnsight frames mid-trace are included in the reconstructed traceback.

You see the full path through `interleaver.py`, `mediator.py`, the worker thread's compiled stub, etc. — useful when the user-facing traceback hides where in the pipeline a problem happened.

## How to enable

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True

# Optional: persist for future processes
nnsight.CONFIG.save()
```

To turn it off again:

```python
nnsight.CONFIG.APP.DEBUG = False
nnsight.CONFIG.save()
```

## When to use it

- The user-facing traceback points at the right line but you can't tell **why** that line raised (e.g., is the exception coming from the worker, the hook, the batcher, or the user's intervention?).
- Working on nnsight internals — most frame elision happens in `_collect_frames`, so DEBUG shows you the actual call path.
- An exception is being raised but the user-facing message looks reconstructed/corrupted.

## How the hook is wired

`src/nnsight/__init__.py:82` replaces `sys.excepthook` with `_nnsight_excepthook`:

```python
def _nnsight_excepthook(exc_type, exc_value, exc_tb):
    if isinstance(exc_value, ExceptionWrapper):
        exc_value.print_exception(file=sys.stderr, outer_tb=exc_tb)
    else:
        _original_excepthook(exc_type, exc_value, exc_tb)
```

In IPython, an analogous handler is registered via `set_custom_exc((ExceptionWrapper,), ...)` (`src/nnsight/__init__.py:111`). With `DEBUG = True`, the same hook still fires but `_collect_frames` no longer drops internal frames, so you see the full trace.

## Programmatic access to the original exception

When you `try`/`except`, the caught object is the dynamic subclass; `.original` gives you the underlying exception (`src/nnsight/intervention/tracing/util.py:113`):

```python
try:
    with model.trace("Hello"):
        h = model.transformer.h[100].output.save()  # IndexError
except IndexError as e:
    print(type(e).__name__)   # NNsightException
    print(e.original)         # the real IndexError
```

## Related

- `docs/errors/index.md`
- `docs/concepts/threading-and-mediators.md`
- `src/nnsight/intervention/tracing/util.py` (full traceback reconstruction)
