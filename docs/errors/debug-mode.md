---
title: Debug Mode and Traceback Reconstruction
one_liner: "How nnsight wraps, hides, and reconstructs exceptions — and how to turn that off so you can see what really happened."
tags: [error, debug, traceback]
related: [docs/errors/index.md, docs/concepts/threading-and-mediators.md, docs/developing/tracing-pipeline.md]
sources: [src/nnsight/__init__.py:79, src/nnsight/__init__.py:111, src/nnsight/intervention/tracing/util.py:94, src/nnsight/intervention/tracing/util.py:132, src/nnsight/intervention/tracing/util.py:218, src/nnsight/intervention/tracing/util.py:278, src/nnsight/intervention/tracing/util.py:308, src/nnsight/schema/config.py:28]
---

# Debug Mode and Traceback Reconstruction

This is the comprehensive reference for understanding nnsight error output. nnsight handles exceptions in a way that is unusual compared to normal Python: it wraps, suppresses, reconstructs, and re-presents them. Understanding the full pipeline is crucial when an error message looks wrong, looks doctored, or doesn't tell you what you actually need to know.

## Why nnsight does this

Intervention code is **not** executed where you wrote it. nnsight extracts the source via AST, compiles it into a function with a synthetic filename like `<nnsight_...>`, and runs it on a worker thread. A naive Python traceback would point at the worker thread's compiled stub instead of your actual source line — so nnsight reconstructs the traceback to look like a normal exception thrown from your code.

The trade-off: by default nnsight **hides** its internal frames so the user-facing traceback is clean. If you need to see what actually happened internally, you need to turn on `DEBUG`.

## The components

### `ExceptionWrapper` (`src/nnsight/intervention/tracing/util.py:94`)

The core wrapper class. Every exception thrown from inside a worker thread passes through here. It carries:

- `self.original` — the actual underlying exception (e.g. the `IndexError` you threw).
- `self.infos` — a list of `Tracer.Info` objects. One per nesting level of trace context (each `with model.trace(...)` adds one). Used to map synthetic `<nnsight...>` frames back to user source.

Key methods:

- `_collect_frames(outer_tb=None)` (`util.py:132`-`235`) — builds the traceback frames list. This is where the frame filtering happens: with `CONFIG.APP.DEBUG = False`, frames whose filename contains `nnsight/` (i.e. library internals) are skipped. With `DEBUG = True`, they're kept.
- `format_traceback(outer_tb=None)` (`util.py:237`) — turns the collected frames into the printed traceback string.
- `print_exception(file, outer_tb)` (`util.py:264`) — used by the excepthooks below.

### `wrap_exception(exception, info)` (`src/nnsight/intervention/tracing/util.py:278`)

Wraps a thrown exception in a `NNsightException` (a dynamic subclass — see next).

If the exception is **already** an `ExceptionWrapper` (because it bubbled through one trace boundary already and is now hitting another nested one), it just appends another `Tracer.Info` to `self.infos`. This is how nested `model.session()` / `model.trace()` blocks accumulate the source-mapping info needed for traceback reconstruction.

### The dynamic `NNsightException` class (`src/nnsight/intervention/tracing/util.py:308`)

When `wrap_exception` is called for the first time, it builds a **dynamic class** at runtime:

```python
class NNsightException(exception_type, ExceptionWrapper):
    __qualname__ = "NNsightException"
    __module__ = "nnsight"
    ...
```

This class **multiply-inherits** from the original exception type **and** `ExceptionWrapper`. The result:

- `isinstance(e, IndexError)` still returns `True` — the user's `try / except IndexError:` continues to work.
- `isinstance(e, ExceptionWrapper)` also returns `True` — the excepthooks (see below) can dispatch on this to apply the reconstructed traceback.
- `e.original` is the underlying exception you can inspect directly.

The class is built fresh per exception type. There's no caching — building a class is cheap and the alternative would risk leaking unrelated state.

### `sys.excepthook` integration (`src/nnsight/__init__.py:82`-`96`)

When you `import nnsight`, it replaces `sys.excepthook` with `_nnsight_excepthook`:

```python
def _nnsight_excepthook(exc_type, exc_value, exc_tb):
    if isinstance(exc_value, ExceptionWrapper):
        exc_value.print_exception(file=sys.stderr, outer_tb=exc_tb)
    else:
        _original_excepthook(exc_type, exc_value, exc_tb)
```

For `ExceptionWrapper` subclasses, the printed traceback is the reconstructed view from `_collect_frames`. For everything else, the original Python excepthook fires unchanged.

### IPython integration (`src/nnsight/__init__.py:99`-`113`)

IPython has its own exception handling and bypasses `sys.excepthook`. nnsight registers a parallel handler:

```python
_ipython.set_custom_exc((ExceptionWrapper,), _nnsight_ipython_exception_handler)
```

`set_custom_exc` lets you register a callback for a specific exception type. nnsight registers it for `ExceptionWrapper`; the handler calls `evalue.print_exception(file=sys.stderr, outer_tb=tb)`. With `DEBUG = True`, the same hook fires but `_collect_frames` no longer skips internal frames.

## What `DEBUG` actually changes

`CONFIG.APP.DEBUG = True` flips two filters in `ExceptionWrapper._collect_frames`:

1. **Outer frames** (`util.py:155`-`159`) — frames *before* the trace boundary, i.e., frames in nnsight code that called into the user's trace setup. With `DEBUG = True`, these are kept; otherwise skipped if the filename contains `nnsight/`.
2. **Inner frames** (`util.py:216`-`223`) — frames *inside* the trace, in nnsight library code (`interleaver.py`, `mediator.py`, etc.). With `DEBUG = True`, these are kept; otherwise dropped.

What you see with `DEBUG = True`:

- The full call path through `interleaver.py`, `mediator.py`, the worker thread's compiled stub, the AST-rewritten module forward (if hooking via `.source`), etc.
- Useful when the user-facing traceback hides where in the pipeline a problem happened.

What you see without `DEBUG`:

- A clean traceback that looks like the exception was thrown directly from your source line. Internal frames are skipped.
- Useful in 99% of cases — the real fault is almost always your code, not nnsight's.

## When to enable DEBUG

- The user-facing traceback points at the right line but you can't tell **why** that line raised. (Is the exception coming from the worker, the hook, the batcher, or the user's intervention?)
- Working on nnsight internals — `_collect_frames` is where most frame elision happens, so DEBUG shows you the actual call path.
- An exception is being raised but the user-facing message looks reconstructed/corrupted (e.g. mismatched line numbers, code that doesn't match the file).
- You want to confirm a specific code path inside nnsight executed.
- Filing a bug report — include the `DEBUG = True` traceback so maintainers can see the internal stack.

## How to enable

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True
```

This affects all subsequent traces in the current process. The default is `False` (`src/nnsight/schema/config.py:28`). If you want it persistent across runs:

```python
nnsight.CONFIG.save()    # writes to ~/.nnsight or src/nnsight/config.yaml
```

`CONFIG.save()` writes the full config to YAML on disk. It is **persistent until clobbered** — a future explicit `CONFIG.save()`, the user editing the file by hand, or `set_default_app_debug(False)` followed by another save. Be intentional: if you turn `DEBUG` on for a one-off debugging session, turn it off and re-save before deploying.

To turn it off again:

```python
nnsight.CONFIG.APP.DEBUG = False
nnsight.CONFIG.save()
```

CLI shortcut: `python -d your_script.py` — nnsight's `from_cli()` (`src/nnsight/schema/config.py:81`) checks `sys.argv` for `-d` / `--debug` and flips `DEBUG` on for the process. This is **not** persisted to YAML.

## Programmatic access to the original exception

When you `try`/`except`, the caught object is the dynamic `NNsightException` subclass; `.original` gives you the underlying exception (`util.py:113`):

```python
try:
    with model.trace("Hello"):
        h = model.transformer.h[100].output.save()  # IndexError
except IndexError as e:
    print(type(e).__name__)   # NNsightException
    print(e.original)         # the real IndexError
    print(type(e.original))   # <class 'IndexError'>
```

You can use this to escalate or re-raise the underlying exception without the wrapper:

```python
try:
    with model.trace(...):
        ...
except Exception as e:
    if hasattr(e, "original"):
        raise e.original
    raise
```

You generally don't need to do this — `isinstance(e, IndexError)` works on the wrapper directly because of the multiple inheritance.

## Inspecting `ExceptionWrapper` directly

In a debugger or a `try/except`, you can pull apart the wrapper to understand what nnsight is presenting:

```python
try:
    with model.trace(...):
        ...
except Exception as e:
    if isinstance(e, ExceptionWrapper):
        # e.original — the original exception object
        # e.infos    — list of Tracer.Info, one per nested trace
        # e._collect_frames() — list of (filename, lineno, name, code_line, is_internal)
        for f in e._collect_frames():
            print(f)
        # e.format_traceback() — the string nnsight would print
        print(e.format_traceback())
```

`e.format_traceback(outer_tb=...)` accepts the outer Python traceback to interleave caller frames; without it you get only the post-trace frames.

## How DEBUG mode interacts with `e.original`

`e.original` is the same object regardless of `DEBUG`. It's the unwrapped exception, with whatever traceback Python attached at the throw site (i.e. the worker-thread synthetic `<nnsight...>` frame).

`DEBUG` only changes what `_collect_frames` and `format_traceback` produce — so it changes what the *excepthooks* print, not what the underlying exception object contains.

## Related

- [docs/errors/index.md](index.md) — index of nnsight error types.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — why intervention code runs on a worker thread.
- [docs/developing/tracing-pipeline.md](../developing/tracing-pipeline.md) — the capture → parse → compile → execute pipeline that produces synthetic `<nnsight...>` frames.
- `src/nnsight/intervention/tracing/util.py` — the full traceback reconstruction implementation.
