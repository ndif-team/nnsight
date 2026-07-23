---
title: Debug Mode and Tracebacks
one_liner: "What CONFIG.APP.DEBUG does in this rewrite (remote verbose logging), and how trace tracebacks are cleaned so they point at your code."
tags: [error, debug, traceback]
related: [docs/errors/index.md, docs/concepts/threading-and-mediators.md, docs/remote/index.md]
sources: [src/nnsight/schema/config.py:19, src/nnsight/schema/config.py:100, src/nnsight/intervention/backends/remote.py:83, src/nnsight/tracing/util.py:139, src/nnsight/tracing/tracer.py:465]
---

# Debug Mode and Tracebacks

> This page was heavily rewritten. The old `ExceptionWrapper` / dynamic
> `NNsightException` / `sys.excepthook` / IPython `set_custom_exc` machinery **does
> not exist in this rewrite.** Exceptions from a trace body are the real exception
> objects; their tracebacks are cleaned by default and shown in full under `DEBUG`.

## What `CONFIG.APP.DEBUG` does now

`CONFIG.APP.DEBUG` has two effects:

1. **Full tracebacks.** By default an error raised inside a `with model.trace(...):`
   block has nnsight's own frames stripped, so it points at your code
   (`clean_traceback`, `src/nnsight/tracing/util.py`). With `DEBUG` on, nothing is
   stripped — the full stack, nnsight internals included, is shown. Turn it on when
   you suspect the bug is in nnsight's plumbing.
2. **Verbose remote logging.** `RemoteBackend.__init__` sets
   `self.verbose = verbose or CONFIG.APP.DEBUG`
   (`src/nnsight/intervention/backends/remote.py:83`), so remote runs log payload /
   result byte sizes and print each status update on its own line.

## How to set it

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True     # this process
```

On the command line, pass `-v` / `--verbose` (a plain `sys.argv` scan at import, so
any launcher using `-v` — e.g. `pytest -v` — enables it too):

```bash
python your_script.py -v
```

Environment variable (`src/nnsight/schema/config.py`):

```bash
NNSIGHT_DEBUG=1 python your_script.py
```

Persist it to the user config file (`~/.config/nnsight/config.yaml`, or
`$XDG_CONFIG_HOME/nnsight/config.yaml`, or `$NNSIGHT_CONFIG`):

```python
nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()
```

`save()` writes the whole config to YAML and persists until you change it back and
`save()` again. Default is `False` (`src/nnsight/schema/config.py:19`).

Related remote-logging switch: `CONFIG.APP.REMOTE_LOGGING` (default `True`) controls
the status display independently of `DEBUG`.

## How trace tracebacks are cleaned

Intervention code doesn't run where you wrote it — nnsight captures the `with`
block's body, compiles it, and runs it in a greenlet worker interleaved with the
model's forward pass. A raw traceback would be buried under nnsight and model
frames. So when a trace body raises, nnsight cleans the traceback with
`clean_traceback` (`src/nnsight/tracing/util.py:139`), which drops frames whose file
lives inside the nnsight package, leaving your own frames (across whatever files
they span). This happens in `Tracer.__exit__` (`src/nnsight/tracing/tracer.py:465`),
unconditionally — there is no flag to keep the internal frames.

The exception **type is preserved**: the real exception propagates directly, so

```python
try:
    with model.trace("Hello"):
        h = model.transformer.h[100].output.save()   # IndexError
except IndexError as e:
    print(type(e).__name__)   # IndexError
```

works as written. There is no wrapper class and no `.original` attribute — `e` *is*
the underlying exception.

### Pointing at the waiting line

When a worker raises, `Mediator.switch` (`src/nnsight/intervention/interleaver.py:366`)
stashes the intervention-only traceback on the exception as `__intervention_tb__`
*before* the model/hook frames pile on during unwinding, so the surfaced trace can
point at the exact intervention line. `InterleavingTracer.traceback` and the
deferred-error path (`src/nnsight/intervention/errors.py`) prefer that stashed
traceback.

## Deferred (remote / vLLM) worker errors

When a driver keeps running past a worker's error (e.g. vLLM, whose engine schedules
the next step itself), the interleaver records the error on the mediator instead of
raising it out of the hook (`Interleaver.defer_exceptions`). The error is reduced to
a wire-safe dict (`capture_exception`) and re-raised at the client as a
`RuntimeError` carrying the original type name, message, and intervention traceback
(`raise_deferred`, `src/nnsight/intervention/errors.py:45`) — the original class
isn't reconstructed across the process boundary.

## Related

- [docs/errors/index.md](index.md) — the exceptions nnsight raises.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — why intervention code runs in a greenlet worker.
