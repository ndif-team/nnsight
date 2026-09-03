---
title: Debug Mode and Tracebacks
one_liner: "Why a trace's traceback is short, what CONFIG.APP.DEBUG puts back, and how to read a traceback that ends inside torch."
tags: [error, debug, traceback]
related: [docs/errors/index.md, docs/reference/config.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/tracing/util.py, src/nnsight/tracing/tracer.py, src/nnsight/schema/config.py, src/nnsight/intervention/errors.py]
---

# Debug Mode and Tracebacks

## Why the traceback is short

Intervention code doesn't run where you wrote it. nnsight captures the `with`
block's body, compiles it, and runs it in a greenlet worker interleaved with the
model's forward pass, so a raw traceback arrives buried under nnsight's own
machinery. When a trace body raises, `clean_traceback`
(`src/nnsight/tracing/util.py`) drops the frames whose file lives inside the
nnsight package and keeps everything else.

For an error raised in the block itself that leaves three frames, ending on the
line you wrote:

```
  File "script.py", line 22, in <module>
    with model.trace(prompt):
  File "…/nnsight/tracing/tracer.py", line 521, in __exit__
    raise exception.with_traceback(
  File "script.py", line 23, in <module>
    h = model.transformer.h[100].output.save()
IndexError: list index out of range
```

The exception **type is preserved** — the real exception propagates, there is no
wrapper class and no `.original` attribute:

```python
try:
    with model.trace("Hello"):
        h = model.transformer.h[100].output.save()   # IndexError
except IndexError as error:
    print(type(error).__name__)   # IndexError
```

`Mediator.switch` also stashes the intervention-only traceback on the exception as
`__intervention_tb__` before the model and controller frames pile on during
unwinding, which is what lets the surfaced trace point at the exact intervention
line. `InterleavingTracer.traceback` and the deferred-error path prefer that
stashed traceback.

## Reading a traceback that ends inside torch

Only *nnsight's* frames are dropped. Frames from torch, transformers, and your own
helper functions all stay, so a failure that happens inside the model because of
what an intervention wrote arrives with the model's stack intact:

```python
with model.trace(prompt):
    acts = model.transformer.h[0].output
    model.transformer.h[0].output = torch.zeros(3, 3, device=acts.device)
```

```
RuntimeError: Given normalized_shape=[768], expected input with shape [*, 768], but got input of size[3, 3]
```

That traceback runs to nineteen frames and ends in `torch.layer_norm`. The line that
wrote the bad value is not in it at all — the write succeeded; the *next* module
is what failed. Read it from the top: the `with` line names the block, and the
bottom names the module that could not use what the block put there. Then look at
your writes for a shape that does not match the activation it replaces.

Building the replacement from the activation avoids the whole class of them:

```python
with model.trace(prompt):
    acts = model.transformer.h[0].output
    model.transformer.h[0].output = torch.zeros_like(acts)
```

## What `CONFIG.APP.DEBUG` does

1. **Keeps nnsight's frames.** `clean_traceback` returns the traceback untouched
   under `DEBUG`, so the interleaver, the controller, and the backend are all
   visible. Measured on GPT-2: an `IndexError` or an `OutOfOrderError` raised in a
   block goes from 3 frames to 13, and the layer-norm failure above from 19 to 33.
   Turn it on when you suspect the bug is in nnsight's plumbing rather than in
   your intervention.
2. **Verbose remote logging.** `RemoteBackend.__init__` sets
   `self.verbose = verbose or CONFIG.APP.DEBUG`, so remote runs log payload and
   result byte sizes and print each status update on its own line.

## How to set it

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True     # this process
```

The flag is read when the exception is raised, so setting it after import works.

On the command line, pass `-v` / `--verbose` (a plain `sys.argv` scan at import, so
any launcher using `-v` — e.g. `pytest -v` — enables it too):

```bash
python your_script.py -v
```

Environment variable:

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
`save()` again. Default is `False`. `CONFIG.APP.REMOTE_LOGGING` (default `True`)
controls the remote status display independently of `DEBUG`. See
[docs/reference/config.md](../reference/config.md).

## Errors from a deferring driver

A driver that keeps running past a worker's error — vLLM, whose engine schedules
the next step itself — records the error on the mediator rather than raising it out
of the controller. It comes back at the client as a `RuntimeError` whose message
begins with the original type name and message, followed by the intervention
traceback (`raise_deferred`, `src/nnsight/intervention/errors.py`). The original
class is not reconstructed across the process boundary, so match on the message:

```python
except RuntimeError as error:
    if str(error).startswith("IndexError"):
        ...
```

## Related

- [index.md](index.md) — the exceptions nnsight raises.
- [symptom-index.md](symptom-index.md) — failures that raise nothing at all.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — why intervention code runs in a greenlet worker.
