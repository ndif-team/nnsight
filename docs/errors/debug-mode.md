---
title: Debug Mode and Tracebacks
one_liner: "Why a trace's traceback is short, what CONFIG.APP.DEBUG puts back, and how to read a traceback that ends inside torch."
tags: [error, debug, traceback]
related: [docs/errors/index.md, docs/reference/config.md, docs/concepts/threading-and-mediators.md]
sources: [src/nnsight/tracing/util.py, src/nnsight/tracing/tracer.py, src/nnsight/schema/config.py, src/nnsight/intervention/errors.py]
---

# Debug Mode and Tracebacks

## Why the traceback is short

Intervention code runs in a worker interleaved with the model's forward pass, so a
raw traceback is buried under nnsight's own machinery. When a trace body raises,
nnsight strips the worker and interleaving frames. A few nnsight frames from the
`with` statement's exit remain; the last frame is the line you wrote:

```
  File "script.py", line 22, in <module>
    with model.trace(prompt):
  File "…/nnsight/intervention/tracer.py", line 109, in __exit__
    return super().__exit__(exc_type, exc_value, exc_traceback)
  File "…/nnsight/tracing/tracer.py", line 589, in __exit__
    raise exception.with_traceback(
  File "script.py", line 23, in <module>
    h = model.transformer.h[100].output.save()
AttributeError: 'Envoy' object (nor its module) has attribute '100'. Did you mean: '10'?
```

The exception **type is preserved** — the real exception propagates, with no
wrapper class, so `except AttributeError:` around the `with` catches it.

## Reading a traceback that ends inside torch

Only nnsight's own frames are stripped. Frames from torch, transformers, and your own
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

That traceback ends in `torch.layer_norm`, and the line that wrote the bad value is
not in it — the write succeeded; the *next* module failed. Read it from the top: the
`with` line names the block, the bottom names the module that could not use what the
block put there. Then check your writes for a shape that does not match the
activation it replaces. Building the replacement from the activation avoids this:

```python
with model.trace(prompt):
    acts = model.transformer.h[0].output
    model.transformer.h[0].output = torch.zeros_like(acts)
```

## What `CONFIG.APP.DEBUG` does

1. **Keeps nnsight's frames** (interleaver, controller, backend). Turn it on when
   you suspect nnsight's plumbing rather than your intervention.
2. **Verbose remote logging** — remote runs log payload and result sizes and print
   each status update on its own line.

## How to set it

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True     # this process; read when the exception is raised
```

Or `NNSIGHT_DEBUG=1`, or `-v` / `--verbose` on the command line (a plain `sys.argv`
scan, so `pytest -v` enables it too). To persist it, and for every other setting, see
[config.md](../reference/config.md).

## Errors from a deferring driver

A driver that keeps running past a worker's error — vLLM, whose engine schedules
the next step itself — records the error and keeps going. It comes back at the client as a `RuntimeError`
whose message begins with the original type name and message, followed by the
intervention traceback. The original class is not reconstructed across the process boundary, so match on the message:

```python
except RuntimeError as error:
    if str(error).startswith("IndexError"):
        ...
```

## Related

- [index.md](index.md) — the exceptions nnsight raises.
- [symptom-index.md](symptom-index.md) — failures that raise nothing at all.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — why intervention code runs in a greenlet worker.
