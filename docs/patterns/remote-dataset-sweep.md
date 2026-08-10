---
title: Sweeping a Dataset Remotely
one_liner: Load the data on NDIF instead of shipping it, loop over it inside one session, and bring back only the reduction.
tags: [pattern, remote, ndif, dataset, session]
related: [docs/remote/remote-session.md, docs/gotchas/remote.md, docs/patterns/remote-training.md]
sources: [src/nnsight/intervention/serialization.py, src/nnsight/modeling/mixins/remotable.py]
---

# Sweeping a Dataset Remotely

## What this is for

Running a model over hundreds or thousands of examples — collecting activations,
scoring a probe, measuring an intervention's effect across a benchmark. The
technique is ordinary; what makes it a pattern is where the bytes go.

Two rules carry almost all the benefit:

1. **Upload the minimum.** The dataset is already reachable from the server. Load
   it there rather than pickling it into the request.
2. **Download the minimum.** Reduce inside the session and save the summary, not
   the activations you reduced.

## When to use

- Any evaluation loop over more than a handful of prompts.
- Whenever the thing you want back is a number, a score per example, or a small
  tensor — not the hidden states themselves.

## Measure it

nnsight will tell you what actually crossed the wire, which is the only way to know
whether any of the below worked:

```python
from nnsight import CONFIG
CONFIG.APP.DEBUG = True
```

```
[remote] payload: 4,228 bytes (compressed)
[remote] result: 1,671 bytes downloaded
```

For reference, a 256-example sweep like the one below runs at ~4 KB up and ~1.7 KB
down. Shipping the same dataset from the client instead costs ~17 KB up; saving the
raw residual stream instead of a scalar would cost ~2 MB down, and raw logits ~131 MB.

## Load the data on the server

A HuggingFace `Dataset` is memory-mapped. Pickling one puts an **arrow file path**
in the payload, and that path only exists on your machine:

```
RemoteError: Your request payload could not be read (FileNotFoundError:
Failed to open local file '/home/you/.cache/huggingface/datasets/.../cache-ce54.arrow')
```

Import `datasets` *inside* the session instead. The block runs on the server, so
the download and the memory-mapping happen there:

```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")

with model.session(remote=True) as session:
    from datasets import load_dataset

    rows = load_dataset("nyu-mll/glue", "sst2", split="train[:40]")
    tally = nnsight.save({"n": 0, "hits": 0})

    for start in range(0, len(rows), 10):
        batch = rows[start : start + 10]
        with model.trace(batch["sentence"]):
            top = model.output.logits[:, -1].argmax(-1)
        tally["n"] += len(top)
        tally["hits"] += int((top > 0).sum())

    print(f"server saw {tally['n']} rows")

print(tally)
```

```
ℹ [235b1a59…] LOG          server saw 40 rows
{'n': 40, 'hits': 40}
```

The whole sweep is **one** request. `print` comes back as a `LOG` line while it
runs, so you can watch progress without saving anything.

Slicing in the split string (`split="train[:40]"`) is worth doing on principle: it
is the difference between the server materialising forty rows and the full 67k.

## Reduce before you save

The saved value is what gets serialized, uploaded to object storage, and pulled
back over HTTP. A per-layer residual stream for 500 examples is gigabytes; the
score you wanted from it is kilobytes.

```python
# wrong — ships every hidden state home to compute one number
with model.session(remote=True):
    states = nnsight.save([])
    for prompt in prompts:
        with model.trace(prompt):
            states.append(model.transformer.h[6].output.save())
score = torch.stack([s[0, -1] @ probe for s in states]).mean()

# right — the reduction runs next to the weights
with model.session(remote=True):
    scores = nnsight.save([])
    for prompt in prompts:
        with model.trace(prompt):
            hidden = model.transformer.h[6].output
            scores.append((hidden[0, -1] @ probe.to(hidden.device, hidden.dtype)).item())
```

Note `probe.to(hidden.device, hidden.dtype)`: `probe` was built on the client, so
it arrives on the CPU in float32 regardless of how the server holds the model. See
[gotchas/remote.md](../gotchas/remote.md).

## Accumulating into an outer variable

A container bound before the block works too, as long as you save it from inside —
the save pushes back by name and rebinds your local:

```python
results = []
with model.session(remote=True):
    for prompt in prompts:
        with model.trace(prompt):
            results.append(model.output.logits[0, -1].argmax().item())
    nnsight.save(results)
print(len(results))
```

Without the `nnsight.save(results)`, the appends land on the server's copy and your
`results` is still `[]` when the block exits.

## Gotchas

- **`datasets` must be installed server-side.** `nnsight.compare()` tables the
  server's packages against yours; a missing one surfaces as `ModuleNotFoundError`
  raised from the import line inside your block.
- **One session, not a loop of traces.** `for row in rows: with model.trace(...,
  remote=True)` is one queued job per row. Put `remote=True` on the session and
  leave the inner traces plain.
- **Batch inside the session.** Passing a list of strings to `model.trace` runs
  them as one forward pass; the loop above batches ten at a time. Batched input is
  **left**-padded, so `[:, -1]` is the true final token of every row even when the
  prompts differ in length (see [invoke-and-batching.md](../usage/invoke-and-batching.md)).
- **Sessions have a wall-clock ceiling.** Split a sweep that could exceed an hour
  into several sessions rather than one long one.
- **Read modules in forward order.** Reading a late module and then an early one in
  the same trace raises `OutOfOrderError` — see
  [errors/out-of-order-error.md](../errors/out-of-order-error.md).

## Related

- [remote-training.md](remote-training.md) — the same shape, with an optimizer in the loop.
- [docs/remote/remote-session.md](../remote/remote-session.md) — sessions as one request.
- [docs/gotchas/remote.md](../gotchas/remote.md) — what crosses the wire, and in which direction.
