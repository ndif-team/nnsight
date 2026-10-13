---
title: Sweeping a Dataset Remotely
one_liner: Load the data on NDIF instead of shipping it, loop over it inside one session, and bring back only the reduction.
tags: [pattern, remote, ndif, dataset, session]
related: [docs/remote/remote-session.md, docs/remote/remote-trace.md#gotchas, docs/patterns/remote-training.md]
sources: [src/nnsight/intervention/serialization.py, src/nnsight/modeling/mixins/remotable.py]
---

# Sweeping a Dataset Remotely

## What this is for

Running a model over hundreds or thousands of examples on NDIF — collecting
activations, scoring a probe, measuring an intervention across a benchmark. Two rules
carry almost all the benefit: **upload the minimum** (load the dataset on the server
rather than pickling it into the request) and **download the minimum** (reduce inside
the session and save the summary, not the activations).

Set `CONFIG.APP.DEBUG = True` to see what actually crossed the wire:

```
[remote] payload: 4,228 bytes (compressed)
[remote] result: 1,671 bytes downloaded
```

That is a 256-example sweep shaped like the one below. Shipping the dataset from the
client costs ~17 KB up instead; saving the raw residual stream instead of a scalar
costs ~2 MB down, and raw logits ~131 MB.

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
good = model.tokenizer.encode(" good")[0]
bad  = model.tokenizer.encode(" bad")[0]

with model.session(remote=True) as session:
    import torch
    from datasets import load_dataset

    rows = load_dataset("nyu-mll/glue", "sst2", split="train[:40]")
    tally = nnsight.save({"n": 0, "correct": 0})

    for start in range(0, len(rows), 10):
        batch = rows[start : start + 10]
        with model.trace(batch["sentence"]):
            logits = model.output.logits[:, -1]
            pred = (logits[:, good] > logits[:, bad]).long().cpu()   # zero-shot sentiment guess
        tally["n"] += len(pred)
        tally["correct"] += int((pred == torch.tensor(batch["label"])).sum())

    print(f"server saw {tally['n']} rows")

print(tally)
```

```
ℹ [235b1a59…] LOG          server saw 40 rows
{'n': 40, 'correct': 19}
```

The whole sweep is **one** request; `print` comes back as a `LOG` line while it runs.
Slice in the split string (`split="train[:40]"`) so the server materializes forty
rows, not the full 67k.

## Reduce before you save

The saved value is what gets serialized and pulled back. A per-layer residual stream
for 500 examples is gigabytes; the score you wanted from it is kilobytes.

```python
# wrong — ships every hidden state home to compute one number
with model.session(remote=True):
    states = nnsight.save([])
    for start in range(0, len(prompts), 32):
        with model.trace(prompts[start : start + 32]):
            states.append(model.transformer.h[6].output)
score = torch.cat([s[:, -1] @ probe for s in states]).mean()

# right — the reduction runs next to the weights; only floats come back
with model.session(remote=True):
    scores = nnsight.save([])
    for start in range(0, len(prompts), 32):
        with model.trace(prompts[start : start + 32]):
            hidden = model.transformer.h[6].output[:, -1]          # [B, hidden]
            scores.extend((hidden @ probe.to(hidden.device, hidden.dtype)).tolist())
```

For a reduction over tokens rather than the last position, average only the real
ones — see the masked mean in
[invoke-and-batching](../usage/invoke-and-batching.md#reducing-over-real-tokens-the-attention-mask).

Note `probe.to(hidden.device, hidden.dtype)`: `probe` was built on the client, so
it arrives on the CPU in float32 regardless of how the server holds the model. See
[remote-trace.md](../remote/remote-trace.md#gotchas).

A result container must be saved from inside the block — create it with
`nnsight.save(...)` as above. See
[remote-session](../remote/remote-session.md#saving-collections-built-across-traces).

## Gotchas

- **`datasets` must be installed server-side.** `nnsight.compare()` tables the
  server's packages against yours; a missing one surfaces as `ModuleNotFoundError`
  raised from the import line inside your block.
- **One session, not a loop of traces**, with `remote=True` on the session only —
  see [remote-session](../remote/remote-session.md).
- **Batch inside the session.** Passing a list of strings to `model.trace` runs
  them as one forward pass; the loops above batch ten and 32 at a time. Batched input is
  **left**-padded, so `[:, -1]` is the true final token of every row even when the
  prompts differ in length (see [invoke-and-batching.md](../usage/invoke-and-batching.md)).
- **Works under a standard API key.** A sweep only reads and edits activations, so
  the [trusted-key restrictions](../remote/remote-trace.md#standard-and-trusted-keys)
  on weights, gradients and training do not apply.
- **Sessions have a server-side time limit.** Split a long sweep into several
  sessions rather than one — see [remote-session.md](../remote/remote-session.md#gotchas).

## Related

- [remote-training.md](remote-training.md) — the same shape, with an optimizer in the loop.
- [docs/remote/remote-session.md](../remote/remote-session.md) — sessions as one request.
- [docs/remote/remote-trace.md](../remote/remote-trace.md#gotchas) — what crosses the wire, and in which direction.
