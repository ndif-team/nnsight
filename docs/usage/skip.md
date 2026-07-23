---
title: Skip
one_liner: Bypass a module's (or operation's) forward with `.skip(replacement)`, substituting a value for its output.
tags: [usage, intervention, skip]
related: [docs/usage/access-and-modify.md, docs/usage/stop-and-early-exit.md, docs/usage/source.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py, src/nnsight/intervention/batching.py]
---

# Skip

## What this is for

`module.skip(replacement)` tells the interleaver: when the model is about to run
this module, **don't** — use `replacement` as its output instead. Useful for
ablating a submodule, swapping in a reconstructed activation (e.g. an SAE), or
routing around a layer.

A skip gate is installed on every module up front (via the source/skip
controller), so it works even when `replacement` is read from the module's own
`.input` first. Source operations expose the same `.skip(replacement)` — see
[source.md](source.md).

## When to use / when not to use

- Use to ablate or replace a single module's (or operation's) contribution.
- Use to inject an externally-computed value at a specific point.
- Use `tracer.stop()` to abort the whole forward — `skip` only bypasses one module
  ([stop-and-early-exit.md](stop-and-early-exit.md)).
- Use `model.edit(inplace=True)` to make a skip persistent across future traces.

## Canonical pattern

```python
import torch
from nnsight import NNsight
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = NNsight(MLP())
x = torch.randn(2, 8)

with model.trace(x):
    model.fc1.skip(torch.ones(2, 8))   # fc1 doesn't run; its output is ones
    out = model.output.save()
# out == fc2(relu(ones))
```

## Skip with the module's own input (pass-through)

Skipping a module with its own input turns it into a residual pass-through — the
skip gate is offered before the input read, so this works on the first trace:

```python
with model.trace(x):
    model.fc1.skip(model.fc1.input)    # fc1 passes its input straight through
    out = model.output.save()
# out == fc2(relu(x))
```

## Match the module's real output

The replacement must match what the module would normally return. Read the shape
first if unsure — a GPT-2 block's `.output` is a plain tensor `(batch, seq,
hidden)` in current transformers (not a tuple), so skip it with a tensor:

```python
from nnsight.modeling.transformers import TransformersModel
gpt2 = TransformersModel("openai-community/gpt2", dispatch=True)

with gpt2.trace("Hello world"):
    layer0 = gpt2.transformer.h[0].output  # a tensor
    gpt2.transformer.h[1].skip(layer0)     # skip layer 1: reuse layer 0's output
    out = gpt2.output.logits.save()
```

A module that returns a tuple (some attention submodules do) needs a tuple
replacement of the same shape.

## Persistent skip via `model.edit`

Store the skip as a default replayed on every future trace:

```python
with model.edit(inplace=True):
    model.fc1.skip(model.fc1.input)        # always pass fc1 through

with model.trace(x):
    out = model.output.save()              # fc1 skipped here too
```

See [edit.md](edit.md).

## Skip across invokes

In a batched trace, a `.skip()` must cover **every** row: skip the module in every
invoke, or none. A shared forward can't run for only the unskipped rows —
otherwise `ValueError: A batched .skip() has to cover every row`. Each invoke's
replacement fills its own rows and they're concatenated back into the batch. See
[invoke-and-batching.md](invoke-and-batching.md).

## Gotchas

- **Replacement shape/type must match the module's real output.** A mismatch causes
  downstream errors in the model's forward.
- **You can't read a skipped module's inner submodules or source ops** — they never
  execute, so requesting their `.output` is out of order (`OutOfOrderError`).
- **`skip` only works inside an active trace.**
- **Skips respect forward-pass order** like any access within one invoke.
- **A skip is one-shot per module call.** Across generation steps, each step needs
  its own skip — use `tracer.iter[...]` or a persistent edit for every-step
  behavior.

## Related

- [access-and-modify.md](access-and-modify.md)
- [stop-and-early-exit.md](stop-and-early-exit.md)
- [source.md](source.md) — operation-level `.skip()`.
- [edit.md](edit.md) — persistent skips.
