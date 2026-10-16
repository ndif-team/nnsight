---
title: Skip
one_liner: Bypass a module's (or operation's) forward with `.skip(replacement)`, substituting a value for its output.
tags: [usage, intervention, skip]
related: [docs/usage/access-and-modify.md, docs/usage/stop-and-early-exit.md, docs/usage/source.md, docs/usage/edit.md, docs/models/vllm.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py, src/nnsight/intervention/batching.py]
---

# Skip

`module.skip(replacement)` tells the interleaver: when the model is about to run
this module, **don't** — use `replacement` as its output instead. Use it to ablate
a submodule, swap in a reconstructed activation (e.g. an SAE), or route around a
layer. Source operations expose the same `.skip(replacement)`
([source](source.md)). To abort the whole forward, use
[`tracer.stop()`](stop-and-early-exit.md).

**On vLLM, treat `.skip()` as unsupported**: it must cover every row of the
engine step, other requests included, and a failure takes the engine down. See
[vllm](../models/vllm.md#skip-needs-the-whole-step).

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

`.input` is the module's *first* argument, so this is a pass-through only when the
module takes the activation first.

## Match the module's real output

The replacement stands in for what the module would have returned, so it has to
match that value in **structure, shape, dtype and device**. Read the shape first
if unsure: a GPT-2 block's `.output` is a plain tensor `(batch, seq, hidden)` in
current transformers (not a tuple), so skip it with a tensor:

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

A mismatch is caught by the model, not by nnsight, so it arrives as a raw torch
error from inside the next module's forward, naming neither `skip` nor the module:

| Replacement | What surfaces |
|---|---|
| `x.double()` | `RuntimeError: expected scalar type Double but found Float` |
| `x.half()` | `RuntimeError: expected scalar type Half but found Float` |
| `x.cpu()` | `RuntimeError: Expected all tensors to be on the same device, but got weight is on cuda:0, different from other tensors on cpu` |
| `(x,)` around a tensor output | `TypeError: layer_norm(): argument 'input' (position 1) must be Tensor, not tuple` |
| `x[:, :3, :]` | `RuntimeError: shape '[-1, 10, 768]' is invalid for input of size 2304` |

When a traced model errors inside a forward you did not intervene on, the skip
above it is the first place to look.

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

- **You can't read a skipped module's inner submodules or source ops** — they never
  execute, so requesting their `.output` raises `OutOfOrderError`. A skip requested
  after the model ran that module raises the same.
- **A skip is one-shot per module call.** Across generation steps, each step needs
  its own skip — use `tracer.iter[...]` or a persistent edit.
