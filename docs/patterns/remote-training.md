---
title: Training a Parameter Remotely
one_liner: Run the whole optimizer loop inside one remote session — parameters created on the server's device, and only the trained weights come home.
tags: [pattern, remote, ndif, training, lora, session]
related: [docs/patterns/remote-dataset-sweep.md, docs/usage/backward-and-grad.md, docs/gotchas/remote.md]
sources: [src/nnsight/modeling/mixins/remotable.py, src/nnsight/intervention/serialization.py]
---

# Training a Parameter Remotely

## What this is for

Learning something small against a model you don't host: a LoRA adapter, a steering
vector, a probe trained through the frozen network. The gradients flow through
nnsight's interleaved backward exactly as they do locally
([backward-and-grad](../usage/backward-and-grad.md)) — what changes is that the
loop, the optimizer and the parameters all have to live on the server.

## When to use

- The model is remote and the thing you're fitting is far smaller than the model.
- You'd otherwise be submitting one job per training step.

## The shape

Everything goes inside one `session(remote=True)`. The optimizer never crosses the
wire; only the finished weights do.

```python
import torch, nnsight
from nnsight import TransformersModel

model = TransformersModel("meta-llama/Llama-3.1-70B")
module = model.model.layers[-1].mlp

with model.scan(" "):
    dim = module.output.shape[-1].save()


class LoRA(torch.nn.Module):
    def __init__(self, module, dim, rank, WA=None, WB=None):
        # Named, not bare: a shipped class is recompiled outside any class body,
        # so `super()` has no __class__ cell to read.
        super(LoRA, self).__init__()
        self.module = module
        # Built wherever this runs — on the server — so put the parameters on the
        # device holding the module, not the CPU torch would default to.
        device = module.device
        WA = torch.randn(dim, rank) if WA is None else WA
        WB = torch.zeros(rank, dim) if WB is None else WB
        self.WA = torch.nn.Parameter(WA.to(device), requires_grad=True)
        self.WB = torch.nn.Parameter(WB.to(device), requires_grad=True)

    def __call__(self):
        hidden = self.module.input
        # float32 adapter, bfloat16 model: cast into and back out of the adapter.
        delta = torch.matmul(torch.matmul(hidden.to(self.WA.dtype), self.WA), self.WB)
        self.module.output = delta.to(hidden.dtype) + self.module.output

    def parameters(self):
        return [self.WA, self.WB]


with model.session(remote=True) as session:
    from datasets import load_dataset

    rows = load_dataset("nyu-mll/glue", "sst2", split="train[:5000]")

    adapter = LoRA(module, dim, 4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3)

    for start in range(0, len(rows), 10):
        batch = rows[start : start + 10]
        labels = torch.tensor(batch["label"])

        with model.trace(batch["sentence"]):
            adapter()
            logits = model.lm_head.output
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1], labels.to(logits.device)
            )
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        print(loss.item(), adapter.WA.norm().item())

    # Plain tensors, not the adapter: it holds an Envoy pointing at a module that
    # only exists on the server.
    trained_WA = adapter.WA.detach().save()
    trained_WB = adapter.WB.detach().save()
```

`print` streams back as `LOG` lines, so the loss curve arrives while the job runs
rather than after it.

To use the result, rebuild the adapter from the weights that came home:

```python
with model.generate("I'm upset", remote=True):
    adapter = LoRA(module, dim, 4, WA=trained_WA, WB=trained_WB)
    adapter()
    out = model.lm_head.output.save()
```

## Why each of those details matters

| Detail | What goes wrong without it |
|---|---|
| Loop inside the session | One queued job per step instead of one for the run |
| `module.device` for the parameters | `Expected all tensors to be on the same device` — `torch.randn` gives CPU even server-side |
| `.to(self.WA.dtype)` around the matmuls | dtype mismatch against a bfloat16 model |
| `labels.to(logits.device)` | The batch came off the client, so the labels are on the CPU |
| `super(LoRA, self)` | `RuntimeError: super(): __class__ cell not found` |
| Save the weights, not the adapter | The adapter holds an `Envoy`; the object cannot be reconstructed client-side |
| Replacement assignment, not `output[:] =` | `one of the variables needed for gradient computation has been modified by an inplace operation` |

## Gotchas

- **The optimizer's `.grad` reads happen between traces, not inside them.** Step
  after the `with model.trace(...)` block exits, exactly as in ordinary PyTorch.
- **`nnsight.save` marks by identity.** `adapter.WA` is an attribute, not a name in
  the block's scope, so `.save()` on it inside `__init__` returns nothing. Bind the
  value to a local — `trained_WA = adapter.WA.detach().save()` — as above.
- **Server version matters.** NDIF used to wrap a whole request in one
  `torch.autocast` region with its weight cache enabled, which froze the
  half-precision copy of every parameter at the value it had before the first
  `optimizer.step()`. Gradients looked healthy and the weights moved, but the model
  never saw the updates and the loss sat flat. Fixed server-side; if you are
  training against an older deployment, call `torch.clear_autocast_cache()` after
  each step.

## Related

- [remote-dataset-sweep.md](remote-dataset-sweep.md) — the same session shape without an optimizer.
- [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md) — how `with tensor.backward():` interleaves.
- [docs/gotchas/remote.md](../gotchas/remote.md) — device, dtype, and what crosses the wire.
