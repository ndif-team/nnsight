---
title: Batching Internals
one_liner: How nnsight combines several tracer.invoke() inputs into one forward and scopes each block's reads/writes to its own batch rows.
tags: [internals, dev]
related: [docs/developing/interleaver-internals.md, docs/developing/architecture-overview.md]
sources: [src/nnsight/intervention/batching.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py]
---

# Batching Internals

## What this covers

A `with model.trace() as tracer:` block may contain several `with
tracer.invoke(x):` blocks. Their inputs are combined into a single batched forward,
and each block's interventions see only *its* rows of every activation. This doc
walks the `Batcher` (one per trace), the `batch_group` row ranges, and the
`narrow`/`widen`/`gather_skip`/`assemble_skip` operations the interleaver drives.

## Architecture

### Two halves: model-side prep and the per-trace Batcher

- **Model side** — `Envoy._batch_size(*inputs, **kwargs) -> int` (`envoy.py`)
  returns how many batch rows an invoke's input contributes (0 for an empty
  invoke). `Envoy._batch(invokes, fn) -> (args, kwargs)` (`envoy.py`) combines
  the collected invokes into one call. The base `Envoy` treats any input as a
  single row and passes a lone invoke straight through; batching two or more raises
  `NotImplementedError` unless a model overrides `_batch` (e.g.
  `TransformersModel`, which tokenizes and pads). One input invoke plus any number
  of empty invokes still works without `_batch`.
- **Per-trace** — the `Batcher` (`batching.py`) is constructed once per trace in
  `InterleavingTracer.execute` (`tracer.py`), as
  `self.envoy._batcher_class(self.envoy, self.kwargs)`, and set on the interleaver. It
  records each invoke's input, builds the combined forward input, and — during
  interleaving — narrows/widens activations per block.

### batch_group = [start, size]

`Batcher.add(*inputs, **kwargs)` (`batching.py`) records one invoke and returns
its `batch_group`:

- Calls `_batch_size`; a size of `0` — params only (e.g. `max_new_tokens=`) or an
  empty `invoke()` — returns `None` (a groupless worker that sees the whole batch)
  and folds the kwargs into `extra_kwargs`.
- Otherwise assigns `[self.total, size]`, appends the raw `(inputs, kwargs)` to
  `invokes`, and advances `total`.

The `batch_group` is stored on the invoke's `Mediator` — `InterleavingTracer.execute`
for direct input, `Invoker.execute` for `tracer.invoke` (`tracer.py`). The
interleaver reads it to scope that worker's reads/writes.

### The `batching` flag

`Batcher.batching` (`batching.py`) is `True` only once **two or more** non-empty
invokes have been added (`len(self.invokes) > 1`). With a single invoke, `narrow`
and `widen` are no-ops — a lone invoke *is* the whole batch, so it sees every row
untouched.

### narrow — scoping a read to a block's rows

`Batcher.narrow(value, group)` (`batching.py`) walks `value` and slices every
batched tensor in it down to the group's rows, delegating each tensor to
`_narrow_tensor` — the per-tensor method a layout subclass overrides:

```python
def _narrow_tensor(self, tensor, group):
    start, size = group
    if tensor.shape[0] == self.total:      # only actually-batched tensors
        return tensor.narrow(0, start, size)
    return tensor
```

A tensor is treated as batched only when its leading dim equals `total` (the
combined batch size), so a tensor whose dim 0 is sequence length or hidden size
passes through untouched. `narrow` returns `value` unchanged when not batching or
for a groupless (empty) invoke. It is called from `Mediator.handle` for every
`Event.VALUE` (`interleaver.py`).

The slice is stamped `_nnsight_batch = True`. A `.backward()` gradient hook uses
that marker to tell a batch slice from a user-made view: the slice isn't in the loss
graph (the model runs on the full batch), so the hook redirects to the
storage-owning base that is (`intervention/backward.py`).

### widen — splicing a block's edit back into the batch

`Batcher.widen(full, group, edited)` (`batching.py`) walks `full` and `edited` in
parallel — through lists, tuples, namedtuples and dicts, preserving the container
type — and hands each batched tensor pair to `_widen_tensor`, the per-tensor method
a layout subclass overrides:

```python
pre  = full.narrow(0, 0, start)
post = full.narrow(0, start + size, self.total - start - size)
return torch.cat([pre, edited, post], dim=0)
```

`cat` (rather than in-place assignment) keeps autograd correct for leaf/view
tensors and avoids aliasing when `edited` is itself a narrowed view of `full`.
Called from `Mediator.handle` for every `Event.SWAP` (`interleaver.py`).

**A replacement has to keep the group's row count.** `_widen_tensor` does not
check it: a `cat` of the wrong height succeeds — it just builds a batch that is
no longer the model's, and the mismatch surfaces, if at all, inside some later
module. On vLLM that can land as a device-side assert, which poisons the CUDA
context and takes the engine and every other request with it. Keeping the height
is the block author's obligation; every dim is the `cat`'s to check.

### gather_skip / assemble_skip — batched skips

A `.skip()` bypasses a module's body and substitutes a value for its output. In a
batched forward there is no body output to splice into (the body didn't run), so the
combined output is built from the invokes' replacements alone:

- `gather_skip(running, group, replacement)` (`batching.py`) — a lone invoke's
  replacement is the output outright; with two or more, it accumulates
  `(group, replacement)` pairs into a `SkipParts` (`batching.py`).
- `assemble_skip(running)` (`batching.py`) — after every worker has been
  served, concatenates the collected replacements in row order (`concat`,
  `batching.py`) into the full-batch output. It **requires every row to be
  covered** — a batched skip must skip the module in every invoke, or none, because
  a shared forward can't run for only the rows an invoke left unskipped:

  ```text
  A batched `.skip()` has to cover every row: skip the module in every invoke, or
  none — a shared forward can't run for only the rows an invoke left unskipped.
  ```

  `Interleaver.handle` calls `assemble_skip` once, after the mediator loop
  (`interleaver.py`).

### assemble — building the combined forward input

`Batcher.assemble(fn)` hands the row-contributing `invokes` to
`Envoy._batch(invokes, fn)`, which produces the `(args, kwargs)` for the run — for
`TransformersModel`, where `input_ids` are concatenated and re-padded and a combined
attention mask is built — then lays `extra_kwargs` on top. The row math above is
dim-0 only; `_batch` equalizes everything else (e.g. sequence length).

`Envoy.interleave` uses this uniformly for direct and traced calls: it always `add`s
the call's `(args, kwargs)` to a batcher and then `assemble`s. A direct (untraced)
call's input becomes a row; a trace's forward params (passed alongside the batcher)
fold in as `extra_kwargs` — so there is no separate merge step.

### Subclassing the Batcher

`Batcher` is meant to be subclassed for a model whose batch layout isn't a plain
dim-0 stack. Override the per-tensor `_narrow_tensor`/`_widen_tensor` (and
`gather_skip`/`assemble_skip` if skips need it) and name the subclass on the model
class as `_batcher_class` — the tracer builds the run's batcher from it, so that
line is the whole installation. The diffusion and vLLM runtimes do this:

- Diffusion's classifier-free-guidance doubles the batch (unconditional +
  conditional halves), so its batcher slices and splices both halves.
- vLLM's flat-token layout narrows on token ranges during the forward and prompt
  ranges after, and gathers/scatters tensor-parallel shards.

See [vllm-integration.md](vllm-integration.md) for the vLLM specifics, and
[extending.md](extending.md#4-custom-batcher-for-non-standard-batch-layouts) for a
worked custom batcher.

## Lifecycle / sequence

The batcher's part of the [trace lifecycle](./architecture-overview.md#lifecycle--sequence), per trace:

1. `InterleavingTracer.execute` creates `self.batcher = envoy._batcher_class(envoy, kwargs)`.
2. **Direct input** (`trace(x)`): one `Mediator`, given its group by
   `batcher.add(x)`. **Invoke mode** (`trace()`): the body is exec'd to collect
   `tracer.invoke(...)` sub-blocks; each `Invoker.execute` calls `batcher.add(...)`
   and appends a worker.
3. `Envoy.interleave(fn, batcher=...)` sets `interleaver.batcher` and calls
   `batcher.assemble(fn)` to build the combined `(args, kwargs)`.
4. During the forward, each `Event.VALUE` narrows to the worker's rows and each
   `Event.SWAP` widens back.
5. `cancel` clears the interleaver's batcher.

For example, invokes of `"a b c"` and `"x"` both see
`model.transformer.h[0].output.shape == [1, 3, 768]`: the batch dim is narrowed to
1, and `_batch` padded the shorter prompt to the combined length.

## Related

- [batching-and-invokers.md](../concepts/batching-and-invokers.md) — the mental-model version.
