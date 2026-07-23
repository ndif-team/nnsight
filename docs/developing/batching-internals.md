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

Everything lives in `src/nnsight/intervention/batching.py`, plus the two model-side
hooks `_batch_size` and `_batch` on `Envoy` (`src/nnsight/intervention/envoy.py`).

## Architecture

### Two halves: model-side prep and the per-trace Batcher

- **Model side** — `Envoy._batch_size(*inputs, **kwargs) -> int` (`envoy.py:588`)
  returns how many batch rows an invoke's input contributes (0 for an empty
  invoke). `Envoy._batch(invokes, fn) -> (args, kwargs)` (`envoy.py:597`) combines
  the collected invokes into one call. The base `Envoy` treats any input as a
  single row and passes a lone invoke straight through; batching two or more raises
  `NotImplementedError` unless a model overrides `_batch` (e.g.
  `TransformersModel`, which tokenizes and pads).
- **Per-trace** — `Batcher` (`batching.py:66`) is constructed once per trace in
  `InterleavingTracer.execute` (`tracer.py:245`) and set on the interleaver. It
  records each invoke's input, builds the combined forward input, and — during
  interleaving — narrows/widens activations per block.

### batch_group = [start, size]

`Batcher.add(*inputs, **kwargs)` (`batching.py:171`) records one invoke and returns
its `batch_group`:

- Calls `_batch_size`; a size of `0` (empty invoke) records `None` and returns
  `None` — it contributes no rows and sees the whole batch.
- Otherwise assigns `[self.total, size]`, appends the raw `(inputs, kwargs)` to
  `invokes`, and advances `total`.

The `batch_group` is stored on the invoke's `Mediator` (`tracer.py:259` for direct
input, `:364` for `tracer.invoke`). The interleaver reads it to scope that worker's
reads/writes.

### The `batching` flag

`Batcher.batching` (`batching.py:183`) is `True` only once **two or more** non-empty
invokes have been added (`len(self.invokes) > 1`). With a single invoke, `narrow`
and `widen` are no-ops — a lone invoke *is* the whole batch, so it sees every row
untouched. (This is the analogue of the OLD `needs_batching`; there is no separate
vLLM force flag in this class — vLLM subclasses the batcher.)

### narrow — scoping a read to a block's rows

`Batcher.narrow(value, group)` (`batching.py:85`) slices every batched tensor in
`value` down to the group's rows:

```python
def slice_(tensor):
    if tensor.shape[0] == self.total:      # only actually-batched tensors
        return tensor.narrow(0, start, size)
    return tensor
return apply(value, slice_, torch.Tensor)
```

A tensor is treated as batched only when its leading dim equals `total` (the
combined batch size), so a tensor whose dim 0 is sequence length or hidden size
passes through untouched. Returns `value` unchanged when not batching or for a
groupless (empty) invoke. This is called from `Mediator.handle` for every
`Event.VALUE` (`interleaver.py:408`).

### widen — splicing a block's edit back into the batch

`Batcher.widen(full, group, edited)` (`batching.py:103`) walks `full` and `edited`
in parallel and, for each batched tensor, writes `edited` into rows `[start, start +
size)`:

```python
pre  = full_value.narrow(0, 0, start)
post = full_value.narrow(0, start + size, self.total - start - size)
return torch.cat([pre, edited_value, post], dim=0)
```

`cat` (rather than in-place assignment) keeps autograd correct for leaf/view
tensors and avoids aliasing when `edited` is itself a narrowed view of `full`.
Called from `Mediator.handle` for every `Event.SWAP` (`interleaver.py:414`).

### gather_skip / assemble_skip — batched skips

A `.skip()` bypasses a module's body and substitutes a value for its output. In a
batched forward there is no body output to splice into (the body didn't run), so the
combined output is built from the invokes' replacements alone:

- `gather_skip(running, group, replacement)` (`batching.py:133`) — a lone invoke's
  replacement is the output outright; with two or more, it accumulates
  `(group, replacement)` pairs into a `SkipParts` (`batching.py:33`).
- `assemble_skip(running)` (`batching.py:148`) — after every worker has been
  served, concatenates the collected replacements in row order (`concat`,
  `batching.py:47`) into the full-batch output. It **requires every row to be
  covered** — a batched skip must skip the module in every invoke, or none, because
  a shared forward can't run for only the rows an invoke left unskipped:

  ```text
  A batched `.skip()` has to cover every row: skip the module in every invoke, or
  none — a shared forward can't run for only the rows an invoke left unskipped.
  ```

  `Interleaver.handle` calls `assemble_skip` once, after the mediator loop
  (`interleaver.py:591`-`592`).

### assemble — building the combined forward input

`Batcher.assemble(fn)` (`batching.py:188`) hands the collected `invokes` to
`Envoy._batch(invokes, fn)`, which produces the actual `(args, kwargs)` for the run.
For `TransformersModel` this is where `input_ids` are concatenated and re-padded and
a combined attention mask is built. The row math above is dim-0 only; `_batch`
equalizes everything else (e.g. sequence length) when it builds the combined input.

### Subclassing the Batcher

`Batcher` is meant to be subclassed for a model whose batch layout isn't a plain
dim-0 stack. Override `narrow`/`widen` (and `gather_skip`/`assemble_skip` if skips
need it). The diffusion and vLLM runtimes do this:

- Diffusion's classifier-free-guidance doubles the batch (unconditional +
  conditional halves), so its batcher slices and splices both halves.
- vLLM's flat-token layout narrows on token ranges during the forward and prompt
  ranges after, and gathers/scatters tensor-parallel shards.

See `docs/developing/vllm-integration.md` for the vLLM specifics.

## Key files / classes

- `src/nnsight/intervention/batching.py:66` — `Batcher`. Per-trace batching state.
- `:85` — `narrow`; `:103` — `widen`; `:133` — `gather_skip`; `:148` — `assemble_skip`.
- `:171` — `add`; `:183` — `batching`; `:188` — `assemble`.
- `:33` — `SkipParts`; `:47` — `concat`.
- `src/nnsight/intervention/envoy.py:588` — `_batch_size`; `:597` — `_batch`.
- `src/nnsight/intervention/tracer.py:223` — `InterleavingTracer.execute` (builds the Batcher, adds the direct-input worker).
- `src/nnsight/intervention/tracer.py:352` — `Invoker.execute` (adds an invoke worker).
- `src/nnsight/intervention/interleaver.py:375` — `Mediator.handle` (calls narrow/widen/gather_skip).

## Lifecycle / sequence

Per trace:

1. `InterleavingTracer.execute` creates `self.batcher = Batcher(envoy)` on the tracer.
2. **Direct input** (`trace(x)`): `_batch_size(x) > 0`, so one `Mediator` is made
   and `self.batcher.add(x)` gives it a batch group.
   **Invoke mode** (`trace()`): the body is exec'd to collect `tracer.invoke(...)`
   sub-blocks; each `Invoker.execute` calls `self.tracer.batcher.add(...)` and appends
   a worker.
3. `execute` calls `Envoy.interleave(fn, batcher=self.batcher, **params)`; interleave
   registers it on the interleaver (`interleaver.batcher`, for the run) and calls
   `batcher.assemble(fn)` to build the combined `(args, kwargs)` before running
   `fn(*args, **kwargs)`.
4. During the forward, each module hook's `handle` iterates workers; each
   `Event.VALUE` narrows to the worker's rows, each `Event.SWAP` widens back.
5. After the forward, `push_result` returns each worker's saved values; `cancel`
   clears the interleaver's batcher (the tracer's `self.batcher` goes with the trace).

Verified: two invokes of prompts `"a b c"` and `"x"` each see their own row —
`model.transformer.h[0].output.shape` is `[1, 3, 768]` in **both** blocks (the batch
dim is narrowed to 1; the shorter prompt is padded to the combined sequence length
by `_batch`).

## Extension points

- **A new runtime with custom batching.** Override `_batch_size`/`_batch` on the
  model class and return a `Batcher` subclass; set it on the interleaver in your
  tracer's `execute` (or reuse `InterleavingTracer.execute`, which constructs
  `Batcher(self.envoy)` — subclass `Batcher` and override the model's construction
  if you need a custom one). See `docs/developing/adding-a-new-runtime.md`.
- **Same input format, different tensor layout.** Subclass `Batcher` and override
  `narrow`/`widen` (and `gather_skip`/`assemble_skip`).
- **Single-input-only model.** Leave `_batch` unimplemented; a second input invoke
  raises a clear `NotImplementedError`, and one input invoke plus any number of
  empty invokes still works.

## Related

- `docs/developing/interleaver-internals.md` — how `narrow`/`widen` are driven from `Mediator.handle`.
- `docs/developing/vllm-integration.md` — the vLLM flat-token batcher.
- `docs/concepts/batching-and-invokers.md` — the mental-model version.
