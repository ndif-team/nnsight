---
title: "One seam for distributed values"
one_liner: vLLM and tensor parallelism both make a fragment whole at a location; they now do it through one collaborator on the interleaver instead of two unrelated subclasses. Why it sits there, and what the port cost.
tags: [internals, dev]
related: [docs/developing/vllm-integration.md, docs/developing/interleaver-internals.md, docs/developing/batching-internals.md, docs/models/tensor-parallel.md]
sources: [src/nnsight/intervention/fragments.py, src/nnsight/modeling/tp/fragments.py, src/nnsight/modeling/vllm/fragments.py, src/nnsight/modeling/vllm/batching.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py]
---

# One seam for distributed values

**Status: implemented.** Written down after an audit of the tensor-parallel
work, then built. Both runtimes now go through
[`Fragments`][nnsight.intervention.fragments.Fragments]; `TPInterleaver` and the
gather half of `VLLMBatcher` are gone.

Verified on 8xA100: the vLLM tensor-parallel suite passed 17/17 before the port
and 17/17 after, the MoE suite alongside it, the full vLLM suite 112 passed with
one pre-existing failure (`test_temperature_changes_samples`, confirmed identical
with the change reverted), and the transformers TP suite 129.

## The observation

nnsight had two implementations of the same idea, sharing no code:

| | vLLM | tensor parallelism |
|---|---|---|
| where | `VLLMBatcher` | `TPInterleaver` |
| extends | `Batcher` | `Interleaver` |
| "is this a fragment?" | a module registered by `watch` | a `_hf_tp_plan` stamp, per `SHARDED_SIDES` |
| "make it whole" | all-gather, or all-reduce for a deferred MoE partial | `all_gather` |
| "put it back" | re-shard, scaling a write-back by `tp_size * ep_size` | `split` |

Both answer the same three questions about a value at a location. Neither
`Batcher` (which exists to narrow a batch's rows) nor `Interleaver` (which exists
to schedule greenlets) is *about* that question, so each subclass carries the
answer as a passenger.

The concrete cost is not duplication — the two gathers are genuinely different
collectives. It is that **`VLLMBatcher` already knows how to gather a
deferred-reduce MoE partial** (`vllm/batching.py`: all-reduce, then divide by
`tp_size * ep_size` so the block's own reduce sums it exactly once) and that
knowledge was structurally unreachable from `TPInterleaver`. Meanwhile the
transformers TP path refused every expert-parallel style it could not gather. We
were paying for the same hole twice, in two places that could not lend each other
anything.

## Why they ended up in different base classes

Worth recording, because the reason is not the one that was originally written
down.

`Batcher.narrow`/`widen` are called **once per mediator** — inside the loop that
serves each parked worker in turn. A batcher that gathered would fire one
collective per *reader* of a location, so ranks would run different numbers of
collectives depending on how many workers happened to be parked. That is a
deadlock. vLLM pays for it with `self.gathered` memoization plus `watch`/`release`
brackets that its **model runner** installs (`GPUModelRunner.py`), which it can do
because it owns the forward.

`Interleaver.handle` is already the once-per-visit bracket, so `TPInterleaver`
needed no memo and no external bracket. That is the real argument, and it is what
decided the shape: **it belongs on the interleaver, called once per visit, with
the runtime supplying only the policy.**

(The rationale this replaces claimed `_batcher_class` would resolve to the
*client's* class across a remote trace. It does not: `Envoy.__getstate__` pickles
the envoy by value, but `_batcher_class` is a class attribute and cloudpickle
serializes an importable class by reference, so it resolves to the server's value
exactly like `interleaver`. Verified directly. A `TPBatcher` was reachable; it
would just have been called at the wrong granularity.)

## The shape

A collaborator object, not a subclass. As built it also carries `enabled`,
`begin` and `read` — see below for why:

```python
# src/nnsight/intervention/fragments.py

class Fragments:
    """Whether a value at a location is a piece of a larger one, and how to
    make it whole for a reader and put it back for the model.

    The default is the identity: no location is a fragment, nothing is gathered.
    A distributed runtime supplies a subclass; everything else is unaffected.
    """

    def fragmented(self, location: str) -> bool:
        return False

    def whole(self, location: str, value: Any) -> Any:
        return value

    def fragment(self, location: str, whole: Any) -> Any:
        return whole
```

`Interleaver` grows one attribute and one branch:

```python
class Interleaver:
    fragments: Fragments | None = None

    def handle(self, provider, value):
        gathering = (
            self.fragments is not None
            and self.fragments.fragmented(provider)
            and self.observed(provider)
        )
        if gathering:
            value = self.fragments.whole(provider, value)

        ...the existing mediator loop, skip assembly and cache offering,
           entirely unchanged...

        return self.fragments.fragment(provider, value) if gathering else value
```

`observed` moved up from `TPInterleaver` unchanged — it was already generic ("is
any worker or cache waiting on *this* visit"), and it is what keeps an untouched
location free.

Then:

- **`TPFragments`** is the `SHARDED_SIDES` table plus `_gather`/`_reshard`.
  `TPInterleaver` is gone, and with it the install-one-on-every-`HuggingFaceModel`
  dance — a HuggingFace model now gets an ordinary `Interleaver` carrying
  `TPFragments`. The interleaver no longer knows what a shard is.
- **`VLLMFragments`** is the old `_whole`/`_shard`, keyed by location instead of
  by a `watch`ed module: `instrument` records `location -> (module, side)` as the
  tree is built. `self.gathered`, `watch`, `release`, `narrow`/`widen` and **both
  hook installers in `GPUModelRunner`** are gone. `VLLMBatcher` is down to
  `batching = True` and a constructor, from about 210 lines.

## Three things the sketch above missed

- **`enabled`.** Without it, an unsharded HuggingFace model would call `observed`
  — which loops every mediator — at every location, where before it cost one
  attribute check. A `Fragments` starts disabled and a subclass flips it on
  finding something actually split.
- **`read(location)`.** The `.source` warning under tensor parallelism has to fire
  for values that are *not* fragments (that is the whole point: they can't be
  gathered, so the reader is told). It needed a hook of its own, called for any
  observed location.
- **`begin()`.** `TPInterleaver.__enter__` cleared its warned-set each run so a
  long-lived model actor warns every user, not just the first. With the subclass
  gone that needed an explicit per-run hook.

## What it cost

- One real bug, caught by the vLLM suite: rewriting `VLLMBatcher` dropped its
  `__init__`, and the base `Batcher` requires an `envoy` the model runner has no
  tree to supply yet. The engine failed to start.
- The module-swap worry turned out to be a non-issue, and for a reason worth
  recording: the previous design registered forward hooks on those same module
  objects at load, so a swapped module would have gone just as dead. Holding a
  reference is no weaker than what it replaced.
- External subclasses of `VLLMBatcher` break. Unlikely to exist, not impossible.

## What it bought

- One place to add a gather rule, so the MoE work lands once instead of twice.
- A third runtime implements one small object instead of choosing a base class
  and inheriting a scheduling concern it does not care about.
- `Interleaver` no longer has a subclass whose only job is a policy decision, and
  `Batcher` is back to being only about rows.
- Two fewer forward hooks per parallel layer in vLLM, and the load-order
  dependency they relied on — watch registered *before* the tree, release *after*
  — is gone with them. That was correct and entirely invisible.
