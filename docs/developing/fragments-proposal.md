---
title: "Proposal: one seam for distributed values"
one_liner: vLLM and tensor parallelism both make a fragment whole at a location, in two unrelated subclasses of two unrelated base classes. What a shared collaborator would look like, what it costs, and when to do it.
tags: [internals, dev, proposal]
related: [docs/developing/vllm-integration.md, docs/developing/interleaver-internals.md, docs/developing/batching-internals.md, docs/models/tensor-parallel.md]
sources: [src/nnsight/modeling/tp/interleaver.py, src/nnsight/modeling/vllm/batching.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py]
---

# Proposal: one seam for distributed values

**Status: not implemented.** Written down after an audit of the tensor-parallel
work. Do it when either the MoE gather or a third distributed runtime lands —
doing it now, purely for symmetry, buys tidiness and risks a runtime whose tests
need GPUs.

## The observation

nnsight has two implementations of the same idea, and they share no code:

| | vLLM | tensor parallelism |
|---|---|---|
| where | [`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher] | [`TPInterleaver`][nnsight.modeling.tp.interleaver.TPInterleaver] |
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
knowledge is structurally unreachable from `TPInterleaver`. Meanwhile the
transformers TP path refuses every expert-parallel style it cannot gather. We are
paying for the same hole twice, in two places that cannot lend each other
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
needs no memo and no external bracket. That is the real argument, and it points
at the right shape for a shared seam: **it belongs on the interleaver, called once
per visit, with the runtime supplying only the policy.**

(The rationale this replaces claimed `_batcher_class` would resolve to the
*client's* class across a remote trace. It does not: `Envoy.__getstate__` pickles
the envoy by value, but `_batcher_class` is a class attribute and cloudpickle
serializes an importable class by reference, so it resolves to the server's value
exactly like `interleaver`. Verified directly. A `TPBatcher` was reachable; it
would just have been called at the wrong granularity.)

## The shape

A collaborator object, not a subclass:

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

`observed` moves up from `TPInterleaver` unchanged — it is already generic ("is
any worker or cache waiting on *this* visit"), and it is what keeps an untouched
location free.

Then:

- **`TPFragments`** is today's `SHARDED_SIDES` table plus `_gather`/`_reshard`,
  around 110 lines. `TPInterleaver` disappears entirely, and with it the
  install-one-on-every-`HuggingFaceModel`-in-case-it-is-sharded dance in
  `huggingface.py`. The rules are recorded by whatever walks the tree; the
  interleaver stops knowing what a shard is.
- **`VLLMFragments`** is today's `_whole`/`_shard`, keyed by location instead of
  by a `watch`ed module. vLLM's runner already walks the modules at load, so it
  registers `location -> module` there once instead of installing two hooks per
  parallel layer. `self.gathered`, `watch`, `release` and the four hook installers
  all delete. `VLLMBatcher` shrinks to the `batching = True` override — roughly 15
  lines where there are now 210.

## What it costs

- A refactor of the vLLM path, whose tests need GPUs. This is the whole risk.
- `VLLMBatcher._whole` currently needs the live module for `isinstance` checks, so
  the location-to-module map must be built at load. That is a real behavioral
  change if vLLM ever swaps modules mid-run — **unverified**; check before
  starting.
- External subclasses of `VLLMBatcher` break. Unlikely to exist, not impossible.

## What it buys

- One place to add a gather rule, so the MoE work lands once instead of twice.
- A third runtime implements one small object instead of choosing a base class
  and inheriting a scheduling concern it does not care about.
- `Interleaver` stops having a subclass whose only job is a policy decision, and
  `Batcher` goes back to being only about rows.

## When

When the MoE gather is picked up, or when a third distributed runtime appears.
Not before: the present arrangement is correct, tested on hardware at degree 2 and
4, and the duplication is costing exactly one thing (the MoE knowledge) that
nothing is currently asking for.
