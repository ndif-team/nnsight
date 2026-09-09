---
title: Tensor Parallelism
one_liner: Trace a model sharded across GPUs with transformers tensor parallelism — sharded activations are gathered so a trace reads exactly as it would on one GPU.
tags: [models, transformers, tensor-parallel, multi-gpu, distributed]
related: [docs/models/transformers-model.md, docs/models/vllm.md, docs/models/index.md, docs/usage/generate.md, docs/usage/cache.md]
sources: [src/nnsight/modeling/tp/fragments.py, src/nnsight/modeling/tp/envoys.py, src/nnsight/modeling/tp/plan.py, src/nnsight/modeling/huggingface.py, tests/tp/worker.py, tests/tp/test_cpu_gloo.py]
---

# Tensor Parallelism

## What this is for

A model too big for one GPU can be **split across several** with transformers'
native tensor parallelism: each rank holds a slice of every attention and MLP
projection. This is different from `device_map="auto"`, which puts whole *layers*
on different GPUs and runs them one after another — tensor parallelism splits
*within* each layer and runs the ranks together.

The catch for interpretability is that a sharded module's activation, on any one
rank, is only that rank's slice of the real tensor. nnsight gathers those slices
before your intervention sees the value and re-splits whatever you leave behind,
so **the trace you write is the trace you would write against one GPU**.

There is nothing to install, import, or enable.

## The canonical pattern

Tensor parallelism needs one process per GPU, so the script is launched with
`torchrun` (or `python -m torch.distributed.run`) and **every rank runs the whole
script, including your intervention code**.

```python
# tp_trace.py  —  torchrun --nproc_per_node=4 tp_trace.py
import torch
from transformers.distributed import DistributedConfig
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dispatch=True,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4),
)

with model.trace("The Eiffel Tower is in the city of"):
    # gate_proj is column-parallel: each rank computes 2048 of these 8192
    # features. Read it and you get all 8192.
    gate = model.model.layers[5].mlp.gate_proj.output.save()
    logits = model.lm_head.output.save()

print(gate.shape)  # (1, 11, 8192) on every rank, not (1, 11, 2048)
```

Edits work the same way: you write the whole tensor and nnsight puts each rank's
piece back. The one thing to remember is the clone — a gathered value is the
output of a collective, and torch refuses an in-place write into one (rule 3
below):

```python
with model.trace(prompt):
    # Spans rank boundaries; you never think about that.
    edited = model.model.layers[5].mlp.gate_proj.output.clone()
    edited[..., :3000] = 0
    model.model.layers[5].mlp.gate_proj.output = edited
    logits = model.lm_head.output.save()
```

> **`tp_plan="auto"` is checked too.** A bare `tp_plan="auto"` names no degree
> of its own — transformers shards over whatever ranks the launcher provided —
> so nnsight reads the degree off the world size and runs the same check below.
> gpt2 under `torchrun --nproc_per_node=2` is refused with
> `UnshardableCheckpoint` rather than silently loading its full 0.5 GB on both
> ranks. A *custom* plan (a dict of patterns) overrides the checkpoint's
> published plan, so the published plan's limit says nothing about it and it
> loads unchecked. Prefer `distributed_config=DistributedConfig(tp_size=N)`
> anyway: it states the degree in the code, and it is the only form that can ask
> for expert parallelism.

### When the load fails

All three of these arrive wrapped in a pipeline `ValueError: Could not load model
...`, with the real message near the end of it:

- `KeyError: 'RANK'` — the script was started with `python`, not `torchrun`.
  Tensor parallelism needs the calling process to *be* a rank, which is also why
  it cannot run in a notebook kernel.
- `tp_size (2) * fsdp_size (1) is not equal to world_size (4)` —
  `--nproc_per_node` and `tp_size` have to be the same number.
- `` `tp_plan` and `device_map` are mutually exclusive `` — pick one. They are
  two different ways of spreading a model, and `device_map` is the reflex to
  suppress here.

## What is sharded and what is not

Most of what people read is **already whole** and costs nothing: a row-parallel
layer all-reduces its output, so a decoder layer, `self_attn`, `mlp`, and the
final norm all arrive complete. What is really a slice, and what nnsight does
about it:

| | Sharded? | Example |
|---|---|---|
| Column-parallel **output** | yes, gathered for you | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` |
| Row-parallel **input** | yes, gathered for you | `o_proj.input`, `down_proj.input` |
| Row-parallel output | no — all-reduced | `o_proj.output`, `down_proj.output` |
| Whole modules | no | `model.layers[i].output`, `mlp.output`, `norm.output` |
| The LM head | no — gathered by transformers | `lm_head.output` |
| Embeddings | whole, unless the plan shards them — see below | `embed_tokens.output` |
| **Parameters** | **yes — not gathered** | `q_proj.weight`, `down_proj.weight` |
| `.source` inside a sharded module | yes — a `DTensor` whose `.shape` is the whole and whose data is this rank's, see rule 4 | `q_proj.source.F_linear_0` |
| `.source` in a parent that calls one | no — gather it yourself | `mlp.source.self_gate_proj_0` |
| Anything between two sharded modules | no — gather it yourself | `mlp.act_fn.output`, `query_states_0` |

Calling a sharded module **ad hoc** — the logit lens, `model.lm_head(hidden)` —
is corrected the same way its activations are, so it returns the full-width
result it would on one GPU.

The gather only fires when an intervention is actually parked on that location,
so reading a handful of locations does not pay for the hundreds you ignored. A
`tracer.cache()` gathers only the modules it selects.

**A vocab-parallel embedding's own output cannot be read.** Only some plans shard
the embedding — those that do name it `embedding_rowwise`, which in practice means
the tied-embedding checkpoints (Llama-3.2-1B and -3B, and most small models).
Where it is sharded, the value's layout is single-use: the buffer saying which
vocabulary rows this rank owns is spent by the first reassembly, so parking on
`embed_tokens.output` raises a bare `AssertionError` from torch's embedding op.
The same takes out a `tracer.cache()` called with *no* arguments, which selects
every module and so reaches the embedding.

Which case you are in is in the config:

```python
"embed_tokens" in (model.config.base_model_tp_plan or {})
# True  — Llama-3.2-1B, Llama-3.2-3B     embed_tokens.output raises
# False — Llama-3.3-70B, Qwen3-8B        embed_tokens.output reads whole
```

`model.model.layers[0].input` works either way: it is the same tensor one module
later, and it arrives whole at any degree.

```python
with model.trace(prompt) as tracer:
    embeddings = model.model.layers[0].input.save()   # whole, on any plan
    cache = tracer.cache(modules=[...]).save()        # name them, don't select all
```

Parameters are the exception, and the one place a trace does not read as it would
on one GPU. `layer.weight` is a `DTensor`: this rank holds `1/tp_size` of it, but
**`.shape` reports the whole**, so the shape alone will not tell you it is split.

```python
w = model.model.layers[0].self_attn.q_proj.weight
w.shape                 # (3072, 3072) — the global shape
w.placements            # (Shard(dim=0),)
w.to_local().shape      # (1536, 3072) — what this rank actually holds, at tp=2
w.full_tensor()         # the real thing; every rank must call it, and it
                        # allocates the whole tensor on each of them
```

Weights are what tensor parallelism exists to split, so nnsight does not quietly
reassemble one — that would allocate the whole tensor on every rank, in the
situation where memory was tight enough to reach for TP. Most torch operations
handle a `DTensor` for you; `.to_local()` and `.full_tensor()` are there when you
need to be explicit.

**Reducing a sharded weight gives you this rank's answer, and no error says so.**
`w.mean()`, `w.norm()`, `w.abs().max()` all return a `DTensor` with a `Partial`
placement — the reduction over this rank's slice, still waiting to be combined.
Read a scalar out of it with `float()` and you get that partial number, which
differs between ranks and matches none of them. On Llama-3.2-3B at tp=2:

```python
w = model.model.layers[0].self_attn.q_proj.weight
float(w.mean())                  # rank 0:  2.256e-05   rank 1: -1.171e-05
float(w.mean().full_tensor())    # 5.421e-06 on both — the real mean
float(w.norm())                  # rank 0: 81.80        rank 1: 68.97
float(w.norm().full_tensor())    # 106.996
```

Call `.full_tensor()` on the *reduction*, not on the weight: the result is a
scalar, so it costs one small collective rather than a copy of the layer. Any
weight-norm sweep or layer-magnitude plot needs this, or it silently plots
per-rank noise.

**A sharded weight cannot be edited in place through the `DTensor`.**
`weight[:, :10] = 0` raises `NotImplementedError: Operator aten.fill_.Tensor does
not have a sharding strategy registered`. Write through `.to_local()`, which is
the right thing anyway — each rank edits the rows or columns it holds:

```python
with torch.no_grad():
    model.model.layers[0].mlp.gate_proj.weight.to_local()[:, :10] = 0
```

Out-of-place arithmetic on the whole (`weight.mul_(0.5)`) works unchanged. A
rank-one update written against a whole weight matrix has to be expressed
per-rank the same way.

## Rules for intervention code under TP

**Every rank runs your block.** That is what keeps the collectives lined up, and
it puts four obligations on the code:

1. **No rank-dependent control flow.** Nothing may branch on rank, and nothing
   may take a different path on different ranks — the ranks would stop agreeing
   on when to gather, and the run deadlocks. It deadlocks quietly: no exception,
   and no watchdog inside several minutes. Killing `torchrun` does not take the
   rank processes with it either — they stay alive holding their share of every
   card, so find their pids and kill those, or the next attempt meets a GPU that
   is busy with the last one.

2. **Seed before you sample.** This one is a correctness bug, not an
   inconsistency. `torch.initial_seed()` differs per rank under `torchrun` — the
   ranks are seeded randomly, not from the rank — so unseeded sampling diverges
   immediately. If it does, the ranks generate *different tokens*,
   and then the model's own all-reduces sum activations computed from different
   sequences — the output is wrong on every rank, not merely different. Use
   greedy decoding, or seed identically on every rank:

   ```python
   torch.manual_seed(0)                      # same on every rank
   with model.generate(prompt, max_new_tokens=20) as tracer:
       out = tracer.result.save()
   ```

   Many checkpoints ship `do_sample: true` in `generation_config.json`, so this
   bites without you asking for sampling.

3. **Editing a gathered value takes a clone.** A sharded value is made whole by
   a collective, and torch will not let you write into the output of one in
   place — a `value[...] = x` raises `Output 0 of ... is a view and is being
   modified inplace`. Clone, edit, assign back:

   ```python
   with model.trace(prompt):
       edited = model.model.layers[0].mlp.gate_proj.output.clone()
       edited[..., :64] = 0
       model.model.layers[0].mlp.gate_proj.output = edited
   ```

   Whole-value replacement (`... .output = torch.zeros_like(...)`) needs no
   clone. nnsight does not clone for you: on a large model that is a copy the
   size of the activation, on every gather, for the many traces that only read.

4. **Nothing inside a module's forward is reassembled for you.** A module's own
   `.input`/`.output` is; a `.source` operation is not, and neither is a plain
   module sitting between a column-parallel output and the row-parallel input
   that consumes it (`mlp.act_fn`). Nothing on those values records which axis
   holds the shard once it has left the module that produced it, and the axis
   moves — attention's `view`/`transpose` puts it on the head dimension. So the
   trace names the axis and reassembles it, with `gather`/`shard`; see "Reading a
   value between two sharded modules" below. Nothing warns when you reach for
   `.source` on a sharded model — this page is the notice.

   A `.source` value taken from *inside* a sharded module is the subtle case: it
   is a `DTensor`, so its `.shape` reports the whole even though its data is this
   rank's slice, and reading a number out of it is wrong without a word of
   warning. `gather` handles it — the value carries its own layout, so `gather`
   uses that rather than the `dim` you name — as does `.full_tensor()`.

**Every rank produces the same saved values**, since they are computed from
gathered tensors. Print or write results from one rank, or you get N copies.

## Reading a value between two sharded modules

A module's `.input`/`.output` is gathered for you. A value *between* two sharded
modules is not, and this is the one place a trace does not read as it would on
one GPU.

The shard is created by a column-parallel module and consumed by the
row-parallel one after it, so everything in between is one rank's slice:

```
gate_proj [colwise] ─▶  (1,11,64)  →  (1,11,32)   ┐
act_fn                  (1,11,64)  →  (1,11,32)   │  every value here
up_proj   [colwise]     (1,11,64)  →  (1,11,32)   │  is this rank's slice
down_proj [rowwise] ─▶  (1,11,16)  →  (1,11,16)   ┘  ← whole again
```

nnsight cannot gather these for you, because **nothing on the value says how it
is split.** transformers tracks the layout only while a value is inside the
module that made it; on the way out it unwraps to an ordinary tensor, and a
32-wide slice of a 64-wide activation is then indistinguishable from a genuine
32-wide one. Worse, the axis moves: attention's `view`/`transpose` puts the shard
on the *head* dimension, so there is not even a fixed axis to assume.

You know what the forward did, so you say which axis, with
[`gather`][nnsight.modeling.tp.fragments.gather] and its inverse
[`shard`][nnsight.modeling.tp.fragments.shard]:

```python
from nnsight.modeling.tp import gather, shard

attn = model.model.layers[1].self_attn

with model.trace(prompt):
    q = attn.source.query_states_0.output   # (1, heads/tp_size, seq, head_dim)
    heads = gather(model, q, dim=1).save()  # (1, heads, seq, head_dim) — every head
```

To edit one, put this rank's piece back before the forward carries on — and
clone first, per rule 3:

```python
with model.trace(prompt):
    q = attn.source.query_states_0.output
    heads = gather(model, q, dim=1).clone()
    heads[:, 3] = 0                                  # ablate head 3, whoever holds it
    attn.source.query_states_0.output = shard(model, heads, dim=1)
    logits = model.lm_head.output.save()
```

Measured on Llama at tp=2 against the same block on one GPU: the gathered
`query_states` is `(1,4,11,4)` either way with the same sum, and ablating head 3
gives logits summing to `-19.721992` against `-19.722000`.

Both are **collectives**, so rule 1 applies with full force: every rank must
reach them. Call them unconditionally, never inside a branch that could go
differently on different ranks. They are no-ops on an unsharded model, so the
same block runs at any degree — which is also how you keep one script working on
one GPU and on eight.

If you are not sure which axis holds the shard, print the shape at tp=1 and at
tp=2 and compare: the axis that shrank by `tp_size` is the one. That comparison
answers for a plain tensor, which is what a value between two modules is. It
says nothing about a `.source` value inside a sharded module, whose `.shape` is
the whole at every degree — for those, `isinstance(value, DTensor)` or
`value.placements` is the test, and `gather` reads the layout off the value
itself.

## A sharded model wrapped with `NNsight` directly

`TransformersModel` attaches the tensor-parallel rules for you. A tree built with
plain `NNsight` over an already-sharded module does not get them by default — it
has no way to know the module came from `transformers` — so every sharded value
would be handed over as this device's piece. Opt in:

```python
from nnsight import NNsight
from nnsight.intervention.interleaver import Interleaver
from nnsight.modeling.tp import TPFragments
from nnsight.modeling.tp.envoys import tp_envoys

model = NNsight(
    sharded_hf_model,
    interleaver=Interleaver(fragments=TPFragments()),
    envoys=tp_envoys(),          # ad-hoc calls on sharded modules
)
```

`TPFragments` works the rules out for itself as the tree is built, and stays
inert on a model that is not sharded — so this is safe to pass unconditionally.
`tp_envoys()` is what makes `model.lm_head(hidden)` return a whole tensor rather
than this rank's slice.

## Requires transformers >= 5.16

5.16 rebuilt tensor parallelism on DTensor. The sharding plan moved from a stamp
on each sharded module to a single `_tp_plan` on the model, and the style's
collectives moved from forward hooks into a wrapper around `module.forward`.
nnsight reads the layout the way that backend describes it, so on an older
transformers it would find nothing sharded — and that does not fail, it hands
intervention code one rank's slice as though it were the whole tensor. nnsight
refuses to load rather than let that through. The refusal is not always worded
as a version: on transformers 5.15 the load stops at
`ModuleNotFoundError: No module named 'transformers.distributed.tensor_parallel'`,
which is the 5.16 backend not being there. Check `transformers.__version__`
before reading further into it.

## A checkpoint that cannot be split at all

Not every model publishes a sharding plan. gpt2 is the obvious one: its config
has no `base_model_tp_plan`, so there is nothing telling transformers which
weights to cut or along which dimension.

**transformers does not refuse this, and does not warn.** Given a plan of `None`
it shards *nothing* — `verify_tp_plan` returns early and no forward is wrapped —
so every rank loads a complete copy of the weights. The model then answers
correctly off one rank while the other cards hold redundant copies. Nothing
fails; you simply paid `tp_size` GPUs for one GPU's worth of work.

nnsight raises `UnshardableCheckpoint` instead, before the weights are fetched:

```python
TransformersModel(
    "openai-community/gpt2",
    distributed_config=DistributedConfig(tp_size=2),
    dispatch=True,
)
# UnshardableCheckpoint: this checkpoint cannot be split tensor-parallel, so
# tp_size=2 would load a whole copy of it onto every rank rather than a shard.
```

The check sits on the loading path, so with the default `dispatch=False` the
constructor returns and the refusal arrives at `.dispatch()`, which is the first
point at which there is anything to refuse.

The same error covers a degree that does not divide evenly — `tp_size=3` on a
model that splits 8 ways — because the all-gather assumes every rank holds an
equal piece, so an uneven degree returns the wrong shape rather than running
slower. The message lists the degrees that would work.

To find out in advance, ask
[`max_tp_size`][nnsight.modeling.tp.plan.max_tp_size] for the largest degree a
config supports; every workable degree is one of its divisors, and `None` means
the model has to be spread some other way (one GPU, or `device_map` over
several). It reads the config alone — no weights, no GPUs, no `torchrun` — so it
is the cheapest thing to run before you allocate anything:

```python
from transformers import AutoConfig
from nnsight.modeling.tp import max_tp_size

max_tp_size(AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B"))   # 8
max_tp_size(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"))         # 2  (2 kv heads)
max_tp_size(AutoConfig.from_pretrained("openai-community/gpt2"))     # None
```

A missing plan is not only an old-model problem — several recent Qwen releases
publish no `base_model_tp_plan` either, and `None` is `None` whatever the reason.

## Expert parallelism

A mixture-of-experts checkpoint can be split by *expert* instead of along the
feature dimension. Ask for it the way transformers does — it applies the model's
`base_model_ep_plan` in place of its tensor-parallel one:

```python
model = TransformersModel(
    "openai/gpt-oss-20b",
    task="text-generation",
    distributed_config=DistributedConfig(tp_size=4, enable_expert_parallel=True),
    dispatch=True,
)
```

Traces read the same as any other. The three styles an expert plan uses need
less gathering than the name suggests: `ep_router` leaves the router replicated
and masks non-local experts *after* the handoff, `grouped_gemm` shards expert
parameters and installs no transform at all, and `moe_tp_experts` produces this
rank's term of a sum, which is reduced for you like any row-parallel output.

The degree has to divide the **expert count** rather than the head counts, and
nnsight checks that before the weights are fetched — `tp_size=3` on a 4-expert
model raises `UnshardableCheckpoint` rather than loading and failing.

**Plain tensor parallelism on an MoE checkpoint is also fine.** `moe_tp_experts`,
which Mixtral, DeepSeek-V3, Qwen3-MoE and around twenty-five other shipped
configs use, all-reduces inside its own forward.

## What is not supported

Four styles are still refused: `megamoe_router`, `megamoe_experts`,
`moe_identity_expert`, and MLA's split kv projection (`mla_kv_a_proj`). A plan
containing one supports no degree at all, so a load asking for tensor parallelism
is turned away by the same `UnshardableCheckpoint` as a checkpoint with no plan —
deliberately, rather than handing you a fragment of a tensor and letting you draw
conclusions from it. (DeepSeek-V2-Lite is the one to know: `mla_kv_a_proj` puts
tensor parallelism out of reach, while expert parallelism is still open to it.)
`UnsupportedParallelStyle`, which names the module and the style, is what a style
added to a plan *after* the load produces.

They are refused because no model in the test set exercises them, not because
they are known to be ungatherable. `ep_router` and `grouped_gemm` sat in this
list until a model using them was actually run end to end, at which point both
turned out to need no gather at all. Reading a style's transforms is not the same
as having run one, which is the mistake the table exists to prevent.

Results differ slightly from a single-GPU run, because an all-reduce sums in a
different order than one big matmul — and how much depends entirely on the dtype.
Measured on Llama-3.2-3B at tp=4 against one GPU, as the largest elementwise
difference over the tensor's own scale:

| | layer 0 `gate_proj` | layer 13 | layer 27 | logits |
|---|---|---|---|---|
| `bfloat16` | 3.4e-3 | 9.0e-3 | 7.5e-3 | 8.2e-3 |
| `float32` | 2.5e-7 | 7.3e-7 | 9.7e-7 | 9.1e-7 |

So bfloat16 drifts around 1e-3 and grows with depth, while float32 stays near the
floor of the dtype. Greedy token choices are identical in both, and the test
suite asserts that. If you are using agreement with a single-GPU run as a sanity
check, expect the bfloat16 numbers — a float32 mismatch above 1e-6 is a real
difference, not arithmetic order.

## Under the hood

`TPFragments` ([`nnsight.modeling.tp`][nnsight.modeling.tp]) says which locations
hold one rank's slice and how to reassemble them; the
[`Interleaver`][nnsight.intervention.interleaver.Interleaver] does the bracketing —
gather the value, serve the parked workers the whole tensor, re-split what they
leave — once per visit, and only when something is actually waiting to read it.

Every `HuggingFaceModel` is built with an ordinary interleaver carrying one of
these. It stays inert (`enabled = False`, one attribute check per location) until
it instruments a module the model's own `tp_plan` names — with `_device_mesh`
set, which is what says the model was really split — and that is where it records
the rules. That covers eager loading and the
meta-then-`dispatch()` path without either needing to know about it.

The same seam serves vLLM's tensor parallelism, which shards differently and
gathers with different collectives — see
[`nnsight.intervention.fragments`][nnsight.intervention.fragments].

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being sharded.
- [vllm.md](vllm.md) — the other way to shard across GPUs, with its own tradeoffs
  (throughput and continuous batching, one prompt per invoke).
- [../usage/cache.md](../usage/cache.md) — `tracer.cache()`, which gathers only
  what it selects.
