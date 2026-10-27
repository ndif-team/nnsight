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

On any one rank, a sharded module's activation is only that rank's slice. nnsight
gathers the slices before your intervention sees the value and re-splits whatever
you leave behind, so **the trace you write is the trace you would write against
one GPU**. There is nothing to install, import, or enable.

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

Edits work the same way — you write the whole tensor and nnsight puts each rank's
piece back — but take a clone first (rule 3 below).

> **`tp_plan="auto"`** shards over whatever ranks the launcher provided; nnsight
> reads the degree off the world size and applies the same checks. Prefer
> `distributed_config=DistributedConfig(tp_size=N)`: it states the degree in the
> code and is the only form that can ask for expert parallelism. A *custom* plan
> (a dict of patterns) loads unchecked.

### When the load fails

All three arrive wrapped in a pipeline `ValueError: Could not load model ...`,
with the real message near the end:

- `KeyError: 'RANK'` — the script was started with `python`, not `torchrun`.
  The calling process has to *be* a rank, which is also why TP cannot run in a
  notebook kernel.
- `tp_size (2) * fsdp_size (1) is not equal to world_size (4)` —
  `--nproc_per_node` and `tp_size` have to be the same number.
- `` `tp_plan` and `device_map` are mutually exclusive `` — pick one.

## What is sharded and what is not

Most of what people read is **already whole**: a row-parallel layer all-reduces
its output, so a decoder layer, `self_attn`, `mlp`, and the final norm all arrive
complete.

| | Sharded? | Example |
|---|---|---|
| Column-parallel **output** | yes, gathered for you | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` |
| Row-parallel **input** | yes, gathered for you | `o_proj.input`, `down_proj.input` |
| Row-parallel output | no — all-reduced | `o_proj.output`, `down_proj.output` |
| Whole modules | no | `model.layers[i].output`, `mlp.output`, `norm.output` |
| The LM head | no — gathered by transformers | `lm_head.output` |
| Embeddings | yes, by vocabulary — reduced for you | `embed_tokens.output` |
| **Parameters** | **yes — not gathered** | `q_proj.weight`, `down_proj.weight` |
| `.source` inside a sharded module | yes — a `DTensor`, see [below](#reading-a-value-between-two-sharded-modules) | `q_proj.source.F_linear_0` |
| `.source` in a parent that calls one | no — gather it yourself | `mlp.source.self_gate_proj_0` |
| Anything between two sharded modules | no — gather it yourself | `mlp.act_fn.output`, `query_states_0` |

"No" means the value *reads* whole; writing into it in place is rule 3.

Calling a sharded module **ad hoc** — the logit lens, `model.lm_head(hidden)` —
returns the full-width result it would on one GPU. The gather only fires when an
intervention is parked on that location, and `tracer.cache()` gathers only the
modules it selects.

From transformers 5.21 the embedding is split by vocabulary on every sharded
model (before, only on tied-embedding checkpoints), and the vocabulary need not
divide by `tp_size`: the last ranks hold shorter slices of the embedding and of
`lm_head`'s output. Both read whole, at full width.

### Parameters are DTensors

`layer.weight` is a `DTensor`: this rank holds `1/tp_size` of it, but
**`.shape` reports the whole**. nnsight does not reassemble weights, since that
would allocate the whole tensor on every rank in exactly the situation where
memory was tight.

```python
w = model.model.layers[0].self_attn.q_proj.weight
w.shape                 # (3072, 3072) — the global shape
w.placements            # (Shard(dim=0),)
w.to_local().shape      # (1536, 3072) — what this rank holds, at tp=2
w.full_tensor()         # the whole thing; every rank must call it
```

**Reducing a sharded weight gives this rank's answer, silently.** `w.mean()`,
`w.norm()`, `w.abs().max()` return a `DTensor` with a `Partial` placement;
`float()` on it gives a per-rank partial that matches no real value. Call
`.full_tensor()` on the *reduction* (one small collective, not a copy of the
layer):

```python
float(w.norm())                  # rank 0: 81.80   rank 1: 68.97   (wrong)
float(w.norm().full_tensor())    # 106.996 on both
```

**A sharded weight cannot be edited in place through the `DTensor`**
(`NotImplementedError: Operator aten.fill_.Tensor does not have a sharding
strategy registered`). Write through `.to_local()`, each rank editing the slice it
holds; a rank-one update has to be expressed per rank the same way:

```python
with torch.no_grad():
    model.model.layers[0].mlp.gate_proj.weight.to_local()[:, :10] = 0
```

## Rules for intervention code under TP

**Every rank runs your block.** That keeps the collectives lined up, and puts four
obligations on the code:

1. **No rank-dependent control flow.** Nothing may branch on rank or take a
   different path on different ranks, or the ranks stop agreeing on when to
   gather and the run deadlocks — quietly, with no exception. `kill` (SIGTERM) of
   `torchrun` takes the ranks down with it; after `kill -9`, find and kill the
   rank processes too, or they keep holding their GPUs.

2. **Seed before you sample.** `torch.initial_seed()` differs per rank under
   `torchrun`, so unseeded sampling makes ranks generate *different tokens*, and
   the model's all-reduces then sum activations from different sequences — the
   output is wrong on every rank. Use greedy decoding or seed identically on every
   rank:

   ```python
   torch.manual_seed(0)                      # same on every rank
   with model.generate(prompt, max_new_tokens=20) as tracer:
       out = tracer.result.save()
   ```

   Many checkpoints ship `do_sample: true` in `generation_config.json`, so this
   bites without you asking for sampling. On a vocab-parallel embedding it can
   look healthy on rank 0 while ranks 1..N return sequences the model never ran
   on, so eyeballing rank 0 does not answer whether seeding matters.

3. **Editing takes a clone — on any value a sharded module produced.** Whether
   nnsight gathered it or transformers all-reduced it, the whole value is a
   collective's output, and an in-place write raises `Output 0 of ... is a view
   and is being modified inplace`. That includes values the table calls whole:
   `o_proj.output`, `down_proj.output`, `mlp.output`, the tensor inside
   `self_attn.output`, an MoE's `experts.output`, `lm_head.output`. Clone, edit,
   assign back:

   ```python
   with model.trace(prompt):
       edited = model.model.layers[5].mlp.gate_proj.output.clone()
       edited[..., :3000] = 0           # spans rank boundaries; no matter
       model.model.layers[5].mlp.gate_proj.output = edited
   ```

   Whole-value replacement (`... .output = torch.zeros_like(...)`) needs no
   clone. nnsight does not clone for you, since that would copy every gathered
   activation for the many traces that only read.

4. **Values inside a module's forward are not reassembled** — `.source`
   operations and modules between two sharded ones; see the next section.

**Every rank produces the same saved values.** Print or write results from one
rank, or you get N copies.

## Reading a value between two sharded modules

A module's `.input`/`.output` is gathered for you. A value *between* two sharded
modules is not: the shard is created by a column-parallel module and consumed by
the row-parallel one after it, so everything in between is one rank's slice.

```
gate_proj [colwise] ─▶  (1,11,64)  →  (1,11,32)   ┐
act_fn                  (1,11,64)  →  (1,11,32)   │  every value here
up_proj   [colwise]     (1,11,64)  →  (1,11,32)   │  is this rank's slice
down_proj [rowwise] ─▶  (1,11,16)  →  (1,11,16)   ┘  ← whole again
```

nnsight cannot gather these itself, because **nothing on the value says how it is
split**: transformers unwraps it to a plain tensor on the way out of the module
that made it, and the axis moves (attention's `view`/`transpose` puts it on the
head dimension). Nothing warns when you reach for `.source` on a sharded model.
You name the axis, with `gather` and its inverse `shard` (both in
`src/nnsight/modeling/tp/fragments.py`):

```python
from nnsight.modeling.tp import gather, shard

attn = model.model.layers[1].self_attn

with model.trace(prompt):
    q = attn.source.query_states_0.output       # (1, heads/tp_size, seq, head_dim)
    heads = gather(model, q, dim=1).clone()     # (1, heads, seq, head_dim); clone per rule 3
    heads[:, 3] = 0                             # ablate head 3, whoever holds it
    attn.source.query_states_0.output = shard(model, heads, dim=1)
    logits = model.lm_head.output.save()
```

A `.source` value from *inside* a sharded module is the subtle case: it is a
`DTensor`, so its `.shape` reports the whole while its data is this rank's slice.
`gather` reads the layout off such a value instead of the `dim` you name;
`.full_tensor()` also works. Test for it with `isinstance(value, DTensor)` or
`value.placements`.

Both are **collectives**, so rule 1 applies: call them unconditionally on every
rank. They are no-ops on an unsharded model, so the same block runs at any
degree — which also means **a one-GPU run validates nothing**: a wrong `dim`, or
a missing clone, passes at `tp_size=1` and raises or silently changes nothing on
a sharded run. Smoke-test a gather/shard block at `tp_size=2`. To find the shard
axis of a plain tensor, print its shape at tp=1 and tp=2; the axis that shrank by
`tp_size` is the one.

## A sharded model wrapped with `NNsight` directly

`TransformersModel` attaches the tensor-parallel rules for you. Plain `NNsight`
over an already-sharded module does not, and would hand over each device's piece.
Opt in:

```python
from nnsight import NNsight
from nnsight.intervention.interleaver import Interleaver
from nnsight.modeling.tp import TPFragments
from nnsight.modeling.tp.envoys import tp_envoys

model = NNsight(
    sharded_hf_model,
    interleaver=Interleaver(fragments=TPFragments()),
    envoys=tp_envoys(),          # ad-hoc calls like model.lm_head(hidden) return whole tensors
)
```

`TPFragments` stays inert on a model that is not sharded, so this is safe to pass
unconditionally.

## Requires transformers >= 5.19

Sharding on an earlier release is refused with `UnsupportedTransformersVersion`
the moment a sharded model is seen, because a backend layout nnsight cannot read
would otherwise hand intervention code one rank's slice as though it were whole.
A transformers old enough to lack the DTensor backend stops earlier, at
`ModuleNotFoundError: No module named 'transformers.distributed.tensor_parallel'`.

## A checkpoint that cannot be split at all

Not every model publishes a sharding plan (`base_model_tp_plan`): gpt2 does not,
and neither do several recent Qwen releases. **transformers does not refuse or
warn** — it shards nothing, so every rank loads a full copy. nnsight raises
`UnshardableCheckpoint` instead, before the weights are fetched (at `.dispatch()`
under the default `dispatch=False`):

```python
TransformersModel(
    "openai-community/gpt2",
    distributed_config=DistributedConfig(tp_size=2),
    dispatch=True,
)
# UnshardableCheckpoint: this checkpoint cannot be split tensor-parallel, so
# tp_size=2 would load a whole copy of it onto every rank rather than a shard.
```

The same error covers a degree that does not divide the head counts and the MLP
width (`tp_size=3` on a model that splits 8 ways), on which transformers' own
attention reshape fails; the message lists the degrees that would work. To check
in advance — config only, no weights, GPUs or `torchrun` — ask `max_tp_size`
(`src/nnsight/modeling/tp/plan.py`) for the largest degree; every workable degree
divides it, and `None` means TP is out:

```python
from transformers import AutoConfig
from nnsight.modeling.tp import max_tp_size

max_tp_size(AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B"))   # 8
max_tp_size(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"))         # 2  (2 kv heads)
max_tp_size(AutoConfig.from_pretrained("openai-community/gpt2"))     # None
```

## Expert parallelism

A mixture-of-experts checkpoint can be split by *expert* instead of along the
feature dimension; transformers then applies the model's `base_model_ep_plan`:

```python
model = TransformersModel(
    "openai/gpt-oss-20b",
    task="text-generation",
    distributed_config=DistributedConfig(tp_size=4, ep_size=4),
    dispatch=True,
)
```

Traces read the same as any other. The expert plan's wrapper
(`ep_dispatch_experts`) sends each token to the rank owning its expert and
combines the answers before returning, so the router's output is replicated and
`experts.output` is the full combined tensor on every rank. (Inside
`experts.source`, most steps hold per-rank token batches; only the final output
is whole.) The degree has to divide the **expert count**, checked before the
weights are fetched.

An EP load also wraps the model with FSDP2 even at `fsdp_size=1`, and the
pipeline prints a "The model 'FSDP...ForCausalLM' is not supported for
text-generation" warning on every rank. It is cosmetic.

**Plain tensor parallelism on an MoE is also fine.** Without an EP request the
experts shard along their feature dimensions like any MLP (`moe_tp_experts`), and
every read and edit behaves as above. Overriding the plan by hand to the legacy
`ep_router` + `moe_tp_experts` pair gives correct outputs but leaves
`experts.inputs` holding each rank's masked, local routing, with no warning.

## What is not supported

Four styles are refused: `megamoe_router`, `megamoe_experts`,
`moe_identity_expert`, and MLA's split kv projection (`mla_kv_a_proj`). A plan
containing one supports no degree, so the load is refused with
`UnshardableCheckpoint` (DeepSeek-V2-Lite: `mla_kv_a_proj` rules out tensor
parallelism, but expert parallelism is still open). A style added to a plan
*after* the load raises `UnsupportedParallelStyle`, naming the module and style.
They are refused because no model in the test set exercises them, not because
they are known to be ungatherable.

**Results differ slightly from one GPU**, because an all-reduce sums in a
different order than one matmul. In `bfloat16` expect relative differences
around 1e-3, growing with depth; in `float32`, around 1e-6. Greedy tokens match
in both, so a float32 mismatch well above 1e-6 is a real difference.

## Under the hood

Every `HuggingFaceModel` is built with a `TPFragments` that stays inert until it
instruments a module the model's own `tp_plan` names with `_device_mesh` set, so
eager loading and the meta-then-`dispatch()` path both pick it up. The
gather-serve-reshard bracketing is the interleaver's generic `Fragments` seam,
shared with vLLM; see [interleaver-internals.md](../developing/interleaver-internals.md#handle--one-visit)
and `src/nnsight/intervention/fragments.py`.

## Related

- [transformers-model.md](transformers-model.md) — the wrapper being sharded.
- [vllm.md](vllm.md) — the other way to shard across GPUs, with its own tradeoffs
  (throughput and continuous batching, one prompt per invoke).
- [../usage/cache.md](../usage/cache.md) — `tracer.cache()`, which gathers only
  what it selects.
