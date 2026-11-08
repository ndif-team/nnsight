---
title: Kernel Sites
one_liner: Read values a fused kernel never writes out, from the kernel that ran: attention probabilities, scores and logsumexp (`attn.kernel.probs(...)`), the recurrent states of a gated delta rule, Kimi delta attention, Mamba2 or Mamba layer (`mixer.kernel.states(...)`), and the per-expert rows of a grouped mixture-of-experts kernel (`experts.kernel.expert_output(...)`).
tags: [usage, kernel, attention, intervention]
related: [docs/usage/source.md, docs/patterns/attention-patterns.md, docs/patterns/per-head-attention.md, docs/usage/access-and-modify.md, docs/usage/iteration.md, docs/errors/module-never-fired.md]
sources: [src/nnsight/intervention/kernel.py, src/nnsight/intervention/attention.py, src/nnsight/intervention/recurrent.py, src/nnsight/intervention/experts.py, src/nnsight/intervention/source.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/transformers.py, tests/test_kernel.py]
---

# Kernel Sites

A fused attention kernel (`sdpa`, which runs FlashAttention, cuDNN, the
memory-efficient kernel or a CPU flash kernel) never writes the attention
probabilities: they live in registers and are consumed on the spot. `module.kernel`
reads them anyway, from the kernel that actually ran, with the model's output
unchanged.

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("Qwen/Qwen2.5-0.5B", dispatch=True)

with model.trace("The cat sat on the mat"):
    probs = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

print(probs.shape)        # [batch, keys]: one head, the last query row
print(probs.provenance)   # Provenance(kind='exact reconstruction', kernel='aten._scaled_dot_product_flash_attention_for_cpu', ...)
```

## Kernel probabilities and eager probabilities

Eager attention computes the softmax in Python, so `attn.output[1]` is available.
It is a different arithmetic: on Qwen3-8B in bf16 (56-token prompt, layers 0 to
9) the eager probabilities differ from the kernel's by a median of 0.6% and up to
20% at the top key, while the logits differ by a KL of 1.3e-3. Probabilities read
under eager belong to a different computation from the one the default model runs.

Every fused kernel keeps one small statistic for its backward pass: the per-row
logsumexp, `lse`. The probabilities the kernel used are `exp(S - lse)` with `S` the
fp32 scores, which is the formula the kernel's own backward recomputes them by.
`module.kernel` runs the model's SDPA call under a dispatch mode that records
which fused op the dispatcher chose and what it wrote, then reconstructs only the
slice requested, in fp32. The op runs exactly as the model called it.

Measured: logits bit-equal with and without a request (GPT-2 on CPU; Qwen2.5-0.5B
in bf16 on CUDA under flash, cuDNN and the math fallback). Probabilities against an
fp32 softmax over the kernel's own inputs: within 5e-7 on GPT-2 (CPU, fp32) and
2e-6 on Qwen2.5-0.5B (CUDA, bf16, layer 6). The reconstruction is exact up to fp32
rounding of `S` and `lse`, so rows sum to 1 within a few fp32 ulps of the row's
largest `|S|`: on a Qwen2.5-0.5B layer whose scores reach 1700, within 8e-4.

## Sites

`module.kernel.<site>(heads=, query=, key=)` on any module whose forward reaches
`torch.nn.functional.scaled_dot_product_attention`, directly or through one of
transformers' attention interfaces.

| Site | Value | Shape (unsliced) | Provenance |
|---|---|---|---|
| `lse` | the kernel's per-row logsumexp, fp32 | `[batch, heads, queries]` | `exact export` |
| `scores` | `q . k * scale` in fp32 with the mask added | `[batch, heads, queries, keys]` | `exact reconstruction` |
| `probs` | `exp(scores - lse)` | `[batch, heads, queries, keys]` | `exact reconstruction` |

`lse` takes `heads=` and `query=` only. Each index follows torch indexing: an int
(or a 0-d integer tensor) selects one position and drops the axis, negative counts
from the end; a `slice`, a list or an integer tensor of indices keeps the axis; a
boolean tensor selects where it is `True`. A bool, a float or a float tensor is
refused with `TypeError`. An out-of-range index raises `IndexError` on the request's
line. Asking for a whole `probs` matrix at long context materializes
`batch x heads x L x L` floats; a request over 1 GiB warns first.

```python
attn = model.model.layers[6].self_attn

with model.trace(prompt):
    last_row  = attn.kernel.probs(heads=4, query=-1).save()               # [batch, keys]
    two_heads = attn.kernel.probs(heads=[0, 4], query=slice(-5, None)).save()  # [batch, 2, 5, keys]
    lse       = attn.kernel.lse().save()                                  # [batch, heads, queries]
    scores    = attn.kernel.scores(heads=0, key=0).save()                 # [batch, queries]
```

Masks are what the kernel saw. Under a causal call or a boolean mask, `scores` is
`-inf` where attention is forbidden and `probs` exactly `0`; under a float mask,
`scores` carries the mask's own value (transformers uses the dtype minimum) and
`probs` is exactly `0`. A row with every key masked, a pad query in a left-padded
batch, is all `0`; every other row sums to 1.

## Provenance

The returned tensor carries `.provenance`, a `Provenance(kind, kernel, note)`:

- `kind="exact export"`: a tensor the kernel wrote (`lse`).
- `kind="exact reconstruction"`: computed from the kernel's inputs and its exported
  statistic by an identity (`scores`, `probs`).
- `kind="replay"`: the statistic was recomputed in fp32 beside the kernel. This
  happens when the math fallback ran (no fused op fired; it computes in fp32
  throughout), and when a kernel that writes `lse` only on request (the efficient
  and cuDNN kernels) is launched a second time for it and that launch does not
  reproduce the model's output bit for bit. The model's output is the first launch
  either way.
- `kind="derived"`: a value the kernel's algorithm never forms, computed from an
  exported statistic by the kernel's own formula truncated at the requested point
  (the state at a token inside a chunk, [below](#recurrent-states-of-a-chunked-kernel)).
  The note carries a check of that formula against the kernel's next export.

`kernel` names the op that ran (`aten._scaled_dot_product_flash_attention`,
`aten._scaled_dot_product_cudnn_attention`, `math backend (fp32)`, ...). `note`
carries the formula and, when `dropout_p > 0`, the reminder that dropout is applied
after these probabilities.

`.provenance` is an attribute of the returned tensor; a view or a copy of it
(`probs[0]`, `probs.clone()`) has none. A saved tensor keeps it through `torch.save`
and `torch.load`.

## Grouped-query attention, masks, decoding

- **GQA** is read off the shapes: query head `h` uses key head `h // (Hq // Hkv)`,
  the layout both `enable_gqa` and transformers' `repeat_kv` use. `heads=` always
  indexes query heads.
- **Causal** calls are top-left aligned (query `i` sees keys `0..i`), as
  `is_causal` is in torch. transformers passes an explicit mask whenever the
  queries and keys differ in length.
- **Decoding**: during `generate` the kernel sees one query row against the cache,
  so `probs()` at step `n` is `[batch, heads, 1, prompt + n]` with a dynamic cache.
  Collect per step with `tracer.iter` ([iteration](iteration.md)); a request pinned
  to step `n` is served at step `n` whether or not earlier steps requested anything.
- **Gradients**: the served tensors are detached. The kernel's `lse` is taken below
  autograd, so a gradient through `exp(S - lse)` would treat it as a constant and
  come out wrong; none is offered.

## Ordering

A site is served when the kernel fires, so request it before reading anything that
runs later in the forward. A request after `attn.output` raises
`OutOfOrderError` ([out-of-order-error](../errors/out-of-order-error.md)); outside a
trace it raises `ValueError("Cannot access ... outside of interleaving")`.

Accessing `envoy.kernel` instruments the module's forward, the same instrumentation
`.source` installs, and stays on the module. Inside a trace that is in time unless
the module is already running: a read inside the module earlier in the same block
(`attn.c_attn.output`, then `attn.kernel.probs(...)`) resumes the block while the
uninstrumented forward is mid-call, and the request raises `OutOfOrderError: '...kernel'
was requested while '...' is already running without an instrumented forward`. Access
`attn.kernel` once before the trace and the same block works.

A request on a module whose forward reaches no registered kernel is never served:
the MLP, a whole decoder block (ask its attention submodule), a whole MoE block
(ask its `experts` submodule), a model loaded with `attn_implementation="eager"` or
`flash_attention_2`, experts under `experts_implementation="eager"` (or
`deepgemm`, `sonicmoe`, an FP8 or `kernels` implementation), and a vLLM model,
whose attention runs through its own paged-attention op. The trace then ends with
`OutOfOrderError: '<path>.kernel.i0' was requested but the model already ran past it`
([module-never-fired](../errors/module-never-fired.md)).

## Recurrent states of a chunked kernel

A gated delta rule layer (Qwen3-Next, Qwen3.5), a Kimi delta attention layer
(Kimi Linear, GLM-5-Next) or a Mamba2 mixer runs its recurrence in chunks: inside a
chunk the kernel works with matrices over the chunk's tokens, and the recurrent
state exists only at chunk boundaries. No implementation computes the state at a
token inside a chunk. The kernel does hold the boundary states and the intra-chunk
quantities it forms on the way (for the gated delta rule its pairwise decays, keys
and delta-corrected values; for Kimi delta attention the same with the decay a
vector over the key channels; for Mamba2 its segment decays, discretized inputs and
`B`), and the state after any token is the boundary before its chunk advanced to
that token by the kernel's own formula, the one it applies over the whole chunk to
reach the next boundary, stopped early. A Mamba mixer (Mamba, FalconMamba, Jamba,
Zamba) runs its selective scan token by token, so there every token's state is the
kernel's own.

```python
mixer = model.model.layers[0].linear_attn          # Qwen3-Next; `backbone.layers[0].mixer` on Mamba2 and Mamba, `model.layers[0].self_attn` on Kimi Linear

with model.trace(prompt):
    boundaries = mixer.kernel.chunk_states().save()            # [batch, chunks + 1, heads, K, V]
    s_100 = mixer.kernel.states(tokens=100, heads=3).save()    # [batch, K, V]
    tail = mixer.kernel.states(tokens=slice(-5, None)).save()  # [batch, 5, heads, K, V]
```

The state is `[K, V]` per head on a gated delta rule or Kimi delta attention
layer, `[head_dim, state_size]` per head on Mamba2 and `[state_size]` per channel on
Mamba, whose `heads=` axis is the channel axis. Both sites return fp32.

| Site | Value | Provenance |
|---|---|---|
| `chunk_states(chunks=, heads=)` | the state entering each chunk; index `0` the initial state, index `C` the final state when the kernel produced one (`output_final_state`, which transformers sets when a cache is in use) | `exact export`; the note names the dtype the kernel kept it in (fp32 for transformers' reference implementations and `fla`'s final state; the input dtype for `fla`'s chunk buffer, so those entries carry one bf16 rounding) |
| `states(tokens=, heads=)` | the state after each requested token | `exact export` where the token ends a chunk whose next boundary was exported; otherwise `derived`: the preceding boundary advanced in fp32 by the kernel's intra-chunk formula over its recorded intermediates, one contraction per touched chunk |

`chunks=-1` is the final state only when the kernel produced one; the note says
`the kernel returned no final state` otherwise. For the state after the last token
use `states(tokens=-1)`, which is served either way.

A `derived` note carries the check of that formula: taken to each touched chunk's
end it must land on the kernel's next boundary, and the note says how far off it
is, over the whole batch. On transformers' fp32 reference implementations in the
test models that is 0 for the gated delta rule and Kimi delta attention and about
1e-6 relative for Mamba2 (the contraction sums in a different order). Against an
fp64 per-token recurrence over the same inputs, the derived states of all three
and Mamba's exported states are within 2e-6 absolute. `fla`'s Triton kernels keep their
intra-chunk quantities, so a state inside one of their chunks is the boundary
stepped token by token by the per-token recurrence; in bf16 the note then reads
about 5e-3, which is the bf16 rounding of the boundary it is compared against, and
the states themselves are within 2e-5 of transformers' fp32 recurrence, the same
distance as the exported boundaries. During `generate`, each decode step runs the per-token
kernel, so `states(tokens=-1)` at a step is the kernel's own state after that
token: `torch_recurrent_gated_delta_rule` or `fla`'s fused recurrent kernel on a
gated delta rule layer, `recurrent_kimi_delta_attention` on Kimi delta attention,
`mamba2_selective_state_update` on Mamba2, `mamba_selective_state_update` on Mamba.

In a batch of invokes of different lengths, `tokens=` and `chunks=` index
positions of the left-padded batch: `tokens=-1` is every invoke's last token, and
the state at a pad position is `0`.

Which kernels: transformers' reference implementations (`torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`, `chunk_kimi_delta_attention`,
`recurrent_kimi_delta_attention`, `mamba2_chunk_scan`, `mamba_selective_scan` and
the two `selective_state_update` functions), which every model in these families
reaches when `fla` or `mamba_ssm` is not installed, and `fla`'s
`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` when it is. The reference implementations are
instrumented for the fire (the same instrumentation `.source` uses) and their
boundary-state bindings recorded; `fla`'s chunk-state buffer is recorded on its way
from `chunk_gated_delta_rule_fwd_h` to the output kernel. The model's output is
unchanged either way. Registration happens when a `TransformersModel` is built,
from the kernels its modeling file reaches, and is process-wide: a plain
`NNsight(module)` of the same family is served once such a model has been built
in the process, and never before.

A `.source` drill into the kernel function itself
(`mixer.source.torch_chunk_gated_delta_rule_0.source.implementation_0.source`)
takes precedence on that fire: its operations stay addressable, and a `.kernel`
request on the module in the same trace is not served. Use one or the other per
trace.

## Per-expert rows of a grouped mixture-of-experts kernel

A mixture-of-experts layer routes each token to `top_k` experts and sums their
weighted outputs. transformers' `grouped_mm` experts implementation (the default
when torch's grouped matmul can run; otherwise transformers falls back to `eager`)
runs all the (token, expert) pairs of a forward as two grouped matrix
multiplications over rows sorted by expert, up and down; the `batched_mm`
implementation `generate` switches to for decoding off the CPU runs them in token
order with one weight matrix gathered per pair. What each expert received,
computed and returned for a token exists only as rows inside those forwards, and
the module returns the sum. The sites hand back the forward's own rows in token
order, in the dtype it computed them in.

```python
experts = model.model.layers[3].mlp.experts          # Qwen2-MoE, Qwen3-MoE, Mixtral, OLMoE, ...

with model.trace(prompt):
    routing = experts.inputs[0][1].save()                                # top_k_index, [tokens, top_k]
    out = experts.kernel.expert_output().save()                          # [batch, seq, top_k, hidden]
    seven = experts.kernel.expert_output(expert=7).save()                # [batch, seq, hidden]; 0 where not routed to 7
    act = experts.kernel.expert_activation(tokens=-1, slot=0).save()     # [batch, intermediate]
```

| Site | Value | Provenance |
|---|---|---|
| `expert_input(tokens=, slot=)` | the row the pair's expert received, `[.., top_k, hidden]` | `exact export` |
| `expert_activation(tokens=, slot=)` | the activation between the two projections, `act(gate) * up` (or `act(up)` for experts without a gate), `[.., top_k, intermediate]` | `exact export` |
| `expert_output(tokens=, slot=)` | the pair's down projection before its routing weight, `[.., top_k, hidden]`; the module's output is the sum over slots of these rows times `top_k_weights`, cast to the input dtype | `exact export` |

Slot `k` is the token's k-th routed expert, `top_k_index[token, k]`. `expert=e`
(one int) replaces `slot=` and gives the expert's rows per token, before the
routing weight, `0` for a token not routed to it.

The experts module computes over the flattened batch x sequence axis, as its
`.input` and `.output` show. The sites fold that axis by the trace's batch rows
and hand each invoke its own, so `tokens=` indexes sequence positions and a decode
step under `generate` serves `[batch, 1, ...]`. Three consequences:

- In a batch of invokes of different lengths `tokens=` indexes the left-padded
  positions, and a pad position holds the pad token's rows. `experts.inputs` read
  in an invoke is the whole batch's, as for any module input.
- `generate` with `num_return_sequences=n` or `num_beams=n` expands each prompt to
  `n` rows, which the fold concatenates along the sequence axis: `[batch, n x seq, ...]`
  at prefill and `[batch, n, ...]` at a decode step, each invoke still reading only
  its own rows.
- A token count the batch rows do not divide raises `ValueError` at the request; a
  plain `Envoy` of the experts module counts its whole input as one row, so its
  shapes are `[1, tokens, ...]`.

Measured on a tiny Qwen2-MoE (CPU, fp32): the rows equal a one-pair-at-a-time
fp64 computation within fp32 rounding (6e-9 on activations of size 3e-2), the
weighted sum over slots equals the module's output exactly, and the logits are
bit-equal with and without a request; the same holds for six MoE families
(Mixtral, OLMoE, Qwen3-MoE, DeepSeek-V3, Nemotron-H without a gate, GPT-OSS with
transposed weights and biases) and for Qwen1.5-MoE-A2.7B in bf16 on a GPU. Served
for the `grouped_mm` and `batched_mm` implementations. Under
`experts_implementation="eager"` the per-expert loop is Python and `.source`
reaches it (`experts.source.experts_forward_1.source`); `.source` also reaches the
fused forwards' locals, in expert order
(`experts.source.experts_forward_1.source.proj_out_2.output`), which the sites put
back in token order.

## Limits

- Reads only: a site cannot be assigned. To modify attention scores, add a bias to
  the mask the module passes to the kernel, or edit `q`/`k` before it
  ([access-and-modify](access-and-modify.md)).
- Under tensor parallelism each rank holds `heads / tp` query heads, and `heads=`
  indexes the rank's own heads; the result on rank 0 is its share, with no gather
  ([tensor-parallel](../models/tensor-parallel.md#what-is-not-supported)). The
  recurrent sites are whole on every rank (transformers gathers the projections
  before the kernel).
- `mamba_ssm`'s fused `mamba_chunk_scan_combined`, `selective_scan_fn` and
  `selective_state_update` have no entry, and neither do `fla`'s `chunk_kda` and
  `fused_recurrent_kda`; the reference implementations are covered. Mamba's
  `mambapy` and associative-scan paths keep the final state only and are served as
  `Unavailable`. LFM2's short convolution has no recurrent state to read; RWKV's
  linear attention has no entry. `fla`'s packed layout (`cu_seqlens`), its
  `use_gate_in_kernel` form, and a `fla` backend that bypasses
  `chunk_gated_delta_rule_fwd_h` (FlashQLA on SM90 and later) are served as
  `Unavailable`: the request raises a `ValueError` naming the reason. A model
  whose hub wrapper is replaced by the `kernels` library has no recurrent entry.
  Paged attention has no sites. vLLM's fused MoE has no entry: its `FusedMoE` layer
  runs the experts inside the custom op `torch.ops.vllm.moe_forward`, whose Triton
  experts keep their per-pair rows in workspace buffers local to the op, so
  neither the instrumentation nor a dispatch mode sees them (and with `taps` the
  engine also replays recorded graphs).
- Experts under transformers' tensor or expert parallelism are served as
  `Unavailable`: each rank holds a shard of the intermediate axis and a partial
  down projection, or other ranks' pairs as sentinels, so there are no whole rows
  to serve. A transformers newer than 5.19 that adds a post-expert norm to these
  forwards is served as `Unavailable` too.
- A submodule named `kernel` is reached as `E_kernel`
  ([access-and-modify](access-and-modify.md#overloaded-names)).
- Gemma 2's attention softcap does not reach SDPA (transformers drops it on this
  path); the served values match what the kernel computed.
