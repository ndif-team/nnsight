---
title: Kernel Sites
one_liner: Read intermediate values from fused kernels without switching to eager execution: attention probabilities, scores, and logsumexp (`attn.kernel.probs(...)`), recurrent states from gated delta rule, Kimi delta attention, Mamba2, and Mamba layers (`mixer.kernel.states(...)`), and per-expert rows from grouped mixture-of-experts forwards (`experts.kernel.expert_output(...)`).
tags: [usage, kernel, attention, intervention]
related: [docs/usage/source.md, docs/patterns/attention-patterns.md, docs/patterns/per-head-attention.md, docs/usage/access-and-modify.md, docs/usage/iteration.md, docs/errors/module-never-fired.md]
sources: [src/nnsight/intervention/kernel.py, src/nnsight/intervention/sites.py, src/nnsight/intervention/attention.py, src/nnsight/modeling/kernels/transformers.py, src/nnsight/modeling/kernels/eager.py, src/nnsight/modeling/kernels/flash.py, src/nnsight/modeling/kernels/recurrent.py, src/nnsight/modeling/kernels/experts.py, src/nnsight/intervention/source.py, src/nnsight/intervention/interleaver.py, src/nnsight/modeling/transformers.py, tests/test_kernel.py]
---

# Kernel Sites

Fused attention kernels (`sdpa`, which dispatches to FlashAttention, cuDNN,
memory-efficient attention, or CPU flash attention) compute attention
probabilities in registers and never write the full probability matrix to
memory. `module.kernel` lets you read those values directly from the fused
kernel that ran, while leaving the model's forward pass and output unchanged.

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("Qwen/Qwen2.5-0.5B", dispatch=True)

with model.trace("The cat sat on the mat"):
    probs = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

print(probs.shape)        # [batch, keys]: one head, the last query row
print(probs.provenance)   # Provenance(kind='exact reconstruction', kernel='aten._scaled_dot_product_flash_attention_for_cpu', ...)
```

## Kernel probabilities and eager probabilities

Under `attn_implementation="eager"`, the attention module computes the softmax
in Python and returns the probability matrix at `attn.output[1]`. However,
switching from a fused kernel to eager execution changes the floating-point
arithmetic. On Qwen3-8B in bf16 (56-token prompt, layers 0 to 9), eager
attention probabilities differ from the fused kernel's probabilities by a median
of 0.6% and by up to 20% at the top key, while the model's output logits shift
by a KL divergence of 1.3e-3. `module.kernel.probs(...)` is also served under
`attn_implementation="eager"`, where it returns the weights tensor produced by
the eager call, so the same site request reads the probabilities the loaded
model computed under either implementation.

Every fused SDPA backend computes one small summary statistic per query row for
its backward pass: the logsumexp (`lse`). From the fp32 scores `S` and `lse`,
the attention probabilities used by the kernel are given by `exp(S - lse)`,
which is the same formula the kernel uses to recompute probabilities during its
backward pass. When a trace requests an attention kernel site, `module.kernel`
runs the model's SDPA call inside a dispatch mode that records which fused ATen
operator ran and captures its `lse` output, then reconstructs only the requested
slice in fp32.

Across tested configurations (GPT-2 on CPU; Qwen2.5-0.5B in bf16 on CUDA under
FlashAttention, cuDNN, and the math fallback), model logits are bit-identical
with and without a kernel site request. Compared against a reference fp32
softmax over the kernel's inputs, the reconstructed probabilities match within
5e-7 on GPT-2 (CPU, fp32) and within 2e-6 on Qwen2.5-0.5B (CUDA, bf16, layer 6).
Because the reconstruction is exact up to fp32 rounding of `S` and `lse`, each
unmasked row sums to 1 within a few fp32 ulps of the row's largest `|S|` (within
8e-4 on a Qwen2.5-0.5B layer whose scores reach 1700).

## Sites

Call `module.kernel.<site>(heads=, query=, key=)` on any attention module whose
forward calls `torch.nn.functional.scaled_dot_product_attention` (directly or
through one of transformers' attention interfaces), calls transformers'
`eager_attention_forward` under `attn_implementation="eager"` ([below](#under-eager-attention)),
or calls the FlashAttention 2 functions under
`attn_implementation="flash_attention_2"` ([below](#flashattention-2)).

| Site | Value | Shape (unsliced) | Provenance |
|---|---|---|---|
| `lse` | the kernel's per-row logsumexp in fp32 | `[batch, heads, queries]` | `exact export` |
| `scores` | `q . k * scale` in fp32 with the mask added | `[batch, heads, queries, keys]` | `exact reconstruction` |
| `probs` | `exp(scores - lse)` | `[batch, heads, queries, keys]` | `exact reconstruction` |

`lse` accepts `heads=` and `query=` only. Axis indices follow standard PyTorch
indexing semantics:

- Passing a single `int` (or a 0-d integer tensor) selects that position and
  squeezes the dimension from the returned tensor. Negative integers count from
  the end.
- Passing a `slice`, a `list` of ints, or a 1-D+ integer tensor selects those
  positions and preserves the dimension.
- Passing a boolean tensor selects the positions where the mask is `True`.
- Passing a `bool`, `float`, or floating-point tensor raises `TypeError`.
- Out-of-range integer indices raise `IndexError` on the line where the site was
  requested.

Requesting an unsliced `probs` or `scores` tensor at long context materializes
`batch x heads x L x L` fp32 values; any request whose output would exceed 1 GiB
emits a `UserWarning` before allocating.

```python
attn = model.model.layers[6].self_attn

with model.trace(prompt):
    last_row  = attn.kernel.probs(heads=4, query=-1).save()               # [batch, keys]
    two_heads = attn.kernel.probs(heads=[0, 4], query=slice(-5, None)).save()  # [batch, 2, 5, keys]
    lse       = attn.kernel.lse().save()                                  # [batch, heads, queries]
    scores    = attn.kernel.scores(heads=0, key=0).save()                 # [batch, queries]
```

Masking reflects what the kernel received:

- Under `is_causal=True` or a boolean mask, masked entries in `scores` are
  `-inf` and masked entries in `probs` are `0`.
- Under an additive float mask, `scores` includes the float mask's values (in
  transformers, the minimum representable value of the model dtype) and `probs`
  is `0` at masked positions.
- If every key in a query row is masked (such as a left-padding query position),
  that row in `probs` is set to all `0`. Every other row sums to 1.

### Under eager attention

Under `attn_implementation="eager"`, the attention module calls the
`eager_attention_forward` function of its modeling module, which builds the
probability matrix in Python and returns it alongside the attention output.
The same three sites are served from that call:

| Site | Value under eager | Provenance |
|---|---|---|
| `probs` | the weights tensor the call returned and multiplied with the values, in the model dtype | `exact export` |
| `scores` | `q . k * scale` in fp32 with the mask added, computed beside the call | `replay` |
| `lse` | logsumexp of those fp32 scores | `replay` |

`scores` and `lse` carry provenance kind `replay` because the eager function
computes its own scores in the model dtype. `provenance.kernel` records the
function name (for example, `modeling_gpt2.eager_attention_forward`). Because
the eager mask is additive and already includes causal masking, `scores`
contains the mask's values directly. On GPT-2 in fp32, `exp(scores - lse)`
matches the returned weights within 1e-6 on every row that has at least one
unmasked key. On a fully masked row (such as a left-padding query position),
the eager call returns a uniform row and `probs` serves that uniform row; the
replayed `lse` equals the mask's `finfo.min` value because `min + log(keys)`
rounds back to `min` in fp32, so `exp(scores - lse)` does not match `probs` on
that row.

`probs` is always served from the returned weights, while `scores` and `lse`
raise `ValueError` whenever the call modified or normalized the scores using
terms outside `q . k * scale + mask`:
- When the function receives a non-`None` argument beyond the seven shared
  parameters (such as Gemma's `softcap`, T5's `position_bias`, or
  ModernBERT-decoder's `sliding_window`), `scores` and `lse` raise
  `ValueError("kernel site 'lse': modeling_gemma2.eager_attention_forward received softcap=50.0, which the replay of its scores from query, key and the mask does not model; ...")`.
- When the returned rows sum to less than 1 without dropout (such as when
  GPT-OSS concatenates `module.sinks` before the softmax and drops the sink
  column afterwards), `scores` and `lse` raise
  `ValueError("... the weights ... returned sum to 0.6210 on some rows, so the call normalised over more than the keys (attention sinks); ...")`.

Modeling modules whose eager function is named `simple_`, `vision_`, or
`text_eager_attention_forward` (TimesFM, Phi-4 multimodal, Llama 4 vision, and
TIPS v2) are registered and served the same way. Drilling into the eager
function's body with `.source` takes precedence on that call, as for every
registered kernel ([below](#recurrent-states-of-a-chunked-kernel)).

## Provenance

Every tensor returned by a kernel site has a `.provenance` attribute holding a
`Provenance(kind, kernel, note)` named tuple:

- `kind="exact export"`: a tensor written directly by the kernel (`lse`).
- `kind="exact reconstruction"`: a tensor computed from the kernel's inputs and
  an exported statistic using an exact mathematical identity (`scores`, `probs`).
- `kind="replay"`: a tensor computed in fp32 alongside the kernel because no
  exported statistic was available. This occurs when the SDPA dispatcher falls
  back to the fp32 math backend, or when a backend that only writes `lse` on
  request (memory-efficient attention or cuDNN) is launched a second time with
  `compute_log_sumexp=True` and that second launch does not produce bit-identical
  attention outputs. In all cases, the model's forward pass uses the first
  launch's output.
- `kind="derived"`: a tensor that the chunked algorithm never forms directly,
  computed by advancing an exported boundary state using the kernel's
  intra-chunk formula truncated at the requested token (the state at a token
  inside a chunk, [below](#recurrent-states-of-a-chunked-kernel)). The `note`
  field reports the maximum relative difference between the formula at the end
  of the chunk and the kernel's next exported boundary state.

`kernel` records the name of the operator or function that ran (for example,
`aten._scaled_dot_product_flash_attention`,
`aten._scaled_dot_product_cudnn_attention`, or `math backend (fp32)`). `note`
describes the formula used and, when `dropout_p > 0`, notes that dropout is
applied after the returned probabilities.

`.provenance` is attached to the tensor returned by the site call. Creating a
new view or copy (such as `probs[0]` or `probs.clone()`) does not copy the
attribute, but saving the tensor preserves `.provenance` across `torch.save` and
`torch.load`.

## Grouped-query attention, masks, decoding

- **GQA:** Query head `h` maps to key head `h // (Hq // Hkv)`, matching both
  `enable_gqa=True` and transformers' `repeat_kv`. The `heads=` argument always
  indexes query heads.
- **Causal:** Under `sdpa` and `eager`, causal calls use top-left alignment
  (query `i` attends to keys `0..i`), matching PyTorch's `is_causal=True`, and
  whenever query and key lengths differ transformers passes an explicit
  attention mask. FlashAttention 2 aligns its causal mask bottom right: query
  row `r` of `Lq` attends to keys `0..r + (Lk - Lq)`, and during decoding
  transformers passes `causal=True` with no mask.
- **Decoding:** During `generate`, each decode step calls the kernel with a
  single query row against the KV cache, so `probs()` at step `n` has shape
  `[batch, heads, 1, prompt + n]` under a dynamic cache. You can collect sites
  across generation steps using `tracer.iter` ([iteration](iteration.md)); a
  request at step `n` is served on the `n`-th call even if earlier steps did not
  request a site.
- **Gradients:** Returned tensors are detached from autograd. Because the
  kernel's `lse` is captured below autograd, differentiating through
  `exp(S - lse)` would treat `lse` as a constant and produce incorrect
  gradients.

## Ordering

A kernel site is served when the kernel executes inside the module's forward
pass, so you must request it before reading any value produced later in the
forward pass. Requesting `attn.kernel.probs(...)` after reading `attn.output`
raises `OutOfOrderError`
([out-of-order-error](../errors/out-of-order-error.md)). Values produced
earlier in the forward pass, such as `attn.inputs` or `attn.c_attn.output`, can
be read before requesting `attn.kernel.probs(...)` in the same block. Calling a
kernel site outside a trace raises `ValueError("Cannot access ... outside of
interleaving")`.

If you request a site on a module whose forward pass never calls a registered
kernel, the location never fires and the trace ends with `OutOfOrderError:
'<path>.kernel.i0' was requested but the model already ran past it`
([module-never-fired](../errors/module-never-fired.md)). Common cases include:

- An MLP module, a full decoder block (request `.kernel` on its attention
  submodule instead), or a full MoE block (request `.kernel` on its `experts`
  submodule instead).
- A model loaded with `attn_implementation="flash_attention_3"` or `"flash_attention_4"`. Only FlashAttention 2 is served.
- A FlashAttention 2 call made while a model with a different flash attention
  implementation runs inside the same trace. transformers holds one flash
  implementation at a time and swaps it on that model's forward.
- An eager attention module whose forward pass computes attention inline
  instead of calling a module-level `*eager_attention_forward` function, such
  as GPT-2 with `reorder_and_upcast_attn=True` and older architectures (BLOOM,
  CodeGen, GPT-J, DeBERTa).
- An MoE layer running under `experts_implementation="eager"` (or `deepgemm`,
  `sonicmoe`, FP8, or `kernels`).
- A vLLM model, where attention and state-space layers run through vLLM's own
  custom ops.

## Recurrent states of a chunked kernel

Gated delta rule layers (Qwen3-Next, Qwen3.5), Kimi delta attention layers
(Kimi Linear, GLM-5-Next), and Mamba2 mixers run their recurrences in chunks.
Inside a chunk, the kernel operates on matrices across the chunk's tokens and
only materializes the recurrent state at chunk boundaries. No implementation
forms the recurrent state at interior tokens of a chunk during the forward pass.

However, the kernel holds the chunk-boundary states and the intra-chunk
intermediates computed along the way:

- For the gated delta rule: pairwise decays, keys, and delta-corrected values.
- For Kimi delta attention: the same quantities with a per-key-channel decay
  vector.
- For Mamba2: segment decays, discretized inputs, and `B`.

Using those quantities, the state after any interior token can be computed by
taking the preceding chunk's boundary state and applying the kernel's own
intra-chunk update formula truncated at that token. By contrast, Mamba mixers
(Mamba, FalconMamba, Jamba, Zamba) run their selective scan sequentially token
by token, so every token's state is exported directly from the loop.

```python
mixer = model.model.layers[0].linear_attn          # Qwen3-Next; `backbone.layers[0].mixer` on Mamba2 and Mamba, `model.layers[0].self_attn` on Kimi Linear

with model.trace(prompt):
    boundaries = mixer.kernel.chunk_states().save()            # [batch, chunks + 1, heads, K, V]
    s_100 = mixer.kernel.states(tokens=100, heads=3).save()    # [batch, K, V]
    tail = mixer.kernel.states(tokens=slice(-5, None)).save()  # [batch, 5, heads, K, V]
```

Per head, each state has shape `[K, V]` on gated delta rule and Kimi delta
attention layers and `[head_dim, state_size]` on Mamba2. On Mamba, each channel
has state shape `[state_size]` and the `heads=` argument indexes the channel
axis. Both sites always return fp32 tensors.

| Site | Value | Provenance |
|---|---|---|
| `chunk_states(chunks=, heads=)` | the state entering each chunk; index `0` is the initial state, and index `C` is the final state when the kernel produced one (`output_final_state`, which transformers sets when a cache is in use) | `exact export`; the note names the dtype the kernel stored the state in (fp32 for transformers' reference implementations and `fla`'s final state; the input dtype for `fla`'s chunk buffer, which carries one bf16 rounding) |
| `states(tokens=, heads=)` | the state after each requested token | `exact export` where the token ends a chunk whose next boundary was exported; otherwise `derived`: the preceding boundary advanced in fp32 by the kernel's intra-chunk formula over its recorded intermediates, using one contraction per touched chunk |

Note that `chunks=-1` selects the final state only when the kernel produced one;
the note says `the kernel returned no final state` otherwise. To read the state
after the last token regardless of whether a final boundary state was exported,
use `states(tokens=-1)`.

When `states` returns a `derived` tensor, its provenance note checks the
intra-chunk formula by running it to the end of each touched chunk and reporting
the maximum relative difference against the kernel's next exported boundary
state across the batch:

- On transformers' fp32 reference implementations in the test models, this
  relative difference is 0 for the gated delta rule and Kimi delta attention,
  and about 1e-6 relative for Mamba2 (because the contraction sums in a
  different order). Compared against a sequential fp64 per-token recurrence over
  the same inputs, the derived states of all three chunked families and Mamba's
  exported states agree within 2e-6 absolute error.
- On `fla`, the intra-chunk intermediates remain inside Triton kernels, so
  interior token states are derived by stepping the preceding chunk boundary
  token by token with the fp32 per-token recurrence. In bf16, the provenance
  note reports a relative drift of about 5e-3 against the next boundary, which
  reflects the bf16 rounding of `fla`'s exported chunk boundary; the derived
  states themselves match transformers' fp32 recurrence within 2e-5, matching
  the accuracy of the exported boundaries.
- During `generate`, each decode step calls a single-token recurrent kernel, so
  `states(tokens=-1)` is an `exact export` of the state after that token
  (`torch_recurrent_gated_delta_rule` or `fla`'s fused recurrent kernel on gated
  delta rule layers, `recurrent_kimi_delta_attention` on Kimi delta attention,
  `mamba2_selective_state_update` on Mamba2, and `mamba_selective_state_update`
  on Mamba).

When a trace batches multiple invokes of different lengths, `tokens=` and
`chunks=` index positions in the left-padded batch: `tokens=-1` is each invoke's
final token, and states at left-padding positions are `0`.

Supported recurrent kernels include transformers' Python reference
implementations (`torch_chunk_gated_delta_rule`,
`torch_recurrent_gated_delta_rule`, `chunk_kimi_delta_attention`,
`recurrent_kimi_delta_attention`, `mamba2_chunk_scan`, `mamba_selective_scan`,
and the two `selective_state_update` functions), which every model in these
families uses when `fla` or `mamba_ssm` is not installed, along with `fla`'s
`chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` when `fla` is
installed. For the Python reference kernels, the serve function runs the kernel
under AST instrumentation (the same mechanism `.source` uses) to record its
boundary-state and intermediate bindings. For `fla`, a wrapper on
`chunk_gated_delta_rule_fwd_h` captures the chunk-state buffer before the forward
pass discards it. In both cases, the model's output is unchanged. Kernels are
registered process-wide when a `TransformersModel` is built from the hub
wrappers in its modeling module; a standalone `NNsight(module)` wrapper for one
of these architectures can serve recurrent sites only after a corresponding
`TransformersModel` has been built in the same process.

You can combine `.source` and `.kernel` in a single trace when `.source` stops
at the kernel call. On GPT-2, `attn.source.attention_interface_1.source`
exposes transformers' `sdpa_attention_forward`, where
`torch_nn_functional_scaled_dot_product_attention_0` is the kernel call: its
`.inputs` are served before the call, `attn.kernel.probs(...)` is served during
the call, and its `.output` is served after the call, so request them in that
order. If a trace drills into a Python kernel function's own body via `.source`
(for example,
`mixer.source.torch_chunk_gated_delta_rule_0.source.implementation_0.source`),
the `.source` operations inside the kernel body are served on that call and any
`.kernel` request on the module for the same call raises `ValueError("... is
drilled with .source in this trace; its kernel sites are not served on that
call")`. Use either `.source` into the kernel body or `.kernel` on a given
call.

## Per-expert rows of a grouped mixture-of-experts kernel

A mixture-of-experts (MoE) layer routes each token to `top_k` experts and sums
their weighted outputs. In transformers, the `grouped_mm` experts implementation
(the default when PyTorch's grouped matmul is supported, falling back to `eager`
otherwise) executes all routed (token, expert) pairs across the forward pass as
two grouped matrix multiplications over rows sorted by expert ID. During
autoregressive decoding off the CPU, `generate` switches to `batched_mm`, which
runs the pairs in token order by gathering one weight matrix per pair. In both
implementations, the per-pair inputs, activations, and down-projected outputs
exist only as internal row buffers before the module returns their sum. The
`.kernel` sites on the `experts` submodule return those rows in token order in
the dtype the forward pass used.

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

Slot `k` is the token's k-th routed expert, `top_k_index[token, k]`. Passing
`expert=e` (one int) replaces `slot=` and gives the expert's rows per token
before the routing weight, with `0` for a token not routed to it.

Although the `experts` module itself operates on a flattened batch x sequence
axis (as seen on `.input` and `.output`), the sites fold that axis by the
trace's batch rows and return only the requesting invoke's rows. As a result,
`tokens=` indexes sequence positions and a decode step under `generate` serves
`[batch, 1, ...]`. Keep three consequences in mind:

- In a batch of invokes of different lengths, `tokens=` indexes the left-padded
  positions, and a pad position holds the pad token's rows. Reading
  `experts.inputs` directly inside an invoke returns the whole batch's input, as
  for any module input.
- Calling `generate` with `num_return_sequences=n` or `num_beams=n` expands each
  prompt to `n` rows, which the fold concatenates along the sequence axis:
  `[batch, n x seq, ...]` at prefill and `[batch, n, ...]` at a decode step,
  while each invoke still reads only its own rows.
- A token count that the batch rows do not divide raises `ValueError` at the
  request. A plain `Envoy` wrapping the `experts` module directly counts its
  entire input as one row, giving shapes `[1, tokens, ...]`.

Measured on a tiny Qwen2-MoE (CPU, fp32): the exported rows match a
one-pair-at-a-time fp64 computation within fp32 rounding (6e-9 on activations of
size 3e-2), the weighted sum over slots matches the module's output exactly, and
the logits are bit-equal with and without a request. The same invariants hold
across six MoE families (Mixtral, OLMoE, Qwen3-MoE, DeepSeek-V3, Nemotron-H
without a gate, and GPT-OSS with transposed weights and biases) and on
Qwen1.5-MoE-A2.7B in bf16 on a GPU.

Sites are served for the `grouped_mm` and `batched_mm` implementations. Under
`experts_implementation="eager"`, the per-expert loop runs in Python and can be
reached with `.source` (`experts.source.experts_forward_1.source`). `.source`
also reaches the local variables of the fused forwards in expert-sorted order
(`experts.source.experts_forward_1.source.proj_out_2.output`), whereas the
`.kernel` sites reorder them back into token order.

## FlashAttention 2

A model running with `attn_implementation="flash_attention_2"` serves the same
`lse`, `scores` and `probs` sites, from the `flash_attn` package or the
`kernels-community/flash-attn2` kernel that transformers falls back to when the
package is missing (the fallback needs the `kernels` package). The kernel
returns its per-row logsumexp from the launch that fills the output, and
`probs` is `exp(scores - lse)` in fp32 with provenance `exact reconstruction`.
Each trace loads the model's implementation before it starts, so a model
switched to FlashAttention 2 after loading (`set_attn_implementation`, or a
remote request's `attn_implementation`) is served from its next trace.

Several things differ from the `sdpa` path:

- transformers calls `flash_attn_varlen_func` instead of `flash_attn_func` for
  a padded batch, which it packs into one row per real token, and for
  sequences packed into one row through `position_ids`. The query and key axes
  therefore cover a sequence's own tokens and hold no padding positions: an
  invoke with a 7-token prompt batched beside a 12-token one gets
  `[1, heads, 7, 7]`, and a row packing two 4-token sequences gets
  `[2, heads, 4, 4]`. An invoke whose sequences have different lengths has no
  rectangular result and raises `ValueError`; use one invoke per prompt.
- The first axis counts the kernel's sequences. When those are the trace's
  batch rows, each invoke gets its own; otherwise every invoke gets all of
  them, as `.output` and the `sdpa` sites do. Beam search is the common case:
  `generate(num_beams=2)` over two invokes runs four rows, and each invoke's
  `probs()` holds all four.
- A call is served only when every argument that changes the scores is one the
  reconstruction models. A sliding window, a logit soft cap, ALiBi slopes or
  dropout makes the request raise `ValueError` naming the argument. So does a
  `flash_attn_varlen_func` from `flash_attn` before 2.6.0, which returns its
  logsumexp padded to `[batch, heads, max_seqlen_q]`.

## Supported kernels

Each row below corresponds to an entry in a source catalog: `CATALOG` in
`intervention/attention.py` for PyTorch's SDPA kernel, and `CATALOG` and
`FLASH` in `modeling/kernels/transformers.py` for kernels called by
transformers models (`FLASH` is registered at the start of each trace).
Each catalog entry specifies a row label, a site family, a locator that yields
the functions and holders to patch, and a serve function. Recurrent rows are
labeled by their hub-kernel name (prefixed with `fla.` for the `fla` library
rows); the function a Python reference row registers has a different name
(for example, the row `chunk_gated_delta_rule` registers
`torch_chunk_gated_delta_rule`).

| Kernel | Family | Sites | How it is found |
|---|---|---|---|
| `scaled_dot_product_attention` | attention | `lse`, `scores`, `probs` | The attribute of `torch.nn.functional`, registered when nnsight is imported. |
| `eager_attention_forward` | attention | `lse`, `scores`, `probs` | Every `*eager_attention_forward` function in the model's modeling modules. |
| `flash_attn_func` | attention | same | The FlashAttention function transformers loads for `attn_implementation="flash_attention_2"` and calls for a batch without padding. |
| `flash_attn_varlen_func` | attention | same | Same, for the function it calls for a padded batch or sequences packed through `position_ids`. |
| `grouped_mm_experts_forward` | experts | `expert_input`, `expert_activation`, `expert_output` | The `grouped_mm` entry of transformers' experts-implementation mapping. |
| `batched_mm_experts_forward` | experts | same | The `batched_mm` entry of that mapping. |
| `chunk_gated_delta_rule` | recurrent | `chunk_states`, `states` | The hub-kernel wrapper of that name, running transformers' Python reference. |
| `fused_recurrent_gated_delta_rule` | recurrent | same | Same, for the single-token decode kernel. |
| `chunk_kda` | recurrent | same | Same, for Kimi delta attention. |
| `fused_recurrent_kda` | recurrent | same | Same, for its single-token decode kernel. |
| `mamba_chunk_scan_combined` | recurrent | same | Same, for Mamba2's chunked scan. |
| `selective_state_update` | recurrent | same | Same, for the single-token decode update of Mamba2 and Mamba1. |
| `selective_scan_fn` | recurrent | same | Same, for Mamba1's selective scan. |
| `fla.chunk_gated_delta_rule` | recurrent | same | The `chunk_gated_delta_rule` hub-kernel wrapper, running the `fla` library's kernel. |
| `fla.fused_recurrent_gated_delta_rule` | recurrent | same | The `fused_recurrent_gated_delta_rule` hub-kernel wrapper, running `fla`'s single-token decode kernel. |

A hub-kernel wrapper calls the library kernel when the library is installed and
the Python reference otherwise. For the two gated delta rule kernels, the `fla.`
row covers the library kernel and the unprefixed row covers the reference. The
other recurrent kernels have a reference row only, so a wrapper running a
library kernel is covered by no row and serves no sites (see [Limits](#limits)).

To support another kernel, write its serve function and add a
`nnsight.intervention.kernel.Spec(name, family, locate, serve)` row, where
`locate(context)` yields `(function, holders)` pairs for the storage locations
from which the model reads each function. `kernel.attribute(obj, name)` builds
a locator for a function read as an attribute, and
`kernel.register_catalog(catalog, context)` registers the rows. A row appended
to the transformers `CATALOG` applies to models constructed afterward.

## Limits

- **Reads only:** A site cannot be assigned to. To modify attention scores, add
  a bias to the mask the module passes to the kernel, or edit `q` and `k` before
  the call ([access-and-modify](access-and-modify.md)).
- **Tensor parallelism:** Under tensor parallelism, each rank holds `heads / tp`
  query heads, and `heads=` indexes the local rank's heads; rank 0 returns its
  own share without a gather
  ([tensor-parallel](../models/tensor-parallel.md#what-is-not-supported)).
  Recurrent sites are full on every rank because transformers gathers the
  projections before calling the kernel.
- **Unsupported recurrent and attention kernels:** `mamba_ssm`'s fused
  `mamba_chunk_scan_combined`, `selective_scan_fn`, and `selective_state_update`
  are not registered, and neither are `fla`'s `chunk_kda` and
  `fused_recurrent_kda`; only the transformers reference implementations (and
  `fla`'s gated delta rule kernels) are covered. Mamba's `mambapy` and
  associative-scan paths keep only the final state and are served as
  `Unavailable`. LFM2's short convolution has no recurrent state, and RWKV's
  linear attention has no registered entry. `fla`'s packed layout
  (`cu_seqlens`), its `use_gate_in_kernel` mode, and `fla` backends that bypass
  `chunk_gated_delta_rule_fwd_h` (such as FlashQLA on SM90 and later) are served
  as `Unavailable`, raising a `ValueError` with the reason at the request. A
  model whose hub wrapper is replaced by the `kernels` library has no recurrent
  entry. Paged attention has no sites. vLLM's fused MoE has no entry: its
  `FusedMoE` layer runs experts inside the custom op
  `torch.ops.vllm.moe_forward`, whose Triton kernels keep per-pair rows in
  op-local workspace buffers that neither AST instrumentation nor a dispatch mode
  can see (and under `taps`, the engine also replays recorded CUDA graphs).
- **Parallel experts:** Experts under transformers' tensor or
  expert parallelism are served as `Unavailable`, because each rank holds only a
  shard of the intermediate axis and a partial down projection, or holds other
  ranks' pairs as sentinels.
- **Submodule name collisions:** A submodule named `kernel` is accessed as
  `E_kernel` ([access-and-modify](access-and-modify.md#overloaded-names)).
- **Registering a custom kernel:**
  `nnsight.intervention.kernel.register(fn, serve, *holders)` (called by
  `register_catalog` for each function a row locates) associates a
  callable `fn` with a serve function `serve(fn, args, kwargs) -> (output,
  sites)` and the storage locations from which the model looks up `fn`:
  `Attr(module, name)` for a module attribute, `Attr(cell, "cell_contents")`
  for a closure cell, and `Item(mapping, key)` for a dictionary entry. When
  `fn` is a Python function whose own module global holds `fn`, that module
  attribute is added automatically. While a trace is open, each holder is
  patched with a shim that serves the innermost running module when a worker is
  waiting on its `.kernel` location. A module that imported `fn` by value
  (`from torch.nn.functional import scaled_dot_product_attention`) and calls
  that local binding does not reach the shim unless that module is passed as a
  holder. A function instrumented by `.source` executes with a copy of its
  module's globals dict; `kernel.adopt` registers every kernel found in that
  copy as a `Global` holder for the lifetime of the copy, and `register` scans
  all live copies when a new kernel is registered, so a kernel called as a bare
  global inside an instrumented function also reaches the shim. Any holder that already contains a different
  callable before the trace opens (such as a custom attention function or a
  local override in transformers' `ALL_EXPERTS_FUNCTIONS`) is not patched, and
  a site request on it raises `OutOfOrderError`.
- **Identity checks on the kernel function:** `torch._C._nn`, where the C++
  SDPA builtin resides, is PyTorch's dispatch table and is not patched, so
  `__torch_function__` dispatch and `torch.compile` observe the original
  builtin. A `__torch_function__` handler or `TorchFunctionMode` that tests
  `func is torch.nn.functional.scaled_dot_product_attention` does not match
  while a trace is open: `func` is the builtin and the attribute holds the
  shim. `CausalBias` in `torch.nn.attention.bias` then falls back to its generic
  path, and diffusers' `AttentionScoreSkipFunctionMode` (`hooks/layer_skip.py`,
  used by skip-layer guidance) computes attention normally instead of skipping
  it inside the trace.
- **Gemma 2 attention softcap:** Gemma 2's attention logit softcap is dropped by
  transformers on the SDPA path, and the served values match what the SDPA
  kernel computed. Under `attn_implementation="eager"`, `eager_attention_forward`
  applies the softcap inside the call, so `probs` returns the post-softcap
  weights and `scores` and `lse` raise `ValueError` naming `softcap`.
- **`torch.compile`:** A module that executes inside a compiled region does not
  push a frame onto the active-module stack and the shim calls the kernel
  directly, so a site request on a compiled module raises `OutOfOrderError`
  ("requested but the model already ran past it") and compilation introduces no
  extra graph breaks. If the kernel call itself runs eagerly inside a compiled
  module (for example, inside a helper decorated with `torch._dynamo.disable`),
  the call is attributed to the innermost enclosing module that is running
  eagerly. Keep modules from which you read kernel sites uncompiled.
