---
title: Kernel Support Map
one_liner: Which fused kernels `.kernel` sites cover, ranked by what the major runtimes default to.
tags: [internals, dev, kernel]
related: [docs/usage/kernel.md]
sources: [src/nnsight/intervention/attention.py, src/nnsight/modeling/kernels/transformers.py, src/nnsight/modeling/kernels/eager.py, src/nnsight/modeling/kernels/flash.py, src/nnsight/modeling/kernels/recurrent.py, src/nnsight/modeling/kernels/experts.py]
---

# Kernel support map

Which fused kernels nnsight's `.kernel` sites cover, and the order in which the
rest get done.

## How this map is organised

- **Unit: a kernel lineage.** One algorithm with one way of exposing its
  internals, such as FlashAttention, which exports a per-row logsumexp for its
  backward pass. One reconstruction technique covers a whole lineage.
- **Entry point:** a function through which a runtime reaches a lineage. This is
  what nnsight shims, and what a catalog row in `docs/usage/kernel.md` names.
- **One table per operation family:** attention, recurrent and state-space, and
  mixture of experts. Only attention is surveyed so far.
- **Priority** comes from the lineage's position in the ecosystem, read from what
  the runtimes default to in their source. Within a lineage, entry points are
  ordered by how many runtimes go through them, then call variants by how many
  current models set them.
- **Status** records nnsight's own level of support only, on this scale: not
  started, default call served, validated (compared with a reference on a
  pretrained model), in CI. CI runs on CPU only, so a GPU kernel is at most
  validated.

Survey date 2026-11-12. Versions read from installed source: torch 2.13.0,
transformers 5.19.0, vLLM 0.29.0, diffusers 0.41.0, flashinfer 0.6.18. SGLang is
from its attention-backend documentation and, for `trtllm_mha`, its
`trtllm_mha_backend.py` on the main branch (2026-11-13).

## Attention

### What the runtimes default to

| Runtime | Default attention path | Source |
|---|---|---|
| PyTorch | `F.scaled_dot_product_attention`, with the flash, memory-efficient, cuDNN and math backends all enabled | `torch.backends.cuda.*_sdp_enabled()` |
| transformers | `sdpa`, falling back to eager when a model cannot dispatch to it. `flash_attention_2/3/4` on request: FlashAttention 2 from the `flash_attn` package or the `kernels-community/flash-attn2` hub kernel, 3 from `flash_attn_interface`, 4 from `flash_attn.cute` | `modeling_utils.py` `get_correct_attn_implementation`; `modeling_flash_attention_utils.py` `_lazy_imports` |
| diffusers | `dispatch_attention_fn` with the `native` backend in most model files under `models/transformers/`, `AttnProcessor2_0` in the others; both call `F.scaled_dot_product_attention` | `utils/constants.py` `DIFFUSERS_ATTN_BACKEND`; `models/attention_dispatch.py`; `models/attention_processor.py` |
| vLLM | FlashAttention, then FlashInfer, then Triton; FlashInfer first for causal attention on Blackwell (SM100), where it calls FlashInfer's `trtllm_batch_*` functions. FlashAttention version 3 on Hopper (compute capability 9), 4 on compute capability 10, 2 elsewhere. MLA models use a separate list | `platforms/cuda.py` `_get_backend_priorities`; `v1/attention/backends/fa_utils.py`; `v1/attention/backends/flashinfer.py` |
| SGLang | FlashAttention 3 on Hopper; `trtllm_mha`, which calls FlashInfer's `trtllm_batch_*` functions, on Blackwell; FlashInfer elsewhere, Triton when FlashInfer is absent | docs.sglang.io, attention backend page; `layers/attention/trtllm_mha_backend.py` |

### Lineages in priority order

**1. FlashAttention.** The default or first choice in PyTorch, transformers,
diffusers, vLLM, and SGLang on Hopper.

| Entry point | Reached from | Status |
|---|---|---|
| `F.scaled_dot_product_attention` (flash, memory-efficient, cuDNN and CPU flash backends, math fallback) | transformers `sdpa`, diffusers, any PyTorch model | in CI for the CPU backends. The CUDA backends are validated by the GPU tests, which CI does not run. Diffusion models not validated |
| FlashAttention 3 and 4 | transformers `flash_attention_3/4`, vLLM on Hopper and Blackwell, SGLang on Hopper | not started; needs a Hopper or Blackwell GPU |
| `flash_attn_func`, `flash_attn_varlen_func` (FlashAttention 2) | transformers `flash_attention_2`, from the `flash_attn` package or `kernels-community/flash-attn2` | default call served. CPU tests run transformers' FlashAttention 2 path end to end with stand-in kernels; on an A100 the GPU test matches the `sdpa` sites on a two-layer random Llama with the hub kernel, including beam search and a model switched after loading. Not run with the `flash_attn` package |
| `flash_attn_varlen_func` in vLLM's fork | vLLM `FLASH_ATTN` | default call validated with FlashAttention 2 on an A100, on the unmerged `kernel-vllm` branch |
| transformers paged attention (`flash_attn_with_kvcache` for the flash implementations) | transformers continuous batching | not started |
| any of the above inside a `torch.compile` region | compiled models | not started |

Open call variants:

- transformers FlashAttention 2: sliding window, logit soft cap, dropout, and
  an invoke holding sequences of different lengths. transformers never passes
  ALiBi slopes to the flash functions.
- vLLM: sliding window, logit soft cap, ALiBi, attention sinks, a per-row causal
  tensor, a custom mask function, an fp8 key cache, keys passed as a tensor, and
  calls vLLM merges into one attention (cascade, context-parallel).

transformers' `eager` implementation runs no fused kernel. Its
`*eager_attention_forward` sites come from the returned weights plus an fp32
replay of the scores, and are in CI.

**2. FlashInfer.** First choice in SGLang off Hopper (through `trtllm_mha` on
Blackwell), and in vLLM for causal attention on Blackwell. Its prefill and
decode wrappers' `run` methods and its `trtllm_batch_context_with_kv_cache` and
`trtllm_batch_decode_with_kv_cache` functions accept `return_lse=True`, so the
FlashAttention technique may carry over; not confirmed by a run.

| Entry point | Reached from | Status |
|---|---|---|
| FlashInfer prefill and decode wrappers | vLLM `FLASHINFER`, SGLang `flashinfer` | not started |
| `trtllm_batch_context_with_kv_cache`, `trtllm_batch_decode_with_kv_cache` | vLLM `FLASHINFER` on SM100, SGLang `trtllm_mha` | not started |

**3. Triton unified attention.** The fallback in vLLM and SGLang. vLLM's 2D
kernel keeps no logsumexp. Its 3D path, taken for decode-only batches below a
sequence-count threshold, writes per-segment maxima and exponential sums
(`softmax_segm_max`, `softmax_segm_expsum`) and reduces them across segments,
which gives the logsumexp. Decode may therefore carry over the FlashAttention
technique; prefill needs a different one.

| Entry point | Reached from | Status |
|---|---|---|
| `unified_attention` | vLLM `TRITON_ATTN` | not started |

**4. MLA kernels** (FlashMLA, FlashInfer MLA, CUTLASS MLA, TRTLLM MLA, Triton MLA
and their sparse variants). Used only by multi-head latent attention models such
as DeepSeek V3.

| Entry point | Reached from | Status |
|---|---|---|
| vLLM `v1/attention/backends/mla/` | vLLM | not started |
| SGLang's MLA backends | SGLang | not started |

Not placed yet: `flex_attention` (both transformers and vLLM register it), and
the ROCm and CPU backends.

## Recurrent and state-space kernels

Not surveyed. Served today: gated delta rule, Kimi delta attention, Mamba2 and
Mamba1 through transformers' hub-kernel wrappers (Python reference, with or
without the `kernels` package installed), and the `fla` gated delta rule
kernels. vLLM registers `gdn_attn`, `linear_attn`, `short_conv_attn`,
`mamba1_attn` and `mamba2_attn` backends; none is shimmed.

## Mixture of experts

Not surveyed. Served today: transformers' `grouped_mm` and `batched_mm` experts
forwards. vLLM's fused MoE is not shimmed.
