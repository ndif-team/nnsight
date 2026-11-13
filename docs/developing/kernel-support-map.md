---
title: Kernel Support Map
one_liner: Which fused kernels `.kernel` sites cover, ranked by what the major runtimes default to.
tags: [internals, dev, kernel]
related: [docs/usage/kernel.md]
sources: [src/nnsight/intervention/attention.py, src/nnsight/modeling/kernels/transformers.py]
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
  pretrained model), in CI.

Survey date 2026-11-12. Versions read from installed source: torch 2.13.0,
transformers 5.19.0, vLLM 0.29.0, diffusers 0.41.0, flashinfer 0.6.18. SGLang is
from its attention-backend documentation.

## Attention

### What the runtimes default to

| Runtime | Default attention path | Source |
|---|---|---|
| PyTorch | `F.scaled_dot_product_attention`, with the flash, memory-efficient, cuDNN and math backends all enabled | `torch.backends.cuda.*_sdp_enabled()` |
| transformers | `sdpa`, falling back to eager when a model cannot dispatch to it; `flash_attention_2/3/4` on request, from the `flash_attn` package or a `kernels-community` hub kernel | `modeling_utils.py` `get_correct_attn_implementation`; `modeling_flash_attention_utils.py` `FLASH_ATTN_KERNEL_FALLBACK` |
| diffusers | `AttnProcessor2_0`, which calls `F.scaled_dot_product_attention` | `models/attention_processor.py` |
| vLLM | FlashAttention, then FlashInfer, then Triton; FlashInfer first for causal attention on Blackwell (SM100). FlashAttention version 3 on Hopper, 4 on Blackwell, 2 elsewhere. MLA models use a separate list | `platforms/cuda.py` `_get_backend_priorities`; `v1/attention/backends/fa_utils.py` |
| SGLang | FlashAttention 3 on Hopper; `trtllm_mha` on Blackwell; FlashInfer elsewhere, Triton when FlashInfer is absent | docs.sglang.io, attention backend page |

### Lineages in priority order

**1. FlashAttention.** The default or first choice in PyTorch, transformers,
diffusers, vLLM, and SGLang on Hopper.

| Entry point | Reached from | Status |
|---|---|---|
| `F.scaled_dot_product_attention` (flash, memory-efficient, cuDNN and CPU flash backends, math fallback) | transformers `sdpa`, diffusers, any PyTorch model | in CI. Diffusion models not validated |
| transformers `*eager_attention_forward` | transformers `eager` | in CI |
| `flash_attn_func`, `flash_attn_varlen_func` (FlashAttention 2) | transformers `flash_attention_2`, from the `flash_attn` package or `kernels-community/flash-attn2` | default call served. CPU tests use stand-in functions; on an A100 the sites match the `sdpa` sites on a two-layer random Llama with the hub kernel. Not run on a pretrained model or with the `flash_attn` package |
| `flash_attn_varlen_func` in vLLM's fork | vLLM `FLASH_ATTN` | default call validated with FlashAttention 2 on an A100; not merged |
| FlashAttention 3 and 4 | transformers `flash_attention_3/4`, vLLM on Hopper and Blackwell, SGLang on Hopper | not started; needs a Hopper or Blackwell GPU |
| transformers paged variants (`paged|...`) | transformers continuous batching | not started |
| any of the above inside a `torch.compile` region | compiled models | not started |

Open call variants:

- transformers FlashAttention 2: sliding window, logit soft cap, ALiBi, dropout,
  the paged `block_table` path, and an invoke holding prompts of different
  lengths.
- vLLM: sliding window, logit soft cap, ALiBi, attention sinks, a per-row causal
  tensor, a custom mask function, an fp8 key cache, keys passed as a tensor, and
  calls vLLM merges into one attention (cascade, context-parallel).

**2. FlashInfer.** First choice in SGLang off Hopper and Blackwell, and in vLLM
for causal attention on Blackwell. Its prefill and decode `run` methods accept
`return_lse=True`, so the FlashAttention technique may carry over; not confirmed
by a run.

| Entry point | Reached from | Status |
|---|---|---|
| FlashInfer prefill and decode wrappers | vLLM `FLASHINFER`, SGLang | not started |

**3. Triton unified attention.** The fallback in vLLM and SGLang. Neither
`triton_attn.py` nor `ops/triton_unified_attention.py` in vLLM names a logsumexp
value, so it needs a different technique.

| Entry point | Reached from | Status |
|---|---|---|
| `unified_attention` | vLLM `TRITON_ATTN` | not started |

**4. MLA kernels** (FlashMLA, FlashInfer MLA, CUTLASS MLA, Triton MLA and their
sparse variants). Used only by multi-head latent attention models such as
DeepSeek V3.

| Entry point | Reached from | Status |
|---|---|---|
| vLLM `v1/attention/backends/mla/` | vLLM, SGLang | not started |

Not placed yet: `flex_attention` (both transformers and vLLM register it), and
the ROCm and CPU backends.

## Recurrent and state-space kernels

Not surveyed. Served today: gated delta rule, Kimi delta attention, Mamba2 and
Mamba1 through transformers' hub-kernel wrappers (Python reference), and the
`fla` gated delta rule kernels. vLLM registers `gdn_attn`, `linear_attn`,
`mamba1_attn` and `mamba2_attn` backends; none is shimmed.

## Mixture of experts

Not surveyed. Served today: transformers' `grouped_mm` and `batched_mm` experts
forwards. vLLM's fused MoE is not shimmed.
