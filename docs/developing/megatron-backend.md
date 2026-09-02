---
title: Megatron backend, current stage
one_liner: Forward and backward passes with hooks on a megatron.core model, single GPU, validated against the HF oracle on interp-serve-bench workloads.
tags: [developing, megatron, gradients]
sources: [src/nnsight/modeling/megatron/megatron.py, src/nnsight/modeling/megatron/loading.py]
---

# Megatron backend: current stage

The current stage is the forward-and-backward-with-hooks workflow: run a
megatron.core model under nnsight tracing, read and write activations at any
module, and compute gradients with respect to activations, on one GPU with no
parallelism. Generation, parallel layouts (TP/EP/PP), and remote execution are
later stages. megatron-core is pinned to 0.16.1; the layer spec is the local
(unfused) one, so every module is plain Python and hookable.

## Workflow

```python
from nnsight.modeling.megatron import MegatronLM

model = MegatronLM("Qwen/Qwen2.5-0.5B-Instruct", dispatch=True, dtype=torch.float32)

# read + write (module paths follow the mcore tree; decoder-internal
# activations are [seq, batch, hidden], wrapper boundaries are batch-first)
with model.trace("The mother tongue of Danielle Darrieux is"):
    h, *rest = model.gpt.decoder.layers[12].output
    model.gpt.decoder.layers[12].output = (h + steer_vec, *rest)
    logits = model.output.save()          # [batch, seq, vocab]

# gradients w.r.t. activations (params frozen; flag the earliest tapped layer,
# downstream taps inherit requires_grad; read .grad in reverse module order)
with model.trace(prompt):
    h = model.gpt.decoder.layers[12].output[0]
    h.requires_grad_(True)
    logits = model.output
    with logits[:, -1].sum().backward():
        grad = h.grad.save()
```

Batched multi-invoke traces work (`with tracer.invoke(...)`), including
gradients through per-invoke slices. `.source` resolves on mcore modules.
`generate`/`pipe` raise `NotImplementedError` at this stage.

## Workload coverage, in interp-serve-bench terms

The stage's supported surface is stated against the method inventory of
interp-serve-bench (`~/interp-serve-bench`, `isb/methodologies/`), whose HF
backend is the reference implementation of each workload and whose datasets
supply the prompts. `tests/manual/interp_workloads_probe.py` runs the in-stage
methods on this backend and the HF eager oracle with identical inputs
(CounterFact prompts, MIB IOI items, 4 each) and compares results.

| bench method | needs | status on this backend | measured vs HF oracle (fp32) |
|---|---|---|---|
| logit_lens | fwd read | validated | rel 6.5e-7, argmax 100% |
| steering | fwd write | validated | rel 5.7e-7, argmax 100% |
| ablation | fwd write | validated | rel 1.8e-6, argmax 100% |
| activation_patching | fwd write, 2 traces | validated | rel 1.3e-6, argmax 100% |
| attribution_patching | gradients | validated | rel 3.2e-6 on the [n_layers] attribution vectors |
| jacobian_collect | many-VJP sweep | validated by the equivalent `tests/manual/jlens_demo.py` (16-seed sketch, 3 layers, 8 prompts: rel < 3e-6, cos 1.0) |
| jacobian_lens | fwd read | supported (logit_lens plus one matmul); not separately demoed |
| das, train=0 | fwd write, 2 traces | supported (same mechanics as activation_patching); not demoed |
| das, train>0 | grads + optimizer loop | mechanism present (grads through frozen model reach an external parameter); not demoed |
| gen_steering, gen_patching | generation | out of stage: no decode loop |
| attention_pattern | `.source` internal site | the bench cell is gpt2-specific; `.source` resolves on mcore attention (347 op lines) but no cell-equivalent comparison exists |

The three bench methods that ERROR on the vLLM backend for lack of autograd
(attribution_patching, das train>0, jacobian_collect) are exactly the ones this
backend exists to serve; the first and third are validated above.

## Validation inventory

All comparisons are against the fp32 HF eager implementation of the same
checkpoint (Qwen2.5-0.5B-Instruct), which is the correctness oracle throughout.

- `tests/manual/megatron_probe.py`: layer-0 activations to 4e-7 (pins the qkv
  interleave and RoPE), logits to 7e-5 with 100% argmax, activation-grads to
  8e-6 relative, batched two-invoke traces, cross-invoke swap isolation, and
  the bf16 mode gated against measured bf16 noise floors.
- `tests/manual/jlens_demo.py`: the J-lens collection loop (one forward, 16
  sequential backwards on a retained graph, reverse-order grad reads at three
  tapped layers, accumulation over prompts).
- `tests/manual/interp_workloads_probe.py`: the five bench workloads above.
- `tests/manual/jlens_timing.py`: nnsight overhead is 3.8 ms per trace and
  zero per backward seed within noise; at demo scale wall-clock is the
  kernel-launch floor of unfused eager code (42 ms forward, ~20 ms per
  backward on a 9-token prompt), which amortizes with batch, sequence length,
  and model scale.

## Known limits of this stage

- Model coverage is the converter registry (`loading.py`), currently the
  qwen2 family; an unsupported `model_type` refuses to load with a clear
  error and cannot load wrong (strict accounting: every parameter written
  exactly once, every checkpoint tensor consumed).
- Trace-only: no generation.
- Single GPU: parallel layouts are stage 2 of the plan in
  `docs/developing/grad-workload-backend-design.md` (section 10); 0.8's
  `modeling/tp/` shard-aware interleaver rules are the in-house lineage for it.
- Lazy loading (`dispatch=False`) and `edit()` are inherited and untested.
- Batch-dim detection in `MegatronBatcher` is by size match and is ambiguous
  when a padded sequence length equals the total batch size (resolved as
  dim 0, the same heuristic class the stock batcher uses).

## Data provenance

Prompts in the validation scripts come from interp-serve-bench's snapshots:
CounterFact (Meng et al. 2022; `data/counterfact/counterfact.json`) and MIB
IOI (Mueller et al. 2025; `data/mib/ioi.json`), loaded from a local checkout
at `~/interp-serve-bench` with embedded samples as fallback.
