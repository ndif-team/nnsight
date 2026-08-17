---
title: Backend design for gradient workloads on very large models
one_liner: What it takes for nnsight/NDIF to support Jacobian-lens-style features on models at Kimi K3 scale, and which execution backend fits.
tags: [developing, design, gradients, ndif, moe]
status: exploration, no implementation started
date: 2026-08-10
---

# Backend design for gradient workloads on very large models

This doc records a design exploration: if nnsight/NDIF wants to let users run gradient-requiring interpretability features (Jacobian lens collection, attribution patching, tuned-lens or probe training) on very large open models, what does the execution backend need to be, and what does it cost? Kimi K3 (2.8T parameters, MoE) is used as the reference model throughout.

## 1. The driving example: J-lens

The Jacobian lens ([Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)) builds one transport matrix per tapped layer:

    J_l = E over (source position t, target position t' >= t, prompt) of  d h_final,t' / d h_l,t

averaged over roughly one thousand pretraining-like prompts, for about 25 evenly spaced layers. The lens is then `softmax(W_U norm(J_l h_l))`. Producing a J-lens for a model means computing an expected d x d Jacobian per layer and storing the matrices. There is no gradient descent and no parameter update.

### Collection recipe on nnsight

One backward pass seeded with direction v at the final residual stream yields row v of J_l for every tapped layer and every source position at once. So the algorithm is: per prompt, one grad-enabled forward, then d backward passes (one per basis direction) with `retain_graph=True`, accumulating a running mean per layer. Reverse mode amortizes over layers; forward-mode JVPs do not (a seed at layer l informs only layer l), so reverse mode is the right direction when many layers are tapped.

nnsight's existing primitives cover this locally on the eager LanguageModel path: `requires_grad_(True)` on tapped outputs, `with target.backward(retain_graph=True):`, `.grad` reads in reverse module order (see `docs/usage/backward-and-grad.md`).

### Gradients are needed, with one qualifier each way

- The Jacobian is a property of the function between layer l and the final layer. Cached activations cannot produce it; collection must run with autograd enabled and the graph retained (or recomputed) from every tapped layer to the top.
- Only gradients with respect to activations are needed. Parameters stay frozen; no optimizer, no weight gradients.
- A forward-only estimator exists: patch `h_l += eps * v`, run forward, compare `h_final` against a clean run. This is a numerical JVP and uses only activation patching, which every nnsight backend supports, including vLLM. Cost is roughly 12x the FLOPs of reverse mode (25 layers x d full forwards versus d backwards), partly recovered by inference-stack throughput and native low-precision compute, and it parallelizes as batch rows. Risk is numerical noise through quantized forwards (MXFP8 activations on K3); the averaging over positions and prompts absorbs some of it. A validation run comparing finite-difference J against autograd J on a small model at matched precision is required before trusting it.

Consequence: for J-lens alone there is a zero-new-infrastructure path on the serving stack at roughly an order of magnitude more forward compute. The grad backend is justified by the features with no forward-only substitute: attribution patching, per-sample gradients, probe/tuned-lens training.

## 2. What the workload keeps in memory

The gradient workloads carry far less state than a training run. Per-parameter training state (parameter gradients, Adam moments, fp32 master weights, DP reduce buffers, roughly 16+ bytes per parameter, ~45 TB at 2.8T scale) never materializes: `requires_grad=False` on all parameters means autograd allocates no parameter gradients, and no optimizer is constructed.

Persistent state is the frozen weights, identical to inference sizing: 1.4 TB at K3's native MXFP4, 5.6 TB dequantized to bf16. Transient state is the per-prompt activation graph: at interp scale (sequence ~1k, small batch) a few GB per prompt, negligible against the weights. Holding it across d backward seeds via `retain_graph` is therefore fine.

Backward for a frozen model computes only the input-gradient GEMM per layer and skips the weight-gradient GEMM, so backward costs about 1x forward, half the usual training backward.

## 3. Backend survey

The organizing question for nnsight integration cost: what do activations look like at module boundaries? Sharding parameters is free for us; sharding activations is what forces merge semantics.

| Backend | Params for 2.8T | Activations at hook points | MoE backward | nnsight-side cost |
|---|---|---|---|---|
| vLLM / SGLang / TensorRT-LLM | sharded | sharded, no autograd | none | not viable for gradients |
| HF + FSDP2 / DeepSpeed ZeRO-3 | sharded, gathered per layer | full ordinary tensors | eager expert loop (slow at 896 experts) | near zero |
| DeepSpeed ZeRO-Infinity | sharded + CPU/NVMe offload | full | same eager loop | near zero |
| torchtitan (FSDP2 + DTensor TP/EP) | sharded | DTensor with explicit placements | grouped-mm with backward | moderate |
| Megatron-Core / Transformer Engine | TP/EP/PP sharded | raw tensors, shard semantics implicit in code | grouped GEMM, mature | high |
| KTransformers | quantized in host RAM, CPU experts | full, single process | custom AMX kernels, scoped to LoRA needs | low |

Notes per row:

- **Inference stacks** are structurally out for gradients: `no_grad` throughout, CUDA-graph capture, fused kernels without registered backwards (FlashMLA decode, fused MoE, marlin-class FP4 GEMMs). Our own nnbench probes confirmed vLLM gradient reads are silently wrong. Their strengths (continuous batching, paged KV) target decode, which Jacobian collection does not have.
- **FSDP2 / ZeRO** ranks each run the full layer computation, so every hooked activation is a complete tensor and the merge problem does not exist. The weakness is that they execute whatever the eager modeling code does; a naive per-expert loop at 896 experts is the known Nemotron failure mode (20-35x slowdown).
- **torchtitan** is the PyTorch team's pretraining platform: single-device model code, parallelism applied externally via DTensor, EP with grouped-mm MoE. Production evidence at target scale: the [PyTorch + Nebius DeepSeek-V3-671B recipe](https://nebius.com/blog/posts/inside-the-nebius-pytorch-deepseek-v3-recipe) with wide EP over DeepEP/NVSHMEM, and [1K-GPU MoE runs](https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/). For nnsight the decisive property: sharded activations are DTensors that carry their placement (replicated, sharded, partial) as typed metadata. The merge layer becomes "read the placement, redistribute". Our vLLM trace-scope merge design exists to reconstruct exactly this information by hand.
- **Megatron-Core** has the highest performance ceiling and the most battle-tested trillion-scale MoE path, with Transformer Engine supplying the best low-precision recipes on NVIDIA hardware (the MXFP4-native path for K3's weights is most likely to appear there first). The costs for us: shard semantics live implicitly in handwritten forwards (a tensor is partial or sharded depending on where the collectives sit), TE fusions collapse module boundaries we hook, and the spec system plus checkpoint conversion add integration friction.
- **KTransformers** ([kvcache-ai](https://github.com/kvcache-ai/ktransformers)) splits the model across one machine: attention, dense, routing on GPU; the expert pool quantized in host RAM executed with AMX/AVX-512 kernels. Published fine-tuning results: DeepSeek-V3-671B LoRA SFT on 2-4 RTX 4090s with 1.2-1.3 TB host RAM ([user guide](https://kvcache-ai.github.io/ktransformers/en/SFT/KTransformers-Fine-Tuning_User-Guide.html)), integrated as a [LLaMA-Factory backend](https://github.com/hiyouga/LLaMA-Factory/issues/9266) with Kimi-family MoE support. Because LoRA gradients must flow through frozen quantized experts, the CPU kernels implement backward with respect to input, which is exactly the activation-gradient path. Single process, so no distributed merge problem at all. Costs: CPU expert throughput is orders below a GPU cluster (practical mainly with sketched direction counts), backward kernel coverage is scoped to what LoRA exercised, and the fast path requires Intel AMX. Hook compatibility (module injection preserves the nn.Module tree) is plausible and unverified.

### Existence proofs that backward-capable K3 stacks are buildable

Public fine-tuning support for this model class already exists: [Fireworks offers K3 SFT/preference/RL as a managed service](https://x.com/FireworksAI_HQ/status/2082245266720371011?lang=en); LLaMA-Factory has a Megatron-Core backend (mcore_adapter) and the KTransformers path. K3's novel layers have trainable public kernels: Kimi Delta Attention ships in the flash-linear-attention library with Triton backwards; MLA trains in Megatron-Core; grouped-mm MoE has backward in torchtitan and Megatron. No per-op backward math needs writing; the work is assembling and validating a sharded implementation of this architecture and converting the checkpoint.

## 4. Parallelism choice: EP + FSDP2, no TP, no PP

On a MoE model the expert pool is nearly all the parameters, and expert parallelism shards it by construction: 1.4 TB of FP4 experts over 64 GPUs is ~22 GB per GPU. The backbone (attention, dense, embeddings) is small and FSDP2 handles it. Memory never asks for PP; PP's pretraining role is throughput on thousand-GPU runs, which an interp service does not monetize.

Dropping PP deletes the only two genuinely new design problems found in this exploration:

1. **Multiple backward seeds per forward under PP.** Pipeline schedules free stage activations after their one backward; `retain_graph` across a schedule is not a supported concept. Workarounds exist (recompute per seed at ~1.5x total compute, or seed batching), but without PP there is one autograd graph per rank and d seeds work exactly as on a single GPU.
2. **Microbatch interleaving versus nnsight's whole-batch trace model.** Without PP there is no schedule and no microbatching.

Without TP, the residual stream at every layer boundary is a full ordinary tensor on each data-parallel rank (batch-sharded only, which nnsight's invoke machinery already models). The only activations with exotic semantics are inside the MoE block: per-rank expert partials, the case already characterized on Qwen1.5-MoE during EP validation. The merge surface shrinks to one module type.

TP re-enters only if a single GPU cannot hold the backbone's per-layer compute; with short interp prompts and KDA being linear attention there is no sign of that. PP re-enters only if the interconnect cannot carry cross-node EP all-to-all (the standard wide-EP regime that DeepEP exists for) or if pipeline throughput ever becomes the binding constraint.

One shared hazard at any rank count: user trace code runs on every rank, and a rank-local conditional that skips a collective deadlocks the job. Known from vLLM TP work; the mitigation is executing interventions at semantically merged points so rank-divergent control flow cannot arise.

## 5. Running a training backend in stripped mode

Both candidate stacks support forward + backward with no trainer, because their parallelism is implemented inside autograd rather than inside the training loop:

- **FSDP2:** all parameters `requires_grad=False` is a first-class configuration; FSDP2 gathers for forward and backward and skips reduce-scatter for gradient-free parameters. Every QLoRA-on-FSDP2 stack (e.g. torchtune's LoRA recipes) runs a large frozen base with backward flowing through it. torchtitan's model construction and `parallelize_*` functions are importable; skip `build_optimizer` and drive `torch.autograd.grad` directly.
- **Megatron-Core:** TP collectives and the EP token dispatcher are `torch.autograd.Function`s with proper backwards, so a bare `megatron.core` model supports plain `backward()` with no pipeline runner. Skip their DDP wrapper (that is what allocates `main_grad` buffers and expects the grad-reduce path) and use the unwrapped module.

Limitation of the training-backend home: no serving-grade generation. Multi-token generation runs as an eager loop, slow but correct. The pool exists for single-forward Jacobians, attribution, and probe training, where this does not bind.

## 6. Costs (estimates; assumptions stated inline)

**Hardware fork, the dominant cost variable.** MXFP4-native forward and backward on Blackwell-class hardware keeps weights at 1.4 TB: roughly 8 nodes of 8 GPUs. If the training-side kernels cannot run MXFP4 on the available hardware generation, dequantize to bf16: 5.6 TB of weights, roughly 12-16 Hopper-class nodes. About a factor of two in standing cost, and it is a procurement decision rather than an engineering unknown. QLoRA-style dequantize-on-the-fly backward through frozen 4-bit weights (routine in KTransformers and bitsandbytes stacks) may close this fork from the cheap side. At market rates (~$2 per GPU-hour) standing cost spans roughly $95k-190k per month for 64-128 GPUs.

**J-lens job compute.** Assumptions: ~50B active parameters, sequence 1024, hidden dimension ~8k (K3's actual d is unpublished; K2 was 7168), 40% MFU, frozen-model backward at ~1x forward. Full collection (d backwards x 1000 prompts) is ~7e20 FLOPs, under a day on 64 H100-class GPUs, order of $1-3k per run. Sketched collection (a few hundred random directions) is 1-2 hours. Marginal job cost is small; hosting dominates.

**Engineering.** With sharding-merge work accepted as known-cost (extension of the vLLM trace-scope and EP work, plus a backward-direction rule set), the remaining items:

| Item | Nature | Size |
|---|---|---|
| K3 layer implementations (KDA via fla, latent MoE, gated MLA) in the chosen stack | shared by every path; kernels exist | weeks to a couple of months |
| Checkpoint conversion (HF MXFP4 to sharded training format) + logit-equivalence validation | bounded, unglamorous | weeks |
| Merge rules for the MoE block (only exotic activation site in the no-TP/no-PP config), forward and backward | extension of existing EP work | weeks |
| Job runner service around a resident sharded model | NDIF-shaped work | weeks to months |
| Ongoing upstream churn on per-module semantics (Megatron) or DTensor APIs (torchtitan) | maintenance tax | ongoing |

Order of magnitude: 2-3 engineer-quarters to a first working grad job at K3 scale, on top of hardware.

## 7. Today's NDIF execution substrate, and why its autograd support does not scale

The audited servers support remote activation-gradient backward (section 8), but that support rests entirely on how naive the execution substrate is. NDIF loads every model with `device_map="balanced"` and `accelerate.dispatch_model` (`deployments/modeling/base.py:222`, `:317-323`): layers are assigned whole to the GPUs of one node, and the forward walks them serially, hidden states hopping devices through accelerate's alignment hooks. One Ray actor process holds the model; user trace code, forward, and backward all run inside it.

Autograd works on this substrate for a structural reason: there are no collectives and no sharded tensors anywhere. Every activation is a whole tensor on some device, the device hops are autograd-transparent, and backward is one ordinary graph in one process. The simplicity that makes it work is the same property that caps it:

- **Model size ceiling = one node's aggregate GPU memory.** accelerate dispatch cannot cross nodes and NDIF's controller assumes one process per model replica. A 2.8T-parameter model (1.4 TB FP4, 5.6 TB bf16) is out of range regardless of gradient support.
- **Throughput ceiling = one GPU at a time.** The serial layer walk leaves N-1 of N GPUs idle in both forward and backward. A d-seed Jacobian collection pays that idleness d times per prompt.
- **MoE ceiling = the eager per-expert loop.** No grouped GEMM, no expert parallelism; at hundreds of experts this is the known order-of-magnitude slowdown before gradients even enter.
- **Policy ceilings sized for this substrate:** the per-GPU memory cap (weights x 1.15 + 500 MB) and the 1-hour advisory timeout assume weights-dominated single-process inference, and a retain_graph backward blows through the former.

So the distance between today's NDIF and the grad pool of section 10 is a substrate change, and the design constraint it imposes is: scale up while preserving the property that made autograd work, whole activations and one autograd graph per process. That is the deeper reason section 4 lands on FSDP2 + EP with no TP and no PP: data-parallel sharding of parameters keeps every hooked activation whole and every rank's backward an ordinary single-process graph, so the working autograd semantics carry over unchanged, with the MoE block as the one exception to engineer.

What the substrate change costs on the NDIF side, independent of which training stack is chosen:

1. **Multi-rank worker groups.** The controller/actor model must go from one process per model to a placement group of N ranks with `torch.distributed` initialization, request broadcast to all ranks, and save-gathering from one. NDIF has none of this today (it does not even serve vLLM); the structure exists in our vllm-serve and PP branches.
2. **An execution-authority decision.** Either every rank executes the user's trace (introducing the rank-divergence collective-deadlock hazard at every conditional) or one authority process executes it and values cross a seam. ndif2's sandbox seam is the second shape but drops the autograd graph at its cloudpickle boundary; the mediator-isolation branch in this repo already ships grad reads across such a seam bit-identically, and is the closest existing code to what the grad pool needs.
3. **Re-derived resource policy.** Memory budgeting must account for activation graphs held across multiple backward seeds, and job control must handle hours-scale preemptible batch work rather than minutes-scale requests.

## 8. NDIF service gaps (audited 2026-08-10 against ndif dev @ bcbe20f, nnsight origin/0.8 @ f355044f, ndif2 main @ 517564a)

- **Remote backward on activations already works.** Neither server runs the model under `no_grad`/`inference_mode`; both freeze parameters at load (ndif `deployments/modeling/base.py:136`, ndif2 `deployments/modeling/base.py:170`) and leave autograd live. The pattern `activation.requires_grad_(True)` then `with loss.backward(): activation.grad` is tested remotely on both (ndif `tests/test_nnsight.py:235-262`, ndif2 `tests/test_nnsight_remote.py:159-172`). On 0.8 the nested `with backward():` block round-trips as source text and re-captures server-side (`schema/request.py:89-94`); no remote test covers it, and the server-execution tests run under `@torch.no_grad()`.
- **Weight gradients are dead by design.** Frozen params leave `weight.grad` silently `None`; ndif's `ProtectedObject` blocks `requires_grad_` on model modules and deepcopies weight reads (`security/protected_objects.py:38-96`). Adapter training per 0.8's `docs/patterns/remote-training.md` sidesteps this: adapter and optimizer are constructed inside the remote session, so their params are fresh and trainable while the host stays frozen.
- **Memory cap.** ndif caps GPU memory per process at weights x 1.15 + 500 MB (`evaluator.py:87`). A retain_graph multi-seed backward must fit its saved activations in that headroom; grad-pool sizing must be separate from inference sizing.
- **Job shape.** Default execution timeout is one hour, enforced by `PyThreadState_SetAsyncExc`, which cannot interrupt a C-level backward or CUDA kernel; jobs are preemptible mid-run on model eviction. Hours-scale collection needs a long-running tier or resumable accumulation across jobs; neither exists, and one failed trace aborts a session. Backward also executes inside the server's `autocast` region (`base.py:442`).
- **Sandbox seam has no grad plumbing.** ndif2's opt-in sandbox drops the autograd graph at the cloudpickle boundary and its protocol has no grad event; its default deployment bypasses the sandbox (`trusted=True` when auth is off) and runs user code in-process. Any future hardened execution path must ship grad hook events across the seam explicitly.
- **Whitelists allow training primitives.** ndif's execution whitelist admits `torch.optim`, `peft`, `nn.Parameter`, and user-shipped modules (`whitelist.yaml:185-199, 289-290`); nnsight 0.8 removed the client-side serialization whitelist entirely. Nothing gates training-shaped payloads; what stops them is frozen weights, the memory cap, and the timeout. Known hole: `module.parameters()`/`named_modules()` return real objects through the protection wrapper, so in-place module surgery persists on the shared replica across users' requests.
- **Server-side LoRA-by-hub-id** (`PeftModel.from_pretrained` from an env field) is merged and tested in ndif2, unmerged in ndif (`hackathon/peft-actor`, which also drops the `protect()` wrapping after the first request); both persist the adapter on the shared replica across requests.
- **Artifact storage.** Output is layers x d^2 floats (~6.4 GB fp32 at d=8192, 25 layers), and users will immediately want to apply the lens in later remote traces. Needed: server-side named tensor artifacts, persistent across jobs, referencable from traces. The analogue of register-local-modules for outputs.
- **Two-pool routing.** The service shape mirrors what RL frameworks (veRL, slime, OpenRLHF) productionized: an inference engine for no-grad jobs, a training engine for grad jobs, weight-sync plumbing between them. The dispatch signal is visible in the trace: presence of a backward context.

## 9. Verification checklist before committing

Probe results 2026-08-11 (`tests/manual/fsdp2_probe.py`, GPT-2 fp32 under `fully_shard` on 2 A100s, torch 2.11, vs unsharded single-GPU reference):

- Hooked activations, activation-grad backward, and a second backward seed via a second forward are all bit-identical to the reference (max diff 0.0). `.source` resolves on the `fully_shard`-wrapped module. Items 3 and the FSDP2 half of item 6 below are verified.
- MoE run (same probe, `--model qwen-moe`: Qwen1.5-MoE-A2.7B, 14.3B params, bf16, 2 A100s): routing weights, the experts module's combined output, the layer activation, and both grad seeds are all bit-identical to the unsharded reference (max diff 0.0), and `.source` resolves on the sharded experts module. Two structure notes from transformers 5.x: decoder blocks return plain tensors (the tuple-output gotcha is version-dependent), and MoE experts are one module with stacked weights and a Python loop rather than per-expert submodules, so per-expert visibility goes through `.source` on the experts module instead of child-module hooks.
- Megatron v0 backend (2026-08-16, `src/nnsight/modeling/megatron/`, probe `tests/manual/megatron_probe.py`): a same-process wrapper (`MegatronLM(LanguageModel)`) backing Qwen2.5-0.5B-Instruct with an mcore GPTModel (megatron-core==0.16.1, local unfused spec, no TE/apex). All probe checks pass against the fp32 HF oracle: layer-0 activations to 4e-7, logits to 7e-5 with 100% argmax, activation-grads to 8e-6 relative, `.source` on mcore attention, batched two-invoke traces with the [seq, batch, hidden] batcher, and cross-invoke swap isolation. bf16 gates are calibrated to measured noise floors (HF-bf16 itself is cos 0.9937/88.9% argmax and grad-rel 1.65e-1 from fp32; mcore-bf16 measured closer to fp32 on both). Contract cost matched the study: five overridden methods plus a ~40-line batcher; the HF-to-mcore converter (qkv per-group interleave, gate-first fc1 concat) is the per-architecture tax.
- Two findings. First, an nnsight bug: during `with tensor.backward():` the `torch.Tensor.grad` property patch intercepted FSDP2's own `param.grad` reads in `post_backward` on frozen params and raised `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`. Fixed (uncommitted, `src/nnsight/intervention/tracing/backwards.py`): the getter/setter falls through to the real descriptor for tensors with `requires_grad=False`. The 0.8 rewrite (`intervention/backward.py`) uses the same interception pattern and needs the same guard. Second, structural: `retain_graph` multi-seed backward fails under FSDP2 (`setStorage ... storage of size 0`) because gathered params are freed at post-backward, so the retained graph references freed storage. Multi-seed collection under FSDP2 must batch seeds into one backward call or re-run the forward per seed; per-seed re-forward is what the probe validates.

1. Finite-difference Jacobian versus autograd Jacobian on a small model at matched (quantized) precision. Decides whether J-lens ever needs the grad pool.
2. Multiple backwards against one retained graph through the fused-kernel set (FlashAttention-class, TE FP8 scaling metadata, EP dispatcher). Small-model experiment per stack.
3. All-frozen-parameter configuration smoke test on FSDP2 and bare Megatron-Core.
4. Quantized-Jacobian faithfulness: backward paths using straight-through surrogates would distort the measured Jacobian of a QAT model; validate against a bf16 reference on a small model.
5. KTransformers: Envoy hooks on the injected module tree, and backward coverage beyond the LoRA-exercised ops (DeepSeek-V2-Lite at 150 GB host RAM is the natural target).
6. torch.func / batched-VJP (`is_grads_batched`) composability with DTensor, as an optimization over the plain backward loop.

## 10. Recommendation

- For J-lens specifically, validate the finite-difference path on the existing serving stack first; it may need no new infrastructure at all.
- For the grad pool, fork torchtitan: FSDP2 + EP, no TP, no PP, all parameters frozen, no optimizer. Full-tensor activations everywhere except inside the MoE block; DTensor placements as typed input to any remaining merge logic; one autograd graph per rank so multi-seed backward works as on a single GPU.
- Ladder: FSDP2-only on eager modeling code first (slow MoE, correct, near-zero integration) to validate the nnsight-grad pipeline end to end, then grouped-mm EP for throughput.
- Megatron-Core is the escalation path if TE kernel support (MXFP4-native) or the compute bill at torchtitan efficiency becomes binding.
- KTransformers is the low-cost single-node alternative worth one probe (hooks + backward coverage), attractive when sketched direction counts make CPU-expert throughput acceptable.

## Sources

- [Verbalizable Representations Form a Global Workspace in Language Models (Transformer Circuits, 2026)](https://transformer-circuits.pub/2026/workspace/index.html)
- [Kimi K3 model overview: MXFP4, open weights (HuggingFace blog)](https://huggingface.co/blog/ResterChed/kimi-k3-model-overview-mxfp4-quantization-open-wei)
- [Nebius + PyTorch DeepSeek-V3 torchtitan recipe](https://nebius.com/blog/posts/inside-the-nebius-pytorch-deepseek-v3-recipe)
- [Efficient MoE pretraining at scale with torchtitan (PyTorch blog)](https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/)
- [KTransformers](https://github.com/kvcache-ai/ktransformers), [fine-tuning user guide](https://kvcache-ai.github.io/ktransformers/en/SFT/KTransformers-Fine-Tuning_User-Guide.html)
- [LLaMA-Factory KTransformers integration](https://github.com/hiyouga/LLaMA-Factory/issues/9266)
- [Fireworks K3 fine-tuning announcement](https://x.com/FireworksAI_HQ/status/2082245266720371011?lang=en)
- [torchao MXFP8 expert-parallel training](https://docs.pytorch.org/ao/0.17/eager_tutorials/mxfp8_expert_parallel_training.html)
