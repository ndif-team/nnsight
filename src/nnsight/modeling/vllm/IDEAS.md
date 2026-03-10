# vLLM Integration: Future Ideas

## Pipeline Parallelism Support

### The Problem

NNsight's execution model assumes one mediator thread per invoke, running on one worker, with all modules accessible from that thread. The mediator blocks waiting for module values in forward-pass order as hooks fire.

With PP=2 on a 32-layer model:
- **Stage 0** (GPU 0): layers 0-15, has its own `NNsightGPUModelRunner`
- **Stage 1** (GPU 1): layers 16-31, has its own `NNsightGPUModelRunner`

If a user writes:
```python
with model.trace("Hello"):
    out_5 = model.layers[5].output.save()    # exists on stage 0
    out_20 = model.layers[20].output.save()   # exists on stage 1
```

The mediator thread runs on one worker. Stage 0's runner only has layers 0-15 in its Envoy tree. Stage 1's runner only has layers 16-31. No single worker can service both requests.

The forward pass is physically split — stage 0 runs, sends `IntermediateTensors` to stage 1 via NCCL, stage 1 runs. NNsight's hooks only fire on modules physically present on each worker. A mediator thread can't block waiting for `layers[20].output` on a worker that doesn't have layer 20.

### Proposed Approach: Expose PP/TP Rank

Each PP stage runs its own copy of the mediator. Users guard interventions with rank checks:

```python
with model.trace("Hello") as tracer:
    if tracer.pp_rank == 0:
        out_5 = model.layers[5].output.save()
    if tracer.pp_rank == 1:
        out_20 = model.layers[20].output.save()
```

Each stage's mediator independently executes the same code. The conditionals ensure each stage only accesses modules that physically exist on it. No cross-stage communication needed for the interventions themselves.

### Implementation Details

**Envoy tree coverage**: The worker-side `VLLM(self.model)` wraps only the local stage's modules. `model.layers[20]` wouldn't exist in the Envoy tree on stage 0. The conditional means it's never *accessed*, but it still needs to *parse* without error. The meta model on the client side has all layers, so tracing/compilation is fine. On the worker, the `if` branch is just dead code that never runs — as long as the Envoy attribute lookup doesn't happen eagerly, it works.

**Save collection across stages**: Currently `finish_nnsight` only collects from `pp_rank == 0`. Would need to collect from all ranks and merge. The `collective_rpc` already talks to all workers — just needs to not discard non-rank-0 results.

**Exposing rank**: The rank values need to be available inside the intervention function. They could be injected into the mediator's `__globals__` during deserialization on the worker, or exposed as properties on the interleaver/tracer.

**Cross-stage variable sharing**: If a user wants `out_5` from stage 0 to be used in stage 1's intervention, that's a harder problem (requires inter-stage send/recv). But for the common case of just reading/modifying activations per-stage independently, rank-guarded interventions are sufficient.

### Alternative Approaches Considered

**Split mediator code by stage**: Analyze which modules the user accesses, map to PP stages, generate per-stage intervention functions, and forward variables between stages. Transparent API but extremely complex — requires dependency analysis and cross-stage serialization.

**Proxy remote modules from stage 0**: Stage 0 owns the full Envoy tree. When hooks fire on remote stages, values are sent to stage 0 via RPC. Stage 0's mediator runs normally. Cleanest API but adds round-trip latency for every cross-stage module access.

---

## Feature Parity Gaps

Features available in `LanguageModel` but not yet ported to vLLM:

| Feature | Effort | Notes |
|---------|--------|-------|
| Scan mode | Low | Works at the tracing layer, no vLLM-specific changes needed |
| Caching API (`tracer.cache()`) | Low | Same — tracing layer feature |
| Module renaming | Low | Config forwarding |
| Model editing (`model.edit()`) | Medium | Envoy system already wraps the model |
| Module skipping | Medium | Needs testing with flat tensor format and batch groups |
| Source tracing | Medium | Only works on Python-level forward methods, not fused CUDA kernels |
| Gradients | Hard | Requires backward tracing in worker processes |

---

## vLLM Feature Integration

### Plugin System

vLLM has an official `vllm.general_plugins` entry point system. NNsight currently integrates via monkey-patching (`NNsightGPUWorker` replaces `GPUModelRunner`) and the `worker_cls` string parameter. The plugin system runs too late in the initialization process to replace the worker class — `worker_cls` is already the intended vLLM mechanism for custom workers. Plugins could handle minor tasks like registering custom attention backends or logits processors.

### Multi-Modal Models

vLLM supports LLaVA, Phi-3.5-vision, InternVL, Qwen-VL, etc. High research value for studying cross-modal representations (intercepting vision encoder outputs, patching visual features). Challenge: encoder disaggregation means the vision encoder may run on separate workers — would need a separate interception point.

### Speculative Decoding

Eagle 3 reuses target model layer features for draft generation. NNsight could expose draft/verify phase boundaries for research into model uncertainty. Challenge: draft/verify pipeline uses custom CUDA kernels tightly integrated with the scheduling loop.

### Partial CUDA Graph Compatibility

vLLM V1's piecewise CUDA graph design runs attention in eager mode between graph-captured segments. NNsight currently forces `enforce_eager=True`. Could potentially allow compiled segments to run optimized while hooking into eager attention portions. Non-trivial: interventions at arbitrary layers (not just attention) would break graph captures.

### Online Serving

Current integration is offline (`LLM`) only. An NNsight-aware server endpoint could enable real-time activation analysis during serving. vLLM's middleware and callback systems provide extension points. Challenge: async processing vs NNsight's thread-based synchronization model.
