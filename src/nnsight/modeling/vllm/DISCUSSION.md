# Interpretability in Production: NNsight's Approach

## The Problem

There is growing demand for deploying interpretability at scale: a chat interface backed by a production LLM where you can observe internal representations, steer behavior via activation manipulation, and serve this to many concurrent users — all without retraining.

Companies like Goodfire have productized a specific slice of this: they train Sparse Autoencoders (SAEs) on model activations, decompose the residual stream into human-labeled "features," and let users steer generation by adjusting feature weights through an API. The user says "be more concise" or sets `variant.set(conciseness_feature, 0.7)`, and the SAE modifies the residual stream at a fixed layer during every generated token.

The broader problem is harder than what SAE-based steering addresses. Researchers and engineers want to:

- **Observe** activations at arbitrary layers, not just the one layer where an SAE is trained
- **Intervene** with arbitrary logic: patching activations from one prompt onto another, zeroing specific neurons, applying learned steering vectors, running custom classifiers on intermediate states
- **Compose** multiple interventions in a single forward pass
- **Iterate** — change what you're looking at without retraining an SAE or waiting for a new feature dictionary
- **Scale** — do all of this on production-grade inference infrastructure, not a research notebook with `model.generate()` on a single GPU

This is the problem NNsight's vLLM integration solves.

## How NNsight Solves It

NNsight gives you programmatic access to every intermediate tensor in a model's forward pass. The vLLM integration runs this on top of vLLM's high-performance inference engine — PagedAttention, continuous batching, tensor parallelism — so you get research-grade introspection with production-grade throughput.

### The Core Abstraction

You write Python code that references model internals. NNsight extracts it, compiles it, serializes it, ships it to GPU workers, and executes it inline with the forward pass:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("meta-llama/Llama-3.1-70B", tensor_parallel_size=4, dispatch=True)

with model.trace("The Eiffel Tower is in", temperature=0.0) as tracer:
    # Read any layer
    h5 = model.model.layers[5].output[0].save()

    # Modify any layer
    model.model.layers[10].mlp.output = model.model.layers[10].mlp.output.clone()
    model.model.layers[10].mlp.output[-1, :] = 0

    # Access logits and sampling
    logits = model.logits.output.save()
    sampled = model.samples.output.save()
```

This is not a constrained API with predefined operations. It's arbitrary Python running inside the forward pass. You can call `torch.svd()`, run a learned probe, apply an SAE you trained yourself, or do anything else you'd do in a research notebook — but it executes on vLLM workers with PagedAttention and TP sharding handled transparently.

### What Happens Under the Hood

1. **Tracing**: Your intervention code is extracted via AST parsing, compiled into a closure, and wrapped in a `Mediator`
2. **Transport**: The mediator is serialized (source-level, not bytecode — works across Python versions) and attached to `SamplingParams.extra_args`, riding vLLM's existing request pipeline
3. **Execution**: On the GPU worker, the mediator runs in a thread that synchronizes with PyTorch forward hooks. When your code reads `model.layers[5].output`, the thread blocks until layer 5 fires, receives the real tensor, and continues
4. **TP transparency**: The `VLLMBatcher` gathers sharded tensors before your code sees them and re-shards after. Your intervention code always sees complete, unsharded tensors regardless of `tensor_parallel_size`
5. **Collection**: Saved values are pickled back through vLLM's request output path to the user process

### Batching and Multi-User

Each `tracer.invoke()` becomes one vLLM request. vLLM's scheduler batches them for GPU efficiency:

```python
prompts = ["Hello", "World", "Test"]

with model.trace(temperature=0.0) as tracer:
    activations = [None] * len(prompts)

    for i, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            activations[i] = model.model.layers[-1].output[0].save()
```

Each invoke runs its own intervention code independently. vLLM handles the GPU batching. For a multi-user serving scenario, NDIF (the National Deep Inference Facility) already provides this: multiple users submit traces to a shared model deployment, and each trace runs in isolation.

## Benefits

### Generality

NNsight imposes no constraints on what you can observe or modify. Any module, any layer, any tensor — inputs, outputs, attention weights, MLP intermediates, logits, sampled tokens. You're not limited to a pre-trained SAE's feature vocabulary at a single layer.

This matters because:

- **SAE features are lossy.** An SAE at layer 19 of an 8B model captures what's decomposable at that layer with that architecture. Concepts distributed across layers, or represented in ways the SAE's sparsity penalty doesn't reward, are invisible. Independent evaluations show SAE-based steering can degrade coherence by 20-50% on benchmarks while simple prompting achieves comparable behavioral changes with zero degradation.
- **Research moves fast.** Today's best intervention technique might be activation patching; tomorrow it might be representation engineering, probing classifiers, or something not yet invented. NNsight doesn't bake in a technique — it provides the primitive (tensor access) that all techniques build on.
- **Different tasks need different access patterns.** Mechanistic interpretability needs attention patterns and MLP intermediates. Safety research needs to monitor hidden states for deceptive behavior. Alignment research needs gradient information. A fixed feature-steering API serves one use case.

### Performance

The vLLM backend provides:

- **PagedAttention**: Near-zero memory waste for KV cache management
- **Continuous batching**: New requests join the batch without waiting for others to finish
- **Tensor parallelism**: Transparent sharding across multiple GPUs with automatic gather/scatter for interventions
- **Efficient scheduling**: vLLM's scheduler handles chunked prefill, preemption, and priority — NNsight interventions ride on top without disrupting these optimizations

The intervention overhead is the cost of PyTorch hooks (microseconds per module), thread synchronization (one context switch per accessed module), and serialization/deserialization (once per trace, typically a few KB). For any real model, this is negligible compared to the forward pass itself.

### Openness

NNsight is fully open source. You run it on your own hardware, with your own models, with no API keys or rate limits. You can inspect every line of the integration code, modify it, and extend it. The serialization format is documented. The hook points are explicit.

This contrasts with hosted black-box APIs where you can't verify what the SAE actually does, can't inspect the serving stack, can't run offline, and are constrained to the provider's supported models (currently two).

### Composability

Interventions compose naturally because they're just Python:

```python
with model.trace(prompt, temperature=0.0) as tracer:
    # 1. Run a probe on layer 5
    h5 = model.model.layers[5].output[0]
    toxicity_score = my_probe(h5[:, -1, :])

    # 2. Conditionally steer layer 10 based on the probe
    if toxicity_score > 0.8:
        model.model.layers[10].output[0][:, -1, :] += safety_vector

    # 3. Log everything
    logits = model.logits.output.save()
    score = toxicity_score.save()
```

There's no separate "feature search" step, no "variant" object, no round-trip to a discovery API. You write the logic you want and it executes.

## Downsides and Limitations

### Requires Expertise

NNsight gives you raw tensors, not human-labeled features. You need to know:

- What layer to look at
- What the tensor dimensions mean
- How to interpret activations
- What intervention will produce the effect you want

Goodfire's abstraction — "here are named features, set this one to 0.7" — is dramatically more accessible to product engineers. NNsight is a research tool, not a product API.

### No Built-In Interpretability

NNsight tells you what the tensor values are. It doesn't tell you what they mean. If you want human-readable feature labels, you need to train your own SAE, run your own automated interpretability pipeline, or use another tool. NNsight provides the mechanism for running an SAE inline (pass the activation through your SAE module using `hook=True`), but the SAE itself is your responsibility.

### Operational Complexity

Running NNsight + vLLM requires:

- GPU infrastructure you manage yourself
- vLLM version compatibility (currently pinned to 0.15.1)
- Understanding of vLLM's execution model (flat tensors, continuous batching, TP sharding)
- `enforce_eager=True` (CUDA graphs are incompatible with arbitrary hooks)

A hosted API eliminates all of this. For teams that want "just steer the model," the operational overhead of self-hosting is a real cost.

### eager-mode Requirement

NNsight requires `enforce_eager=True` because CUDA graph captures freeze the computation graph — PyTorch hooks can't fire inside a captured graph. This means you lose vLLM's CUDA graph optimization, which can be significant for decode-heavy workloads (small batch, single token per step). For prefill-heavy workloads (large prompts), the difference is smaller because prefill already runs in eager mode.

### Multi-User Isolation

The current vLLM integration doesn't have built-in multi-tenant isolation. `Globals.saves` is process-global. For production multi-user serving, you'd need either NDIF (which provides this) or your own isolation layer. This is solvable but not solved out of the box.

### No Pipeline Parallelism (Yet)

NNsight's execution model assumes one mediator thread can access all modules in the model. With pipeline parallelism, modules are split across stages on different GPUs, and no single worker has the full model. Supporting PP requires exposing rank information to intervention code so users can write stage-aware interventions (see [IDEAS.md](./IDEAS.md)).

## Where This Fits

The interpretability tooling landscape has two poles:

**High-level, constrained, production-ready**: Goodfire-style APIs where the provider trains SAEs, labels features, and exposes a steering knob. Low barrier to entry, limited flexibility, hosted and opaque.

**Low-level, unconstrained, research-oriented**: NNsight, where you get raw tensor access with arbitrary intervention logic on top of a real inference engine. High barrier to entry, maximum flexibility, open and self-hosted.

These aren't competitors — they serve different users with different needs. A product team adding guardrails to a chatbot wants named features and a REST API. A research team studying polysemanticity across layers wants tensor-level access at every module.

NNsight's contribution is proving that you don't have to choose between "research flexibility" and "production performance." The same intervention code that works in a notebook with a toy model works on a 70B model sharded across 4 GPUs with PagedAttention and continuous batching. The gap between "I ran an experiment" and "I can run this at scale" is one import change.
