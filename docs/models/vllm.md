---
title: VLLM
one_liner: High-throughput vLLM serving with NNsight interventions; supports tensor parallelism, continuous batching, and async streaming.
tags: [models, vllm, serving, production]
related: [docs/models/index.md, docs/models/language-model.md, docs/remote/index.md]
sources: [src/nnsight/modeling/vllm/vllm.py:43, src/nnsight/modeling/vllm/batching.py:15, src/nnsight/modeling/vllm/sampling.py:4, src/nnsight/modeling/vllm/async_backend.py:19, src/nnsight/modeling/vllm/README.md, src/nnsight/modeling/vllm/IDEAS.md, src/nnsight/modeling/vllm/DISCUSSION.md]
---

# VLLM

## What this is for

`nnsight.modeling.vllm.VLLM` runs NNsight interventions on top of vLLM's high-performance inference engine. You get PagedAttention, continuous batching, tensor parallelism, and async streaming — with arbitrary Python intervention code executing inline with the forward pass.

Same tracing API as `LanguageModel`, but the model runs in vLLM workers (potentially across multiple GPUs / nodes) and your intervention code is serialized, transported via `SamplingParams.extra_args`, and executed in those workers.

This is the production / throughput path. For details on the architecture, read [`src/nnsight/modeling/vllm/README.md`](../../src/nnsight/modeling/vllm/README.md) and [`DISCUSSION.md`](../../src/nnsight/modeling/vllm/DISCUSSION.md).

## When to use / when not to use

Use `VLLM` when:
- You need **throughput** — vLLM is faster than HF `transformers.generate()` by an order of magnitude on real workloads.
- You need **tensor parallelism** across multiple GPUs (single node or multi-node via Ray).
- You're serving **multiple concurrent users** and want continuous batching.
- You want **async streaming** (token-by-token output with intervention saves on every step).
- You're doing **production interpretability** — running steering / probing / activation patching on a live service.

Do not use `VLLM` when:
- You only have a single prompt and don't need throughput — `LanguageModel` is simpler.
- You need features vLLM doesn't fully support yet: gradients (no backward in workers), source tracing on fused CUDA kernels, model editing, scan mode, or pipeline parallelism (PP > 1 is not supported; see `IDEAS.md`).
- You can't accept `enforce_eager=True` (see Limitations below). vLLM's CUDA graph optimization is incompatible with arbitrary PyTorch hooks, so NNsight forces eager mode.
- You're doing diffusion or VLM work — vLLM in NNsight is currently text-only.

## Loading

```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "meta-llama/Llama-3.1-8B",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    dispatch=True,
)
```

### Constructor

```python
VLLM(
    repo_id,
    *,
    mode="sync",                           # "sync" or "async"
    dispatch=False,                        # eager weight loading
    tensor_parallel_size=1,
    pipeline_parallel_size=1,              # NOT SUPPORTED — must be 1
    gpu_memory_utilization=0.9,
    distributed_executor_backend=None,     # "mp" (default), "ray", or an Executor class
    enforce_eager=True,                    # forced internally; required for hooks
    revision=None,
    rename=None,
    envoys=None,
    **vllm_kwargs,                         # forwarded to vllm.LLM / AsyncLLM
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo ID. |
| `mode` | `"sync"` (default) creates a `vllm.LLM` and runs synchronous generation; `"async"` creates a `vllm.v1.engine.async_llm.AsyncLLM` and yields a streaming async generator from `tracer.backend()`. See `vllm.py:70` and the [Async mode](#async-mode) section below. |
| `dispatch` | If `True`, real weights load now via vLLM's standard loader. If `False`, only the meta model is built (using vLLM's `DummyModelLoader` with `device="meta"`) — no GPU memory used until first trace. See `vllm.py:135`. |
| `tensor_parallel_size` | Number of GPUs to shard across. Tensor parallelism is **transparent** to your intervention code thanks to `VLLMBatcher` (`batching.py:15`). |
| `pipeline_parallel_size` | Currently must be `1`. Pipeline parallelism is on the roadmap but not yet supported (`IDEAS.md`). |
| `gpu_memory_utilization` | vLLM's KV-cache memory budget (default 0.9). Lower it (e.g. 0.1) for small models or shared GPUs. |
| `distributed_executor_backend` | `None` / `"mp"` (multiprocessing, default) or `"ray"` (Ray distributed executor; required for multi-node TP). When you pass `"ray"`, NNsight automatically swaps in `NNsightRayExecutor` to work around a vLLM/Ray actor crash. See `vllm.py:179` and `executors/ray_workaround.py`. |
| `enforce_eager` | Always set to `True` internally (`vllm.py:202`). CUDA graphs are incompatible with PyTorch hooks. |
| `worker_cls` | Always set internally to `nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker`. |
| `**vllm_kwargs` | Anything else valid for `vllm.LLM` / `AsyncEngineArgs` is forwarded. |

### Dispatch behavior

- `dispatch=False` (default) loads a meta-tensor placeholder via vLLM's `DummyModelLoader` with `device="meta"` (`vllm.py:151`). No GPU memory used. The Envoy tree is fully populated so you can write intervention code referencing `model.model.layers[5].output`. Real weights load on the first `.trace()` call (or explicit `model.dispatch()`).
- `dispatch=True` creates the `vllm.LLM` / `AsyncLLM` immediately during `__init__`.

The user-process `VLLM` instance has a `vllm_entrypoint` attribute pointing at the actual engine. There is a **second** `VLLM` instance created inside each worker process by `NNsightGPUModelRunner.load_model()` — it wraps the model that vLLM loaded and owns the interleaver and `VLLMBatcher`. See `vllm/README.md:113` for details.

## Canonical pattern

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("openai-community/gpt2", gpu_memory_utilization=0.1, dispatch=True)

with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
    hidden = model.transformer.h[-2].output[0].save()
    logits = model.logits.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### Multi-token generation with `tracer.iter`

```python
with model.trace("Madison Square Garden is in", max_tokens=3) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.logits)

# Each step's argmax is one generated token
print(model.tokenizer.batch_decode([l.argmax(dim=-1) for l in logits]))
# -> [' New', ' York', ' City']
```

### Sampling parameters

`SamplingParams` are forwarded via kwargs to either the root `.trace()` or per-invoke. NNsight wraps them in `NNsightSamplingParams` (`sampling.py:4`).

```python
# Root-level sampling params apply to all invokes by default
with model.trace("Hello", temperature=0.7, top_p=0.95, max_tokens=10) as tracer:
    samples = list().save()
    for step in tracer.iter[:]:
        samples.append(model.samples.item())

# Per-invoke sampling params
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.0, top_p=1.0):
        ids_greedy = list().save()
        for step in tracer.iter[:]:
            ids_greedy.append(model.samples.item())

    with tracer.invoke("Hello", temperature=1.5, top_p=0.95):
        ids_sampled = list().save()
        for step in tracer.iter[:]:
            ids_sampled.append(model.samples.item())
```

Common kwargs: `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `stop`, `stop_token_ids`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logprobs`. See `sampling.py:13-37` for the full set.

### Activation interventions

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1) as tracer:
    # Zero out the last MLP — changes the prediction
    model.transformer.h[-2].mlp.output = torch.zeros_like(
        model.transformer.h[-2].mlp.output
    )
    logits = model.logits.save()

# " London" instead of " Paris"
print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### Continuous batching: invoke loop

vLLM batches requests at the engine level. Each `tracer.invoke(prompt)` becomes **one** vLLM request (one prompt per invoke is enforced by `_prepare_input` at `vllm.py:266`). Multiple invokes within a single trace are submitted as separate requests but processed together by vLLM's continuous batcher.

```python
prompts = ["Prompt A", "Prompt B", "Prompt C"]

with model.trace(max_tokens=512) as tracer:
    out_ids = [list() for _ in range(len(prompts))].save()      # shared parent-scope list

    for i, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            for step in tracer.iter[:]:
                out_ids[i].append(model.samples.item())

for i, ids in enumerate(out_ids):
    print(f"{prompts[i]} -> {model.tokenizer.decode(ids)}")
```

Cross-invoke shared state works via the worker's globals-grafting machinery — all mediators for the same trace share the canonical `__globals__`. See `vllm/README.md:347`.

### Tensor parallelism is transparent

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=4, dispatch=True)

with model.trace("Hello", temperature=0.0):
    # Always sees the full unsharded tensor, regardless of tp_size
    hidden = model.model.layers[16].output[0].save()
    print(hidden.shape)        # [seq, hidden] — full hidden dim
```

`VLLMBatcher` (`batching.py:15`) registers pre/post hooks on `ColumnParallelLinear` and `RowParallelLinear` modules. When your intervention reads from one, the batcher gathers the sharded tensor; when you write back, it re-shards. Every TP rank runs the same intervention code on the same complete tensor.

### Async mode

Pass `mode="async"` to get token-by-token streaming:

```python
import asyncio
from nnsight.modeling.vllm import VLLM

model = VLLM("openai-community/gpt2", gpu_memory_utilization=0.1, dispatch=True, mode="async")

async def main():
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.save()

    async for output in tracer.backend():
        print(f"finished={output.finished}, text={output.outputs[0].text!r}")
        if output.finished:
            print("saves:", list(output.saves.keys()))

asyncio.run(main())
```

Behind the scenes: `VLLM.trace()` injects `AsyncVLLMBackend` (`async_backend.py:19`), which submits the request to `AsyncLLM.generate()` and returns an async generator that yields `RequestOutput` objects. On the **final** output, saves are pulled from workers via `collective_rpc("collect_nnsight", ...)` and attached as `output.saves`. (Saves on intermediate outputs are not currently collected in `__aiter__` — only on `output.finished`. See `async_backend.py:79`.)

### Ray distributed executor

For multi-GPU TP across the local node, `mp` (multiprocessing) is the default and works out of the box. For multi-node TP, pass `distributed_executor_backend="ray"`:

```python
model = VLLM(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel_size=8,
    distributed_executor_backend="ray",
    dispatch=True,
)
```

NNsight automatically:
1. Swaps in `NNsightRayExecutor` to work around a vLLM/Ray actor crash (`executors/ray_workaround.py`).
2. Connects to an existing Ray cluster (set `RAY_ADDRESS=head:6379`), or starts a fresh local one if none exists.
3. Joins as a driver-only node so no GPUs are consumed on the client machine.

See `vllm/README.md:629` for the full Ray section, and `vllm/examples/multi_node_with_ray/` for a Docker-based multi-node example.

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.logits` | `eproperty` — the **pre-sampling** logit tensor produced by the model. Read or modify via `model.logits.save()` / `model.logits = ...`. Iterates across generation steps via `tracer.iter`. | `vllm.py:102` |
| `model.samples` | `eproperty` — the **sampled** token IDs produced by the sampler after `.logits`. Available after sampling fires; iterates across generation steps. | `vllm.py:112` |
| `model.tokenizer` | vLLM's tokenizer (an `AnyTokenizer`). Loaded eagerly. | `vllm.py:165` |
| `model.vllm_entrypoint` | The underlying `vllm.LLM` (sync) or `AsyncLLM` (async). Only populated in the user process after dispatch. | `vllm.py:75` |
| `model.dispatched` | Whether real weights are loaded. | inherited from `MetaMixin` |
| `model._async_engine` | Boolean: `True` if `mode="async"`. | `vllm.py:73` |

`model.logits` and `model.samples` are vLLM-specific. Standard `LanguageModel` doesn't have them — those models expose `lm_head.output` (which fires before sampling) and the sampled tokens via `.generator.output` (final sequence only).

### Module structure

The Envoy tree mirrors vLLM's internal model layout. For Llama-style models you'll typically write:

```python
model.model.layers[i].self_attn.qkv_proj.output       # ColumnParallelLinear
model.model.layers[i].self_attn.o_proj.output         # RowParallelLinear
model.model.layers[i].mlp.gate_up_proj.output         # ColumnParallelLinear
model.model.layers[i].mlp.down_proj.output            # RowParallelLinear
model.model.norm.output
```

For GPT-2-style models in vLLM:

```python
model.transformer.h[i].attn.output
model.transformer.h[i].mlp.output
```

Print `model` to see the actual tree for your model.

## Limitations

- **`enforce_eager=True` is forced.** vLLM's CUDA graph optimization is incompatible with arbitrary PyTorch hooks. This costs you some throughput on decode-heavy workloads (see `DISCUSSION.md` for context).
- **Pipeline parallelism (PP > 1) is not supported.** A single mediator thread can't span multiple PP stages because each stage has only its own modules. Future work — see `IDEAS.md`.
- **One prompt per invoke.** Unlike `LanguageModel`, you cannot pass `tracer.invoke(["a", "b"])`. Each invoke = one vLLM request. Use a loop of invokes for multiple prompts (`vllm.py:267`).
- **No backward / gradients.** Backward tracing is not supported in vLLM workers (`IDEAS.md`).
- **No `.scan()`, no `tracer.cache()`, no module editing yet.** These work at the tracing layer but haven't been validated on the vLLM path. See `IDEAS.md` for the parity gap table.
- **No source tracing on fused CUDA kernels.** vLLM uses custom CUDA ops for attention and other hot paths; `.source` only works on Python-level forward methods.
- **Multi-tenant isolation is on you.** `Globals.saves` is process-global. For multi-user serving with isolation, use NDIF or build your own layer.
- **Version sensitivity.** Currently pinned to vLLM 0.15.1, Ray 2.53.0, grpcio 1.76.0. The Ray actor workaround is a vLLM-version-specific hack.
- **vLLM v1 only.** The integration targets vLLM's v1 architecture (the `AsyncLLM` import path is `vllm.v1.engine.async_llm`).
- **Multi-modal models are not yet integrated.** vLLM supports VLMs but the NNsight `VLLM` wrapper is text-only for now (`IDEAS.md`).

## Gotchas

- **Mode is set at construction time, not per-trace.** You can't switch between sync and async on the same `VLLM` instance. Construct with `mode="async"` if you want streaming.
- **`tracer.backend()` only exists in async mode.** In sync mode, results are pushed back into your local variables automatically when the trace block exits.
- **`model.logits` and `model.samples` are NNsight-specific eproperties** (`vllm.py:102-122`) — they don't exist on a vanilla `vllm.LLM`. Don't try to use them outside a trace.
- **Per-invoke kwargs override root kwargs.** Anything you pass to `tracer.invoke(prompt, temperature=...)` overrides what you passed to `model.trace(...)` for that invoke.
- **Empty invokes (`tracer.invoke()` with no args) work** — they see the full batch, useful for batch-wide observations.
- **Dispatching is automatic but takes a while.** First `.trace()` after `dispatch=False` triggers full vLLM engine init. Pass `dispatch=True` if you want that pause during construction.
- **`gpu_memory_utilization` defaults to 0.9.** For small models or shared GPUs, lower it explicitly. The test suite uses `0.1`.
- **CUDA graphs are not the only thing forbidden.** Speculative decoding, custom CUDA samplers, and certain attention backends may also break hooks. Stick with the default attention backend if interventions misbehave.
- **Async streaming intermediate saves are not collected in `__aiter__`.** As of `async_backend.py:79`, only `output.finished == True` outputs trigger `collect_nnsight`. If you want per-step saves on the final output, use `tracer.iter[:]` inside the trace block (saves accumulate in your list, then are returned at the end).

## Future work (NOT yet supported)

The vLLM integration's `IDEAS.md` lists features explicitly **not** implemented today:

- **Pipeline parallelism (PP > 1)** — needs per-stage mediator copies and rank-guarded interventions
- **Scan mode** — works at the tracing layer, hasn't been wired to vLLM
- **`tracer.cache()`** — same
- **Module renaming** — config forwarding only
- **Model editing (`model.edit()`)** — Envoy already wraps the model, but persistence isn't tested
- **Module skipping (`module.skip(...)`)** — needs testing with flat tensor format
- **Source tracing** — only works on Python forward methods, not fused kernels
- **Gradients / backward tracing** — would require backward in workers
- **Multi-modal vLLM** — vLLM has VLM support, NNsight doesn't expose it yet
- **Speculative decoding** — Eagle 3 etc. would need draft/verify phase boundaries
- **Online serving endpoint** — current integration is offline (`LLM`) only

If you need any of these, file an issue or read `IDEAS.md` for the design sketches.

## Related

- [docs/models/index.md](index.md) — pick the right wrapper
- [docs/models/language-model.md](language-model.md) — text-only HF alternative
- [docs/remote/](../remote/) — running traces on NDIF (an NDIF deployment may be vLLM-backed)
- `src/nnsight/modeling/vllm/README.md` — full architectural reference (file structure, key classes, execution flow, mediator transport, batch group management, multiple interleaving phases, tensor parallelism, continuous batching, multi-token generation, async engine, Ray executor, multi-node)
- `src/nnsight/modeling/vllm/DISCUSSION.md` — the philosophy: production-grade interpretability vs. SAE-based steering APIs
- `src/nnsight/modeling/vllm/IDEAS.md` — feature parity gaps and future directions (PP, multi-modal, speculative decoding, online serving)
- `src/nnsight/modeling/vllm/vllm.py` — `VLLM` class
- `src/nnsight/modeling/vllm/batching.py` — `VLLMBatcher` (TP gather/scatter)
- `src/nnsight/modeling/vllm/sampling.py` — `NNsightSamplingParams`
- `src/nnsight/modeling/vllm/async_backend.py` — `AsyncVLLMBackend`
- `tests/test_vllm.py` — runnable examples covering inference, generation, sampling, interventions, batching, TP, async streaming
- 0.6.0 release notes — vLLM is the headline feature of v0.6.0
