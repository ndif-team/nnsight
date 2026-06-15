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
- You need features vLLM doesn't fully support yet: gradients (no backward in workers), source tracing on fused CUDA kernels, model editing, or scan mode.
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
    pipeline_parallel_size=1,              # PP supported; see "Pipeline parallelism"
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
| `pipeline_parallel_size` | Number of pipeline stages. PP is **transparent** to your intervention code — cross-stage reads, writes, and saves work with single-GPU-style traces; composes with TP and Ray. Read [Pipeline parallelism](#pipeline-parallelism) for the execution semantics (your trace body runs once per stage). |
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
    hidden = model.transformer.h[-2].output.save()
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

Assign through `.output` to replace a value — do **not** write in place (`output[...] = ...`), which raises on vLLM's inference-mode tensors. For decoder layers (which return a `(stream, residual)` tuple), ablation, steering, logit lens, and patching, see [Intervention recipes](#intervention-recipes) below.

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
    hidden = model.model.layers[16].output.save()
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

Behind the scenes: `VLLM.trace()` injects `AsyncVLLMBackend` (`async_backend.py:19`), which submits the request to `AsyncLLM.generate()` and returns an async generator that yields `RequestOutput` objects. Saves are collected **only when `output.finished == True`** — at that point NNsight pulls them from workers via `collective_rpc("collect_nnsight", ...)` and attaches them as `output.saves`. Intermediate (non-final) outputs do **not** trigger save collection. See `async_backend.py:77`.

> If you need per-step saves at every yielded output, use `tracer.iter[:]` inside the trace block — they accumulate in your list and are returned together at the end. (A per-yield collection mode existed briefly during development and may return as an opt-in option in the future, but the current behavior is finished-only.)

#### Awaiting a single result (no streaming)

If you don't care about streaming and just want the final output, await the backend directly. `AsyncVLLMBackend.__await__` proxies the underlying generator (`async_backend.py:74`):

```python
async def main():
    with model.trace("Hello", max_tokens=3) as tracer:
        logits = model.logits.save()

    final_output = await tracer.backend()
    print(final_output.saves["logits"].shape)
```

For most use cases `async for` is more useful — it gives you per-step output. Use `await` when you only want the terminal `RequestOutput`.

#### Multi-prompt async streaming

Each `tracer.invoke(...)` is one vLLM request. With async, all requests are submitted at once and stream concurrently — vLLM's continuous batcher dynamically batches them on the GPU:

```python
prompts = ["The Eiffel Tower is in", "The Colosseum is in"]

async def main():
    with model.trace(max_tokens=5) as tracer:
        out_ids = [list() for _ in range(len(prompts))].save()
        for i, prompt in enumerate(prompts):
            with tracer.invoke(prompt):
                with tracer.all():
                    out_ids[i].append(model.samples.item())

    async for output in tracer.backend():
        print(f"req={output.request_id} finished={output.finished}")

asyncio.run(main())
```

Order of `output.request_id`s is **not** the order of your invokes — match by `request_id` if order matters.

#### Async gotchas

- `mode="async"` must be on the `VLLM(...)` constructor; it has no effect on `trace()`.
- `remote=True` is incompatible with `mode="async"` — `VLLM.trace` skips async-backend injection if `remote` is passed (`vllm.py:449`). NDIF currently runs the sync vLLM path only.
- The underlying generator is **single-shot**. Once iterated to completion, calling `tracer.backend()` again will not restart generation.
- `output.saves` only contains values on the **finished** output (`output.finished == True`). Intermediate outputs have an empty / sparsely-populated saves dict.

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
model.model.layers[i].self_attn.qkv_proj.output       # ColumnParallelLinear (merged Q,K,V)
model.model.layers[i].self_attn.o_proj.output         # RowParallelLinear
model.model.layers[i].mlp.gate_up_proj.output         # ColumnParallelLinear (merged gate,up)
model.model.layers[i].mlp.down_proj.output            # RowParallelLinear
model.model.norm.output
```

For GPT-2-style models in vLLM:

```python
model.transformer.h[i].attn.output
model.transformer.h[i].mlp.output
```

Print `model` to see the actual tree for your model.

#### What each module returns

vLLM's fused kernels, dual residual stream, and tensor-parallel layers change what a module returns relative to HuggingFace. Reading the wrong thing usually **runs without error but is silently wrong**, so check this before intervening:

| Access | Returns | Read it as |
|--------|---------|-----------|
| `model.model.layers[i].output` | `(sub_layer_output, residual)` 2-tuple — **not** a combined hidden state | full residual stream = `output[0] + output[1]` |
| `model.model.layers[i].input` | int64 **position IDs** (the first positional arg) | hidden states are `.inputs[0][1]` (args[1]); residual is `args[2]` |
| `model.model.norm.output` / any fused RMSNorm | `(normalized, residual)` 2-tuple | `norm.output[0]` |
| `o_proj` / `down_proj` `.output` (`RowParallelLinear`) | `(output, bias)` 2-tuple | `.output[0]` |
| `qkv_proj` / `gate_up_proj` `.output` (merged) | one concatenated tensor | `q, k, v = out.split([q_size, kv_size, kv_size], dim=-1)` / `gate, up = out.chunk(2, dim=-1)` — the separate `q_proj`/`gate_proj`/… don't exist |
| any activation | flat `[total_tokens, hidden]` — **no batch dimension** | select the last token with `[-1, :]`, not `[:, -1, :]` |

The dual residual stream is the most common trap: a decoder layer keeps `hidden_states` and `residual` as **separate** tensors (the next layer's fused `add_rms_norm` operates on their sum), so `layer.output[0]` alone is just one stream. Tensor parallelism stays transparent here — sub-module reads (`qkv_proj`, etc.) return the **full gathered** tensor at any `tensor_parallel_size`, not a shard.

## Intervention recipes

The vLLM-specific parts of every recipe below are the same two facts: outputs are read-only (clone-and-replace, never mutate in place) and decoder layers carry a `(stream, residual)` tuple. The research methodology is identical to `LanguageModel` — see the [patterns cookbook](../patterns/index.md).

### Writing activations — replace, don't mutate in place

vLLM executes inside `torch.inference_mode()`, so module outputs are **read-only**: an in-place write (`layer.output[0][:] = 0`, `+=`, …) raises `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`. Clone, mutate the clone, and assign the whole value back through `.output`:

```python
import torch

with model.trace("The Eiffel Tower is in the city of", temperature=0.0, top_p=1):
    # Ablate a decoder layer — zero BOTH streams (zeroing one leaves the residual)
    layer = model.model.layers[6]
    layer.output = (torch.zeros_like(layer.output[0]),
                    torch.zeros_like(layer.output[1]))

with model.trace("The Eiffel Tower is in the city of", temperature=0.0, top_p=1):
    # Steer the residual stream at the last token
    layer = model.model.layers[10]
    res = layer.output[1].clone()
    res[-1, :] += steering_vector            # [hidden], matching dtype/device
    layer.output = (layer.output[0], res)
```

### Logit lens

vLLM runs `lm_head` only on each request's **final token**, and `lm_head` is a `VocabParallelEmbedding` (an embedding table) — calling `model.lm_head(hidden)` does an index gather and crashes the engine. To read logits at an earlier layer or position, apply the final norm and unembed **inside the trace** with an explicit matmul against `lm_head.weight`:

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, top_p=1):
    layer = model.model.layers[6]
    hs = layer.output[0] + layer.output[1]            # combined residual stream
    normed = model.model.norm(hs)                      # final RMSNorm; a tensor when called with one arg
    if isinstance(normed, tuple):
        normed = normed[0]
    lens_logits = normed @ model.lm_head.weight.T      # [tokens, vocab] — do NOT call lm_head(...)
    top = lens_logits[-1].argmax(dim=-1).save()
```

Applied at the **last** layer this reproduces `model.logits` exactly. Reference `lm_head.weight` inside the matmul; don't `.save()` the parameter itself (saving the wrapped Parameter fails to serialize).

### Activation patching across prompts

vLLM's scheduler doesn't guarantee any execution order across invokes, so the robust way to copy an activation from one prompt into another is **two separate traces** — the second reuses the first's saved tensor:

```python
# 1. extract the clean activation
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, top_p=1):
    clean = (model.model.layers[5].output[0] + model.model.layers[5].output[1])[-1, :].save()

# 2. patch it into the other prompt (separate trace = deterministic ordering)
with model.trace("The Colosseum is in the city of", temperature=0.0, top_p=1):
    layer = model.model.layers[5]
    h0, h1 = layer.output[0].clone(), layer.output[1].clone()
    h0[-1, :] = clean.to(h0.device)        # put the clean vector in one stream...
    h1[-1, :] = 0                          # ...and zero the other so the sum equals `clean`
    layer.output = (h0, h1)
    logits = model.logits.save()
```

A shared trace-scope list that each invoke `.append`s to works fine within one trace; it's ordered *value passing between* invokes that two traces make reliable.

### Activation caching with `tracer.cache()`

`tracer.cache()` works on the vLLM path — it hooks each target module and collects activations into a `CacheDict` keyed by Envoy path:

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=8) as tracer:
    cache = tracer.cache(modules=[model.model.layers[6]]).save()

entry = cache["model.model.layers.6"]
```

A module that fires on prefill **and** each decode step yields a **list** of `Entry` objects; sum the dual streams (`e.output[0] + e.output[1]`) and concatenate across entries for every captured token. `include_inputs=`, `device=`, and `dtype=` work as in [cache](../usage/cache.md).

## Pipeline parallelism

`pipeline_parallel_size > 1` is supported and **transparent**: cross-stage reads, writes, and saves work with single-GPU-style trace code, across single/multi-token generation, batching, sync/async, the serve path, TP, and multi-node Ray. You never write rank-aware code:

```python
model = VLLM("meta-llama/Llama-3.1-70B", tensor_parallel_size=4, pipeline_parallel_size=2)

with model.trace("Hello") as tracer:
    h5 = model.model.layers[5].output[0]            # produced on stage 0
    model.model.layers[60].mlp.output = h5 * 2      # consumed + written on stage 1
    logits = model.logits.save()                    # produced on stage 1
```

How it works (full design: `docs/developing/pp-design.md`): accesses to modules on another stage return a lazy placeholder that materializes — a cross-rank pull — only when genuinely consumed; writes and saves of remote values are local no-ops because the owning stage performs the real ones.

### Execution semantics — the fine print

**Your trace body runs once per PP stage.** Each stage executes the same Python independently; nnsight reconciles the results (writes/saves take effect only on the stage that owns the module; saved values are merged across stages). Tensor results are deterministic, but three things follow from per-stage execution:

- **Side effects run once per stage.** A `print`, a file append, an external API call inside the trace executes `pipeline_parallel_size` times, in different processes (possibly on different machines under Ray).
- **Per-stage environment values diverge.** `os.getpid()`, `time.time()`, device queries, hostnames — each stage sees its own. If you *save* such a value, the result is one stage's copy.
- **In-trace randomness: generate it OUTSIDE the trace.** A tensor created before `with model.trace(...)` is serialized with the intervention and arrives **identical on every stage** — this is the supported way to use noise (and what paired-comparison methods like causal-tracing corruption want anyway: the same noise across runs). `torch.randn` *inside* the trace happens to agree across stages today — vLLM seeds every worker identically and the stages draw in lockstep — but this is incidental, not contractual; any asymmetric consumption of the generator desyncs it. Treat in-trace RNG as unspecified under PP.

**The divergence tripwire.** If two stages ship *different* values for the same saved variable, the merge emits a `PPRankDivergenceWarning` naming the slot (e.g. `saved slot 'noise' (max|Δ| = 1.7)`) instead of silently keeping an arbitrary copy. If you see it, the saved value is not trustworthy — hoist the offending computation out of the trace. Identical redundant copies (the normal case) merge silently; float comparison uses a tight tolerance so low-order kernel noise never warns.

**Access modules in forward-pass order.** The base nnsight rule — access modules in the order the forward fires them within one invoke — extends *across stages* under PP: read earlier-stage modules before later-stage ones, and `model.logits` / `model.samples` last. Reading a later-stage module before an earlier-stage one in the same iteration (e.g. `model.logits` *then* a stage-0 layer inside a `tracer.iter` loop) raises `OutOfOrderError` — the same error single-GPU gives for out-of-order access. Reorder the accesses (earlier-stage first), or split them across separate invokes.

PP performance characteristics are in `docs/developing/pp-design.md` §7 — notably, PP=2 plain generation is *faster* than PP=1 under `enforce_eager`, and cross-stage reads cost ~3–7 ms per pulled value.

## Limitations

- **`enforce_eager=True` is forced.** vLLM's CUDA graph optimization is incompatible with arbitrary PyTorch hooks. This costs you some throughput on decode-heavy workloads (see `DISCUSSION.md` for context).
- **PP runs your trace body once per stage.** Side effects and per-stage environment values are per-rank; in-trace RNG is unspecified (hoist it). See [Pipeline parallelism](#pipeline-parallelism).
- **One prompt per invoke.** Unlike `LanguageModel`, you cannot pass `tracer.invoke(["a", "b"])`. Each invoke = one vLLM request. Use a loop of invokes for multiple prompts (`vllm.py:267`).
- **No backward / gradients.** Backward tracing is not supported in vLLM workers (`IDEAS.md`).
- **No `.scan()` or module editing yet.** These work at the tracing layer but haven't been validated on the vLLM path. See `IDEAS.md` for the parity gap table. (`tracer.cache()` **is** supported — see [Intervention recipes](#intervention-recipes).)
- **No source tracing on fused CUDA kernels.** vLLM uses custom CUDA ops for attention and other hot paths; `.source` only works on Python-level forward methods.
- **Multi-tenant isolation is on you.** `Globals.saves` is process-global. For multi-user serving with isolation, use NDIF or build your own layer.
- **Version sensitivity.** Currently developed against vLLM 0.19.1. The Ray actor workaround is a vLLM-version-specific hack.
- **vLLM v1 only.** The integration targets vLLM's v1 architecture (the `AsyncLLM` import path is `vllm.v1.engine.async_llm`).
- **Multi-modal models are not yet integrated.** vLLM supports VLMs but the NNsight `VLLM` wrapper is text-only for now (`IDEAS.md`).

## Gotchas

- **Scripts must use an `if __name__ == "__main__":` guard.** vLLM uses `spawn` multiprocessing for its EngineCore subprocess (CUDA contexts can't be safely forked), and `spawn` re-imports your main module in the child. Without the guard, the child re-runs the top-level `VLLM(...)` / `.trace(...)` calls and tries to spawn another EngineCore — Python's `_check_not_importing_main` then raises `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`. The fix is the standard idiom:

  ```python
  from nnsight.modeling.vllm import VLLM

  def main():
      model = VLLM("gpt2")
      with model.trace("hello", max_tokens=1):
          out = model.transformer.h[0].output.save()
      print(out.shape)

  if __name__ == "__main__":
      main()
  ```

  This is a vLLM / Python multiprocessing requirement, not nnsight-specific — raw `vllm.LLM(...)` at module level has the same constraint. Notebooks (Jupyter / Colab) are fine because they don't re-import.

- **Mode is set at construction time, not per-trace.** You can't switch between sync and async on the same `VLLM` instance. Construct with `mode="async"` if you want streaming.
- **`tracer.backend()` only exists in async mode.** In sync mode, results are pushed back into your local variables automatically when the trace block exits.
- **`model.logits` and `model.samples` are NNsight-specific eproperties** (`vllm.py:102-122`) — they don't exist on a vanilla `vllm.LLM`. Don't try to use them outside a trace.
- **Per-invoke kwargs override root kwargs.** Anything you pass to `tracer.invoke(prompt, temperature=...)` overrides what you passed to `model.trace(...)` for that invoke.
- **Empty invokes (`tracer.invoke()` with no args) work** — they see the full batch, useful for batch-wide observations.
- **Dispatching is automatic but takes a while.** First `.trace()` after `dispatch=False` triggers full vLLM engine init. Pass `dispatch=True` if you want that pause during construction.
- **`gpu_memory_utilization` defaults to 0.9.** For small models or shared GPUs, lower it explicitly. The test suite uses `0.1`.
- **CUDA graphs are not the only thing forbidden.** Speculative decoding, custom CUDA samplers, and certain attention backends may also break hooks. Stick with the default attention backend if interventions misbehave.
- **Async streaming saves are collected only when `output.finished == True`.** As of `async_backend.py:77`, `__aiter__` calls `collect_nnsight` on the final output only — intermediate yields don't trigger collection. If you want per-step saves, accumulate them inside `tracer.iter[:]` (saves end up on the final output's `.saves`).
- **Outputs are read-only — replace, don't mutate.** In-place writes (`layer.output[0][:] = ...`) raise `RuntimeError: Inplace update to inference tensor...`. Clone and assign through `.output`. See [Intervention recipes](#intervention-recipes).
- **`.save()` is mutation-safe.** vLLM's fused kernels mutate buffers in place after hooks fire; `.save()` auto-clones inference-mode tensors so the value you keep isn't corrupted by later ops (`intervention/tracing/globals.py`). No-op on the HF path.
- **Prefix caching is off by default.** `VLLM(...)` sets `enable_prefix_caching=False` (`vllm.py`) so interventions see every token — with it on, tokens served from a cached prefix skip the forward pass and your hooks silently don't fire. Only pass `enable_prefix_caching=True` if you don't hook prefill tokens.
- **Errors don't kill the engine.** An exception inside a trace (e.g. a bad layer index) is deferred and surfaced at the trace boundary; the engine survives and subsequent traces work. `tracer.stop()` similarly only stops the calling invoke.
- **No attention weights.** PagedAttention runs in C/CUDA, so there's no Python-level attention-weight tensor — attention-pattern / head-knockout patterns aren't available (use `LanguageModel`).
- **vLLM ≠ transformers numerically.** Fused kernels, quantization defaults, and batching make outputs differ from `LanguageModel` for the same input. Compare intervention *effects* (deltas from a baseline), not absolute values.

## Future work (NOT yet supported)

The vLLM integration's `IDEAS.md` lists features explicitly **not** implemented today:

- **Scan mode** — works at the tracing layer, hasn't been wired to vLLM
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

### Demo repositories

- [**nnsight-vllm-demos**](https://github.com/ndif-team/nnsight-vllm-demos) — runnable end-to-end demos including an async chat interface with SAE-based steering. Linked from the v0.6.0 release notes.
- [**nnsight-vllm-lens-comparison**](https://github.com/ndif-team/nnsight-vllm-lens-comparison) — comparison of logit-lens / tuned-lens style probes running on the NNsight vLLM integration.

### Docs and source

- [docs/models/index.md](index.md) — pick the right wrapper
- [docs/models/language-model.md](language-model.md) — text-only HF alternative
- [docs/remote/](../remote/) — running traces on NDIF (an NDIF deployment may be vLLM-backed)
- `src/nnsight/modeling/vllm/README.md` — full architectural reference (file structure, key classes, execution flow, mediator transport, batch group management, multiple interleaving phases, tensor parallelism, continuous batching, multi-token generation, async engine, Ray executor, multi-node)
- `src/nnsight/modeling/vllm/DISCUSSION.md` — the philosophy: production-grade interpretability vs. SAE-based steering APIs
- `src/nnsight/modeling/vllm/IDEAS.md` — feature parity gaps and future directions (multi-modal, speculative decoding, online serving)
- `src/nnsight/modeling/vllm/vllm.py` — `VLLM` class
- `src/nnsight/modeling/vllm/batching.py` — `VLLMBatcher` (TP gather/scatter)
- `src/nnsight/modeling/vllm/sampling.py` — `NNsightSamplingParams`
- `src/nnsight/modeling/vllm/async_backend.py` — `AsyncVLLMBackend`
- `tests/test_vllm.py` — runnable examples covering inference, generation, sampling, interventions, batching, TP, async streaming
- [v0.6.0 release notes](../../0.6.0.md) — vLLM is the headline feature of v0.6.0
