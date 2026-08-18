---
title: VLLM
one_liner: High-throughput vLLM serving with NNsight interventions; sync or async, tensor parallelism, continuous batching, streaming.
tags: [models, vllm, serving, production]
related: [docs/models/index.md, docs/models/transformers-model.md, docs/remote/index.md]
sources: [src/nnsight/modeling/vllm/vllm.py:36, src/nnsight/modeling/vllm/async_backend.py:40, src/nnsight/modeling/vllm/batching.py, src/nnsight/modeling/vllm/serve/backend.py, tests/vllm/]
---

# VLLM

## What this is for

`nnsight.modeling.vllm.VLLM` runs NNsight interventions on top of vLLM's inference engine. The model runs in vLLM's own worker process(es); your intervention code is serialized onto each request (via `SamplingParams.extra_args`), carried into the worker, run against the real module, and the saved values shipped back. You get PagedAttention, continuous batching, tensor parallelism, and optional async streaming, with arbitrary Python interventions inline with the forward.

Interventions are written exactly as for any other model — the module tree mirrors what vLLM loaded. The one structural difference: each `tracer.invoke(...)` is **one vLLM request** (one prompt), so several prompts means several invokes, not a list.

> **Running this** needs `vllm` installed and a GPU. The examples below are drawn from the source and `tests/vllm/` (which skip without a GPU).

## When to use / when not to use

Use `VLLM` when:
- You need **throughput** / continuous batching for many concurrent requests.
- You need **tensor parallelism** across GPUs.
- You want **async token-by-token streaming** with interventions.

Do not use `VLLM` when:
- You have a single prompt and don't need throughput — [`TransformersModel`](transformers-model.md) is simpler.
- You need gradients/backward, `.scan()`, or source tracing on fused CUDA kernels — not supported on the vLLM path.
- You're doing diffusion or VLM work — the NNsight vLLM integration is text-only.

## Loading

```python
from nnsight.modeling.vllm import VLLM

model = VLLM(
    "gpt2",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.1,
    dispatch=True,
)
```

### Constructor

```python
VLLM(
    repo_id,
    *,
    mode="sync",                  # "sync" (vllm.LLM) or "async" (AsyncLLM)
    dispatch=False,               # True = build the engine now
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    **vllm_kwargs,                # forwarded to vllm.LLM / AsyncEngineArgs
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | HuggingFace repo id. |
| `mode` | `"sync"` (default) builds a `vllm.LLM` and runs synchronous generation; `"async"` builds a `vllm.v1.engine.async_llm.AsyncLLM` and a trace streams outputs via `async for output in tracer.backend`. Set at construction — you can't switch per trace (`vllm.py:57`). |
| `dispatch` | `True` creates the real engine during `__init__`. `False` (default) builds only a meta-tensor tree (via vLLM's `DummyModelLoader`, `device="meta"`) — no GPU memory used until the first trace (`vllm.py:155`). |
| `tensor_parallel_size` | GPUs to shard across. Transparent to your interventions (see [Tensor parallelism](#tensor-parallelism-is-transparent)). |
| `gpu_memory_utilization` | vLLM KV-cache budget (default 0.9). Lower it (e.g. 0.1) for small models / shared GPUs. |
| `**vllm_kwargs` | Anything valid for `vllm.LLM` / `AsyncEngineArgs`. |

`enforce_eager=True` and the NNsight worker class are always forced internally (`vllm.py:205`, `:189`) — CUDA graphs freeze the ops they replay, so hooks can't fire inside one.

## Canonical pattern (sync)

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)

with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, top_p=1):
    model.transformer.h[8].output[:] = 0        # intervene
    logits = model.logits.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### `logits` and `samples`

These are vLLM-specific hookable values on the model — `eproperty` descriptors (the same mechanism behind a module's `.output`/`.input`, `vllm.py:144`), not on a vanilla `vllm.LLM`, and only meaningful inside a trace. Read them for the logits and sampled ids of each step; for the finished request as a whole, read `tracer.result`.

| Property | Description |
|----------|-------------|
| `model.logits` | The pre-sampling logit tensor for this step. Read or assign (`model.logits = ...`). |
| `model.samples` | The token ids the sampler drew from `logits` this step. Assigning it forces the token the engine continues from. |

Under `tracer.iter`, each pass sees the next decoded step.

### Multi-token generation

```python
with model.trace("Madison Square Garden is located in the city of",
                 temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
    logits = list().save()
    for _ in tracer.iter[0:3]:
        logits.append(model.logits)

print(model.tokenizer.batch_decode([l.argmax(dim=-1) for l in logits]))
# [' New', ' York', ' City']
```

`tracer.all()` iterates every generated step:

```python
with model.trace(PROMPT, max_tokens=10) as tracer:
    logits = list().save()
    for _ in tracer.all():
        logits.append(model.logits)
# len(logits) == 10
```

### `generate` traces, or just runs

vLLM generation is driven by `max_tokens`, so there is no forward/generate split. Used as a `with` block, `model.generate(...)` is `trace` (and rewrites `max_new_tokens` → `max_tokens`; `trace` accepts either).

Called **without** a `with` block it simply runs the engine and returns vLLM's `RequestOutput`s — which is how you read an edit's values without reaching past the model for `model.vllm_entrypoint`:

```python
outputs = model.generate(prompts, max_tokens=5)
outputs[3].saves["hidden"]        # see Editing the engine, below
```

On `mode="async"` the same call returns an awaitable: `outputs = await model.generate(...)`.

### `tracer.result`

`tracer.result` is the finished `RequestOutput` for the request an invoke made — one per invoke:

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=3) as tracer:
    hidden = model.model.layers[8].output[0].save()
    result = tracer.result.save()

result.outputs[0].text        # ' Paris, France'
```

It is served at collect time, which is the first moment both halves exist — the block runs in the engine's worker, the output is assembled by the engine. Two consequences: it carries the generation but **not** `.saves` (those are attached to the engine's own copy afterwards), and per-step values still come from `model.logits` / `model.samples` under `tracer.iter`, since `result` is the request's end state.

### Sampling parameters

Sampling kwargs go to `trace`/`invoke` (not configured on the model) and become the request's `SamplingParams`:

```python
# root-level params apply to all invokes
with model.trace("Hello", temperature=0.7, top_p=0.95, max_tokens=10) as tracer:
    samples = list().save()
    for _ in tracer.iter[:]:
        samples.append(model.samples.item())

# per-invoke params override root
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.0, top_p=1.0):
        ...
    with tracer.invoke("Hello", temperature=1.5, top_p=0.95):
        ...
```

Common kwargs: `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `stop`, `stop_token_ids`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logprobs`, `lora_request` — anything `vllm.SamplingParams` takes.

### Input forms per invoke

One prompt per invoke: a string, a list of token ids, a tokenizer's `{input_ids, attention_mask}` output, or one of vLLM's own prompt dicts (`TokensPrompt`, `TextPrompt`). A list of strings / multiple prompts is rejected — use one invoke each.

### Continuous batching (multiple invokes)

Each invoke is one request; vLLM's continuous batcher processes them together.
**Collect each invoke's values into its own saved variable** — a container declared
outside the invokes and appended inside them does *not* merge back, because each
invoke's intervention is serialized into its own request separately (unlike the
in-process local path):

```python
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        paris = list().save()                 # this invoke's own saved list
        for _ in tracer.all():
            paris.append(model.samples.item())
    with tracer.invoke("The capital of Japan is"):
        tokyo = list().save()                 # a distinct saved list
        for _ in tracer.all():
            tokyo.append(model.samples.item())

print(model.tokenizer.decode(paris), model.tokenizer.decode(tokyo))
```

For a **dynamic** number of prompts you can't give each invoke a distinct name in one
trace, and a shared container won't merge. Instead fire each prompt as its **own**
async trace concurrently (`asyncio.gather`) — the engine still batches the concurrent
requests, and each one's saves arrive on its own finished `output`.

### Tensor parallelism is transparent

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=4, dispatch=True)

with model.trace("Hello", temperature=0.0):
    hidden = model.model.layers[16].output.save()   # full unsharded tensor
```

`VLLMBatcher` (`batching.py`) gathers a `ColumnParallelLinear`/`RowParallelLinear` shard into the full tensor before your intervention reads it and re-splits on write, so every rank runs the same code on the same complete tensor. Verified in `tests/vllm/test_tensor_parallel.py`.

### Mixture-of-experts models and expert parallelism

MoE models (Qwen-MoE, DeepSeek, Mixtral, ...) work with the same transparency, in both expert layouts vLLM offers on the same ranks:

- **default** (`enable_expert_parallel=False`): every rank holds a slice of every expert's matrices (the dense-MLP TP sharding, fused across experts);
- **expert parallel** (`enable_expert_parallel=True`): each rank holds `num_experts / world_size` whole experts.

The router (`mlp.gate`, a `ReplicatedLinear`) is full and identical on every rank, so reading router logits or swapping them (expert steering / expert-masking ablation) needs no gathering at all. The fused-experts module (`mlp.experts`, a `FusedMoE`) is the one MoE-specific case the batcher handles: models that build it with `reduce_results=False` (the Qwen-MoE/DeepSeek pattern) make it return **per-rank partial sums** that the outer block all-reduces afterwards, so on access the batcher all-reduces the partials into the true value and on write-back divides by the group size so the block's own all-reduce reconstructs a swapped value exactly once. Verified in `tests/vllm/test_moe_batching.py` against an HF reference.

Individual experts are **not** addressable as submodules: vLLM stacks all local experts into fused weight tensors consumed by one grouped kernel, so there is no `experts[3]` to hook at any parallelism level. To ablate an expert, mask its router logit to `-inf` in `mlp.gate.output` instead.

## Async mode

Construct with `mode="async"`; a trace then streams `RequestOutput`s. Iterate `tracer.backend` (an attribute, not a call):

```python
import asyncio
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True, mode="async")

async def main():
    with model.trace("The Eiffel Tower is located in the city of",
                     temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.save()

    async for output in tracer.backend:
        print(output.finished, output.outputs[0].text)
        if output.finished:
            print("saves:", list(output.saves.keys()))

asyncio.run(main())
```

Saves are attached **only to the finished output** (`output.finished == True`), fetched from the worker via `collect_nnsight` at that point (`async_backend.py:120`). Intermediate yields carry no saves — accumulate per-step values inside `tracer.iter[:]` instead.

Await the backend to drain the stream and get just the last output:

```python
last = await tracer.backend
print(last.saves["logits"].shape)
```

### Async notes

- Async tracing takes a **single prompt** (one invoke or a direct input) — several invokes raise `NotImplementedError` (`async_backend.py:64`).
- The stream is single-shot; once drained it won't restart.
- A stream closed before it finishes aborts the request and frees its worker (`async_backend.py:91`).
- Errors in the block surface when you iterate the stream (a `1/0` raises `RuntimeError: ...ZeroDivisionError`).
- `remote=True` skips async-backend injection (`vllm.py:347`).

## Editing the engine

A trace carries its block on the request it rides. That is the right shape for
"run this one experiment", but it means a sweep serializes the same block once
per prompt, and it can only touch requests that *are* nnsight traces.

`model.edit()` sends the block over once and leaves it there. Every request
the engine runs afterwards gets its own copy — including requests submitted by
something that never heard of nnsight, e.g. an OpenAI-API client on the same
server. Each copy has its own scope, so what it saves is that request's, and it
comes back on that request's output — the same place a trace's values arrive.

```python
model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True, enable_prefix_caching=False)

with model.edit() as (tracer, edit):
    hidden = model.model.layers[16].output[0].save()

# Not traces — plain vLLM requests. The block still runs for them.
outputs = model.generate(["The Eiffel Tower is in", "The capital of Japan is"],
                         max_tokens=5)

outputs[1].saves["hidden"]        # prompt 1's activations
edit.clear()
```

There is no id to join on: the value is on the output of the request that
produced it. For a *traced* request, reach it through `tracer.result.saves`.

The block is written exactly like a trace body — same envoy tree, same `.save()`.
It belongs to no particular request, so there is **no `tracer.invoke(...)`**. The
tracer is bound alongside the handle (as `Envoy.edit()` binds `(tracer, edited)`)
because `tracer.iter` / `tracer.all()` is what lets an installed block follow a
request across its generated tokens rather than seeing only the prefill:

```python
with model.edit() as (tracer, edit):
    readout = nnsight.save([])
    for step in tracer.all():
        readout.append(model.model.layers[16].output[0][-1])
```

| Member | Description |
|---|---|
| `edit.clear()` | Stop running the block. `await edit.aclear()` on an async engine. |
| `model.clear_edits()` | Clear every edit still installed. `await model.aclear_edits()` on an async engine. |

That is the whole handle — the values are not read through it. They are taken as
they are collected, so nothing accumulates on the worker for as long as somebody
is reading the outputs, which on the synchronous engine is every request there
is. An error raised inside an installed block is re-raised where its values would
have arrived.

If an edit saves under the same name as the trace reading it, `output.saves`
keeps the trace's — but the edit's own is still there under
`output.nnsight_saves`. Different names avoid the question.

### On an async engine

Installing the block is a `collective_rpc`, which on `mode="async"` can only be
awaited from inside the running loop — so use `async with`, and `aclear`:

```python
async with model.edit() as (tracer, edit):
    hidden = model.model.layers[16].output[0].save()

outputs = await model.generate(prompts, max_tokens=5)
outputs[1].saves["hidden"]

await edit.aclear()
```

A plain `with` on an async engine raises rather than silently not installing it,
and so does `clear_edits()` — a coroutine nobody awaits never runs, so a
sync-looking call would leave every edit in place and say nothing.

### When to edit instead of trace

- Sweeping many prompts — an edit pays the serialization once instead of per
  request. Capturing one layer over 1024 prompts on Llama-3.1-8B: **2.04 s
  traced, 1.43 s edited** (bare vLLM 0.87 s).
- Instrumenting traffic you don't control (a served endpoint, another client).
- Keep tracing for one-off experiments, and whenever you want the values pushed
  back into your own variables.

> **Prefix caching must be off.** A prefix-cached token is served from the KV
> cache without a forward pass, so no hook fires and an installed block sees a
> short activation with no error. A trace asks for its own request to be
> recomputed; an edit rides requests it did not create and cannot. Build with
> `enable_prefix_caching=False` — editing an engine that has it on warns.

## Remote / serve

- `trace(..., remote=True)` runs on NDIF. The model key is the repo id (`vllm.py:449`).
- `trace(..., serve=url, api_key=...)` runs the trace on a standalone **nnsight-serve** engine: the block is written against a **GPU-less** meta model, serialized like the NDIF path, sent to the server, and its saved values pushed back into your frame — so reading a `.save()`d variable after the block works exactly as locally.

Start a server (holds one dispatched async engine):

```bash
nnsight-serve gpt2 --port 8000 [--api-key SECRET] [--gpu-memory-utilization 0.1]
```

Then a client with **no GPU** submits traces to it:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2")                       # meta tree only, never dispatched
with model.trace("The Eiffel Tower is in", serve="http://127.0.0.1:8000", api_key="SECRET"):
    logits = model.logits.save()
print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

The server returns saved values only — save `tracer.result` to get the finished `RequestOutput` back with them. Build and runtime errors come back with their real type and traceback. See `src/nnsight/modeling/vllm/serve/`.

`model.edit(serve=url, api_key=...)` installs a block on the server's engine the same way, over `POST /v1/nnsight/register/{id}`:

```python
with model.edit(serve="http://127.0.0.1:8000") as (tracer, edit):
    hidden = model.transformer.h[5].output.save()

with model.trace("The Eiffel Tower is in", serve="http://127.0.0.1:8000") as tracer:
    result = tracer.result.save()
result.saves["hidden"]        # the edit's value, on the request it ran on

edit.clear()
```

## Special properties

| Attribute | Description | Source |
|-----------|-------------|--------|
| `model.logits` / `model.samples` | Pre-sampling logits / sampled ids for the step (trace-only, `eproperty`). | `vllm.py:144` |
| `model.tokenizer` | The tokenizer vLLM resolved for the checkpoint. Loaded eagerly. | `vllm.py:181` |
| `model.vllm_entrypoint` | The underlying `vllm.LLM` (sync) or `AsyncLLM` (async); `None` until dispatch. | `vllm.py:58` |
| `model.dispatched` | Whether the engine is built. | `MetaMixin` |

### Module structure

The Envoy tree mirrors vLLM's model layout. For Llama-style models:

```python
model.model.layers[i].self_attn.qkv_proj.output   # ColumnParallelLinear
model.model.layers[i].self_attn.o_proj.output     # RowParallelLinear
model.model.layers[i].mlp.gate_up_proj.output
model.model.layers[i].mlp.down_proj.output
model.model.norm.output
```

For GPT-2-style models: `model.transformer.h[i].attn.output`, `model.transformer.h[i].mlp.output`. Print `model` to see the actual tree.

## Limitations

- **`enforce_eager=True` is forced** — costs some decode throughput, required for hooks.
- **One prompt per invoke** — no `tracer.invoke(["a", "b"])`.
- **No `tracer.barrier(n)`** — each invoke is its own request and the engine schedules them independently, so the blocks never run against the same forward. Calling it raises rather than hanging.
- **No backward / gradients, no `.scan()`, no source tracing on fused CUDA kernels.**
- **Text-only** — multimodal vLLM is not exposed.
- **Version sensitivity** — targets vLLM's v1 architecture (`AsyncLLM` at `vllm.v1.engine.async_llm`).

## Gotchas

- **Scripts need an `if __name__ == "__main__":` guard.** vLLM uses `spawn` for its EngineCore subprocess; without the guard the child re-runs your top-level `VLLM(...)` / `.trace(...)` and raises a bootstrapping `RuntimeError`. Notebooks are fine. This is a vLLM/multiprocessing requirement, not nnsight-specific.

  ```python
  from nnsight.modeling.vllm import VLLM

  def main():
      model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)
      with model.trace("hello", max_tokens=1):
          out = model.transformer.h[0].output.save()
      print(out.shape)

  if __name__ == "__main__":
      main()
  ```

- **A saved value comes back by its variable *name* — bind it.** `logits = model.logits.save()` works; a bare `model.logits.save()` marks the value but has no name to return it under, so `output.saves["logits"]` (async/serve) or the pushed-back local is silently missing. This is the most common "saves don't come back" bug on vLLM.
- **Cross-invoke shared state does not merge.** A container declared outside the invokes and appended inside each one is serialized per request, so the appends don't come back — save per invoke (see [Continuous batching](#continuous-batching-multiple-invokes)).
- **`tracer.result` is the finished `RequestOutput`, not per-step.** It carries the generation but not `.saves`; use `model.logits` / `model.samples` under `tracer.iter` for per-step values.
- **An empty `tracer.invoke()` with interventions raises** (its work would vanish); a do-nothing empty invoke is a harmless no-op.
- **A typo'd sampling kwarg raises** (`trace(temperatur=0.0)` → `TypeError`), rather than being silently ignored.
- **Mode is fixed at construction.** Build with `mode="async"` if you want streaming.
- **`tracer.backend` is iterable only in async mode.** In sync mode, results land in your local variables when the block exits.
- **`logits` / `samples` don't exist on a vanilla `vllm.LLM`** — use them only inside a trace.
- **Per-invoke kwargs override root kwargs.**
- **`gpu_memory_utilization` defaults to 0.9** — lower it for small/shared GPUs (tests use 0.1).
- **Async saves are finished-only** — for per-step saves, accumulate inside `tracer.iter[:]`.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/transformers-model.md](transformers-model.md) — the text-only HF alternative
- [docs/remote/](../remote/) — running traces on NDIF
- `src/nnsight/modeling/vllm/vllm.py` — `VLLM` class
- `src/nnsight/modeling/vllm/batching.py` — `VLLMBatcher` (TP gather/scatter)
- `src/nnsight/modeling/vllm/async_backend.py` — `AsyncVLLMBackend`
- `src/nnsight/modeling/vllm/serve/` — nnsight-serve backend / server / CLI
- `tests/vllm/` — runnable examples (tracing, async, tensor parallelism, mixture-of-experts, requests, serve)
