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
    taps=(),                      # locations to serve under CUDA graphs; empty = eager, every location
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
| `taps` | Module locations to serve with CUDA graphs **on** — `"model.layers.*.output"`, `*` one path segment. Empty (default) runs the engine eagerly, where every location is served. See [CUDA graphs with taps](#cuda-graphs-with-taps). |
| `**vllm_kwargs` | Anything valid for `vllm.LLM` / `AsyncEngineArgs`. |

The NNsight worker class is always forced internally, and so is `enforce_eager=True` unless you declare `taps` — CUDA graphs freeze the ops they replay, so an ordinary hook can't fire inside one.

## Canonical pattern (sync)

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)

with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, top_p=1):
    model.transformer.h[8].output[:] = 0        # intervene
    logits = model.logits.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

### What your block sees on vLLM

Four things differ from the HuggingFace path, and every one of them changes what an index means:

- **No batch axis.** A served value is `[tokens, hidden]` for *your* request's rows: the prefill step
  serves every prompt token, each later decode step serves one row. The last position is `[-1]`,
  never `[:, -1, :]`.
- **A decoder layer's `.output` is a pair `(hidden, residual)`** on Llama/Qwen/Mistral-style models,
  because vLLM fuses the residual add into the next layer's norm. `output[0]` is this layer's
  sub-block output (norm ~30 on Qwen3-8B), `output[1]` the residual stream entering it (norm ~1000);
  the residual stream *after* the layer is their sum. Patching `output[0]` alone changes little;
  writing either element steers, since the next norm adds them.

  ```python
  out = model.model.layers[20].output
  resid = (out[0] + out[1]).clone()        # residual stream after layer 20, [tokens, hidden]
  out[0][-1] += v                          # additive steering at the last position
  ```

- **Clone what you keep.** A served value is the model's live buffer, and the next layer's fused
  add+norm rewrites it in place — `layers[8].output[0].save()` comes back holding a later layer's
  data. Reduce or `.clone()` before you save (`(out[0] + out[1]).mean(0).cpu().save()`).
- **Where tensors live.** A tensor your block references from outside (a steering vector) is
  serialized with the block and arrives in the worker as it was; move it onto the served value
  (`v.to(h.device, h.dtype)`). Saved tensors come back on the worker's device (`cuda:0`) unless
  you `.cpu()` them in the block. `model.device` on an undispatched client is `meta`.

`model.logits` is `[1, vocab]` for the step and `model.samples` is `[1, 1]` (`.item()` for one
sequence). Greedy decoding is `temperature=0.0`; vLLM's default is `1.0`.

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

Step 0 is the prefill; `model.samples` on step *k* is the *k*-th generated token
(`result.outputs[0].token_ids[k]`). Plain Python locals persist across steps, so a block can keep
state — a flag flipped by a probe at one step steers every later one. A write to a layer *below*
a read lands on the next step, not the current one.

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
    out = model.model.layers[8].output
    hidden = (out[0] + out[1]).clone().save()
    result = tracer.result.save()          # last: nothing can be read after it

result.outputs[0].text        # ' Paris, France'
```

It is served at collect time — after every module, `logits` and `samples` visit of the request —
so it must be the **last** read in the block; a read after it raises `OutOfOrderError` naming
that later value. It is served at collect time, which is the first moment both halves exist — the block runs in the engine's worker, the output is assembled by the engine. Two consequences: it carries the generation but **not** `.saves` (those are attached to the engine's own copy afterwards), and per-step values still come from `model.logits` / `model.samples` under `tracer.iter`, since `result` is the request's end state.

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

Common kwargs: `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `stop`, `stop_token_ids`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logprobs`, `lora_request` — anything `vllm.SamplingParams` takes. `ignore_eos=True` makes a
bounded `tracer.iter[:N]` see all `N` steps when the model would stop early.

### Input forms per invoke

One prompt per invoke: a string, a list of token ids, a tokenizer's `{input_ids, attention_mask}` output, or one of vLLM's own prompt dicts (`TokensPrompt`, `TextPrompt`). A list of strings / multiple prompts is rejected — use one invoke each.

### Continuous batching (multiple invokes)

Each invoke is one request; vLLM's continuous batcher processes them together.
Your block runs once per request, so a name saved in every invoke comes back as a
**list**, one entry per invoke in order:

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in prompts:
        with tracer.invoke(prompt):
            hidden = model.transformer.h[5].output.save()

hidden[0]        # the first invoke's, sized to its own prompt
len(hidden)      # == len(prompts)
```

A name saved by exactly one invoke is that value, not a list of one — so giving
each invoke a distinct name works too, and reads as it always did:

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

For a **dynamic** number of prompts, save one name in every invoke and read the
list — that is what the loop above does. Firing each prompt as its own async trace
concurrently (`asyncio.gather`) also works, and each one's saves then arrive on its
own finished `output`.

A container bound *and saved* **above** the invokes is one object locally, and
each request gets its own copy. Assigning into pre-sized slots merges back;
`append` does not — a list appended in every invoke comes back with one element.
For a dynamic number of prompts, save the same name in every invoke and read the
list.

```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    rows = nnsight.save([None, None])
    with tracer.invoke(prompt_a):
        rows[0] = model.transformer.h[5].output
    with tracer.invoke(prompt_b):
        rows[1] = model.transformer.h[5].output
```

### Passing values between invokes

Each invoke's block runs on its own request, in its own scope: a value one invoke
reads is **not** visible to a sibling (`NameError`), `tracer.barrier` is
unavailable, and a `session()` does not bridge traces either. Hand values across
with two traces — a saved value from the first ships with the block of the second:

```python
with model.trace(clean, temperature=0.0, max_tokens=1):
    donor = nnsight.save(tuple(o.clone() for o in model.model.layers[L].output))

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(corrupt):
        for served, saved in zip(model.model.layers[L].output, donor):
            served[POS] = saved[POS].to(served.device)
        logits = model.logits.save()
```

Many invokes in one trace still batch: a layer sweep is one trace with one
patched invoke per layer.

### Several sampled sequences (`n > 1`)

`n=k` asks vLLM for k continuations of one prompt, and it fans the request into a
child per sequence — so your block runs k times, once per sequence, against that
sequence's own rows. Saved names come back as a list here for the same reason:

```python
with model.trace(prompt, max_tokens=5, temperature=1.0, n=3) as tracer:
    hidden = model.transformer.h[5].output.save()
    result = tracer.result.save()

hidden[1]                        # sequence 1's
result.outputs[1].text           # what sequence 1 sampled
```

`tracer.result` is **not** a list: a request has one `RequestOutput`, and the
sequences are its `outputs[i]`. A saved container (`caps = nnsight.save([])`
appended under `tracer.iter`) comes back as a list of `n` containers, `caps[i]`
matching `result.outputs[i]`.

Where you hold outputs rather than variables — async, `serve=`, plain `generate`
with an edit installed — the values ride the completion they belong to:

```python
output.outputs[i].saves["hidden"]   # sequence i's
output.saves["hidden"]              # the primary sequence's, unchanged for n=1
```

Since the prompt is shared, each sequence's *prefill* agrees to kernel tolerance;
they diverge over the tokens they go on to sample.

On an **async** engine the sequences finish at different steps, and each streamed
output carries only the completions that ended in that step — so the last one need
not have all `n`. Accumulate across the stream by `completion.index`, or read the
whole set off the finished output's `nnsight_sequences`, which is one dict of the
trace's own saves per sequence, in order.

### Tensor parallelism is transparent

```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=4, dispatch=True)

with model.trace("Hello", temperature=0.0):
    out = model.model.layers[16].output             # (hidden, residual), both full width
    hidden = (out[0] + out[1]).clone().save()
```

`VLLMFragments` (`fragments.py`) gathers a `ColumnParallelLinear`/`RowParallelLinear` shard into the full tensor before your intervention reads it and re-splits on write, so every rank runs the same code on the same complete tensor: `qkv_proj.output`, `gate_up_proj.output`, `o_proj.input` and `down_proj.input` read at their full width; layer outputs and `norm.output` are whole on every rank already. Parameters are not: `layer.weight` is this rank's slice. Verified in `tests/vllm/test_tensor_parallel.py`.

Rules for block code under TP: every rank runs the block, so keep control flow rank-independent; a tensor referenced from outside travels with the block to every rank; an in-block `torch.randn` agrees across ranks (vLLM seeds every worker alike). The client-side `print(model)` shows `tp_size=1` whatever you asked for — check `layer._module.weight.shape` inside the block for the real slice.

A logit lens goes through the model's own logits path, which gathers the vocab shards; calling `model.lm_head(h)` directly raises `LMHead's weights should be used in the sampler`:

```python
out = model.model.layers[20].output
h = (out[0] + out[1])[-1:]
logits = model.logits_processor(model.lm_head, model.model.norm(h))   # [1, vocab]
top1 = logits.argmax(-1).item()
```

### Mixture-of-experts models and expert parallelism

MoE models (Qwen-MoE, DeepSeek, Mixtral, ...) work with the same transparency, in both expert layouts vLLM offers on the same ranks:

- **default** (`enable_expert_parallel=False`): every rank holds a slice of every expert's matrices (the dense-MLP TP sharding, fused across experts);
- **expert parallel** (`enable_expert_parallel=True`): each rank holds `num_experts / world_size` whole experts.

The router (`mlp.gate`, a `ReplicatedLinear`) is full and identical on every rank, so reading router logits or swapping them (expert steering / expert-masking ablation) needs no gathering at all. The fused-experts module (`mlp.experts`, a `FusedMoE`) is the one MoE-specific case the batcher handles: models that build it with `reduce_results=False` (the Qwen-MoE/DeepSeek pattern) make it return **per-rank partial sums** that the outer block all-reduces afterwards, so on access the batcher all-reduces the partials into the true value and on write-back divides by the group size so the block's own all-reduce reconstructs a swapped value exactly once. Verified in `tests/vllm/test_moe_batching.py` against an HF reference.

Individual experts are **not** addressable as submodules: vLLM stacks all local experts into fused weight tensors consumed by one grouped kernel, so there is no `experts[3]` to hook at any parallelism level. The top-k selection and routing weights are computed inside that kernel too — recompute them from the logits. `mlp.gate.output` is `(logits, bias)`, `[tokens, num_experts]`; to ablate an expert, mask its router logit: `mlp.gate.output[0][:, e] = -inf`.

```python
with model.trace(prompt, temperature=0.0, max_tokens=6) as tracer:
    tops = list().save()
    for _ in tracer.iter[:6]:
        logits, _bias = model.model.layers[5].mlp.gate.output
        tops.append(logits[-1].topk(2).indices.clone())
```

### Hybrid (linear-attention) trunks

Qwen3-Next / Qwen3.5-style models interleave gated-delta-net layers with full-attention layers. Both are ordinary decoder-layer envoys; tell them apart by the child they carry, `layers[i].linear_attn` or `layers[i].self_attn`. The recurrent state lives in vLLM's state cache, not in any module output. `taps=` works on these models: a tapped engine pins `cudagraph_mode="FULL_DECODE_ONLY"` on any model vLLM reports as hybrid or attention-free (a full graph captured over a recurrent layer silently miscomputes the other batch composition), and tapped generation matches eager exactly. Checkpoints with a vision tower (`Qwen3_5ForConditionalGeneration`) load and trace on text; their decoder layers are at `model.language_model.model.layers`.

!!! warning "This recipe is vLLM-specific"
    On **`TransformersModel`** it is a silent no-op. A transformers MoE block calls its
    router as `_, top_k_weights, top_k_index = self.gate(hidden_states)` — element `[0]`,
    the logits, is **discarded**, so masking it changes nothing (setting all 64 logits to
    `-inf` gives `max|delta| = 0.0`). There, edit the *selection* instead: write the
    routing weights or expert indices in `mlp.gate.output[1]` / `[2]`.

    Two further transformers-side traps: with `norm_topk_prob=False` masking a weight
    rescales all surviving experts, so a never-selected expert appears to have an effect
    larger than a real below-median one — ablate by selection, not by weight. And the
    router's `.input`/`.output` have **no batch axis** (`(B*T, E)`), so per-token stats
    over a padded batch silently mix in pad rows.

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
something that never heard of nnsight, e.g. another client of the same
`nnsight-serve` engine. Each copy has its own scope, so what it saves is that request's, and it
comes back on that request's output — the same place a trace's values arrive.

```python
model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True, enable_prefix_caching=False)

with model.edit() as (tracer, edit):
    out = model.model.layers[16].output
    hidden = (out[0] + out[1]).clone().save()

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
        readout.append(model.model.layers[16].output[0][-1].clone())
```

After `edit.clear()` a request's output has no `.saves` attribute at all — read it
with `getattr(output, "saves", {})` if the edit may be gone.

| Member | Description |
|---|---|
| `edit.clear()` | Stop running the block. `await edit.aclear()` on an async engine. |
| `edit.name` | The name it was installed under, or `None`. |
| `model.clear_edits()` | Clear every edit still installed. `await model.aclear_edits()` on an async engine. |

### Named edits, and choosing them per request

`model.edit(name="probe")` tags the block. A request then picks which installed
edits run with `edits=[...]` — on `trace(...)`, on an `invoke(...)` (which wins
over the trace's), on a with-less `generate(...)`, on an async engine and on a
served one alike; it rides the request beside the block:

```python
with model.edit(name="probe") as (tracer, probe):
    score = model.model.layers[16].output[1][-1].norm().save()
with model.edit(name="steer") as (tracer, steer):
    model.model.layers[8].output[0][:] += v
with model.edit() as (tracer, always):                      # unnamed
    stamp = nnsight.save(True)

model.generate(prompts, max_tokens=5)                       # all three run
model.generate(prompts, max_tokens=5, edits=["probe"])      # probe + always; no steer
model.generate(prompts, max_tokens=5, edits=[])             # always only

with model.trace(max_tokens=5, edits=["steer"]) as tracer:
    with tracer.invoke(prompt_a):                           # steer + always
        ...
    with tracer.invoke(prompt_b, edits=[]):                 # always only
        ...
```

The rule: no `edits=` runs every edit; `edits=[...]` runs the named ones it lists
**plus every unnamed edit**. A name is a tag, not a key — two edits installed
under one name both run when it is asked for. Naming an edit nothing is
installed under raises (`ValueError`) at the call on a local engine, and comes
back as the request's error from a served one. `edits="probe"` (a string) is
refused; write the list.

That is the whole handle — the values are not read through it. They are taken as
they are collected, so nothing accumulates on the worker for as long as somebody
is reading the outputs, which on the synchronous engine is every request there
is. An error raised inside an installed block is re-raised where its values would
have arrived.

Two different objects carry these, which matters when the names collide:

- **`tracer.result`** is the copy the *worker* hands your block, and it carries the
  edit's values only — `result.saves["hidden"]` is the edit's even if your trace
  saved `hidden` too, and there is no `nnsight_saves` on it. Your trace's own value
  is not missing; it comes back as your variable, the way every traced value does.
- **An output from `model.generate(...)`** is the engine's copy, assembled after the
  fact: `output.saves` holds both kinds with the trace's winning a collision, and
  `output.nnsight_saves` holds the trace's own apart.

Different names avoid the question entirely.

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
  request. Capturing one layer over 500 prompts on Qwen3-8B: **0.5 s edited**
  (bare vLLM 0.45 s) against 0.8 s traced when the block reads only a layer
  envoy bound outside the trace — and **5.2 s traced** the moment the block also
  reads `model.logits`, `model.samples` or `tracer.result`, because a reference
  to the model inside the block ships the model with every invoke (~9 ms each),
  and those properties cannot be bound outside a trace. In a sweep, take the
  text from the edit's outputs.
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
nnsight-serve gpt2 --port 8000 --enable-prefix-caching False [--api-key SECRET] [--gpu-memory-utilization 0.1]
```

`--help` lists only the server's own options; every other `--flag value` is forwarded to vLLM's
`EngineArgs` as `flag=value` (`--max-model-len 4096`, `--tensor-parallel-size 2`). Booleans take a
literal — `--enable-prefix-caching False`, not vLLM's `--no-enable-prefix-caching` — and prefix
caching must be off if you will `edit(serve=...)` (below). Poll `GET /health` for `{"status": "ok"}`
before sending traces; the engine takes a minute or two to build. The server exposes only the
nnsight routes (`/health`, `/v1/nnsight/generate`, `/v1/nnsight/register/{id}`, `.../clear`) —
it is **not** an OpenAI-compatible server, and a "plain" request is a trace whose body saves only
`tracer.result`. `serve=` is accepted by `trace` and `edit`; a with-less `model.generate(serve=...)`
is not routed and would dispatch a local engine. The engine core is a child process: if the server
is killed rather than interrupted, kill the `EngineCore` pid from its log too.

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

`model.model.layers[i].output` is the `(hidden, residual)` pair described under
[What your block sees](#what-your-block-sees-on-vllm). For GPT-2-style models: `model.transformer.h[i].attn.output`, `model.transformer.h[i].mlp.output`. MoE blocks carry `mlp.gate`, `mlp.experts` (and on Qwen-MoE `mlp.shared_expert`); hybrid trunks carry `linear_attn` or `self_attn` per layer; vision-language checkpoints keep the decoder at `model.language_model.model.layers`. Print `model` (or one layer, `model.model.layers[0]`, on a large checkpoint) to see the actual tree.

Paths from `model.named_modules()` carry the root's name (`model.model.layers.0`); `model.get(...)` and `taps=` take the path without it (`model.layers.0`), and `model.taps` reports the prefixed form.

## vLLM versions

Tested against **0.16 through 0.27**, which is what nnsight's own suite runs on.
Two things about newer engines are worth knowing, because nnsight arranges the
first for you and cannot arrange the second:

- **The model runner.** vLLM 0.27 ships a second GPU model runner and defaults to
  it for every non-MoE model. nnsight's hooks arrive by subclassing the original
  one, so it asks vLLM for that one (`VLLM_USE_V2_MODEL_RUNNER=0`) when it builds
  the engine. Setting that variable to `1` yourself is refused rather than
  overridden — the engine would otherwise come up with no interventions installed
  and fail at the first collect with a missing method. Instrumenting the V2 runner
  is not done yet.
- **Tensor parallelism on 0.27 needs `VLLM_WORKER_MULTIPROC_METHOD=spawn`.** vLLM
  forks its workers there by default, and a forked process cannot re-initialize
  CUDA (`RuntimeError: Cannot re-initialize CUDA in forked subprocess`). This is
  vLLM's own setting and applies with or without nnsight.

**MoE reads differently on 0.27, and needs nothing from you.** That release
rebuilt the fused-experts layer around a factory and a modular kernel, and moved
the final all-reduce *inside* the layer — so its output is already the whole
value, and there is nothing for nnsight to gather. Measured on a two-rank
Qwen1.5-MoE: both ranks hand back the identical tensor. Through 0.26 the layer
left a per-rank partial and nnsight gathered it, as described above; either way
what you read is the whole thing.

The one case 0.27 still leaves partial is a layer built to defer its reduce
(`skip_final_all_reduce`), which nnsight gathers as before. A
sequence-parallel MoE layer is split by rows rather than summed — a different
correction, and one nnsight does not make yet, so that value is read as one rank's
rows.

## CUDA graphs with taps

vLLM's decode throughput comes largely from CUDA-graph replay, which `enforce_eager=True` gives up because a replayed graph runs no Python and so no hooks. `taps` keeps the graphs: the locations you name are recorded *into* the graph as breaks (vLLM's `VLLM_USE_BREAKABLE_CUDAGRAPH`), and at each break the interleaver's normal handoff runs on every replay. Everything else is unchanged — the same trace syntax, per-request scoping, `tracer.iter`, `model.logits` / `model.samples`.

```python
model = VLLM(
    "meta-llama/Llama-3.1-8B",
    dispatch=True,
    taps=["model.layers.*.output", "model.layers.10.mlp.input"],
)

with model.trace(prompt, max_tokens=20, temperature=0.0) as tracer:
    hiddens = list().save()
    for _ in tracer.iter[:20]:                                     # every step, prefill included
        model.model.layers[10].output[0][:] += 4 * steering_vector   # in place
        hiddens.append(model.model.layers[16].output[0].clone())     # clone: see below
```

Outside the loop the edit would fire on the prefill only. Measured on Qwen3-8B, one request, 32
greedy tokens with a per-step steering edit at one tap and a per-step capture at another: eager 60
tok/s, graphs 86 tok/s; the same graph engine with no trace 89 tok/s.

What changes under graphs, and why:

- **Only taps are reachable.** A location you didn't declare is never visited by a replayed step, so a block that reads one parks until the request ends and the request's error says `'...' is not a tap on this engine`. Declare it, or drop `taps` for the eager engine. `model.taps` lists the resolved set after dispatch. Keep the set small: each tap splits the graph, and a break at every module would cost what replay bought.
- **Edits land in place.** The next kernel reads the tap's tensor from a fixed address, so an in-place edit (`output[0][:] += v`, `output[0][:, i] = 0`) is exactly right and a replacement (`output = t`) is copied back into that memory — it has to have the same shape, else the request errors.
- **Clone what you keep.** The value served at a tap *is* the graph's memory, rewritten next step. A tensor you `.save()` or append under `tracer.iter` aliases it; call `.clone()` if you read it after the step. The un-cloned list still comes back as N separate tensors — each is copied at collect time — all holding the last step's values, and nothing warns. The eager engine has the same rule: a served value is the model's live buffer, and the next layer's fused add+norm (or DeepSeek's MLA attention, which rotates the `q_proj` output in place) rewrites it after the module returns — `.clone()` makes a read a read.
- **`torch.compile` is off.** Breakable graphs keep replay and drop the compiled path; that is most of the throughput, not all of it. The flag is process-wide, so one process holds either graph engines or compiled ones.

## Limitations

- **`enforce_eager=True` is forced unless you declare `taps`** — see [CUDA graphs with taps](#cuda-graphs-with-taps).
- **One prompt per invoke** — no `tracer.invoke(["a", "b"])`.
- **No `tracer.barrier(n)`** — each invoke is its own request and the engine schedules them independently, so the blocks never run against the same forward. Calling it raises rather than hanging.
- **No backward / gradients, no `.scan()`, no source tracing on fused CUDA kernels.**
- **Text prompts only** — image/video inputs are not accepted; vision-language checkpoints load and their language trunk traces normally.
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
- **One name saved by several runs comes back as a list** — one entry per invoke, and per sampled sequence when `n > 1`, in submission order. A name saved once stays that value. A container saved *above* the invokes is one object and merges instead (see [Continuous batching](#continuous-batching-multiple-invokes)).
- **`tracer.result` is the finished `RequestOutput`, not per-step.** Use `model.logits` / `model.samples` under `tracer.iter` for per-step values. It is one object however many sequences `n` asked for, and it carries an installed edit's values but not the trace's own — those come back as your variables.
- **An empty `tracer.invoke()` with interventions raises** (its work would vanish); a do-nothing empty invoke is a harmless no-op.
- **A typo'd sampling kwarg raises** (`trace(temperatur=0.0)` → `TypeError`), rather than being silently ignored.
- **Mode is fixed at construction.** Build with `mode="async"` if you want streaming.
- **`tracer.backend` is iterable only in async mode.** In sync mode, results land in your local variables when the block exits.
- **`logits` / `samples` don't exist on a vanilla `vllm.LLM`** — use them only inside a trace.
- **Per-invoke kwargs override trace-level ones**, including when the value an invoke names happens to be vLLM's own default (`temperature=1.0`, `max_tokens=16`). Trace-level settings only fill in what an invoke did not name.
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
