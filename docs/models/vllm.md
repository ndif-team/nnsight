---
title: VLLM
one_liner: Trace and edit a model whose forward pass is a vLLM engine — continuous batching, flat token rows, per-step values, CUDA-graph taps.
tags: [models, vllm, serving, production]
related: [docs/models/index.md, docs/models/transformers-model.md, docs/models/vllm-editing.md, docs/models/vllm-serving.md, docs/models/vllm-parallelism.md]
sources: [src/nnsight/modeling/vllm/vllm.py, src/nnsight/modeling/vllm/tracer.py, src/nnsight/modeling/vllm/interleaver.py, src/nnsight/modeling/vllm/batching.py, src/nnsight/modeling/vllm/fragments.py, src/nnsight/modeling/vllm/collect.py, tests/vllm/]
---

# VLLM

## What this is for

`nnsight.modeling.vllm.VLLM` runs NNsight interventions on top of vLLM's inference engine. The model runs in vLLM's own worker process(es); your intervention code is serialized onto each request (via `SamplingParams.extra_args`), carried into the worker, run against the real module, and the saved values shipped back. You get PagedAttention, continuous batching, tensor parallelism, and optional async streaming, with arbitrary Python interventions inline with the forward.

Interventions are written exactly as for any other model — the module tree mirrors what vLLM loaded. The one structural difference: each `tracer.invoke(...)` is **one vLLM request** (one prompt), so several prompts means several invokes, not a list.

This page is the engine and what a block sees inside it. Three companions carry the rest:
[Editing a vLLM engine](vllm-editing.md) for `model.edit()`, [Async mode and
serving](vllm-serving.md) for `mode="async"` and `nnsight-serve`, and [Parallelism and
architectures](vllm-parallelism.md) for tensor parallelism, mixture-of-experts and hybrid trunks.

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
| `mode` | `"sync"` (default) builds a `vllm.LLM` and runs synchronous generation; `"async"` builds a `vllm.v1.engine.async_llm.AsyncLLM` and a trace streams outputs via `async for output in tracer.backend`. Set at construction — you can't switch per trace. |
| `dispatch` | `True` creates the real engine during `__init__`. `False` (default) builds only a meta-tensor tree (via vLLM's `DummyModelLoader`, `device="meta"`) — no GPU memory used until the first trace. |
| `tensor_parallel_size` | GPUs to shard across. Transparent to your interventions (see [Parallelism and architectures](vllm-parallelism.md)). |
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

### Replacing a value keeps its rows

An in-place write (`out[0][-1] += v`, `mlp.output[:] = 0`) is the direct form and always fits.
A *replacement* — `layer.output = t` — is spliced back into the batch the model is running, so it
has to carry the same rows the block owns. A tensor with a different leading dimension is refused
before it reaches the next module:

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1) as tracer:
    out = model.transformer.h[6].output            # (10, 768) — this request's ten prompt rows
    model.transformer.h[6].output = out[:2]        # 2 rows
# RuntimeError: ValueError: A batched write has to keep its rows: this block owns rows 0:10 of 10,
#     so the replacement must be (10, 768), not (2, 768).
```

This is what a patching sweep hits when a donor activation was captured at a different prompt
length. Slice the donor to the rows you are writing (`served[POS] = donor[POS]`) rather than
handing back a shorter tensor. Every other dimension is the model's to check, and mismatches
there surface from the next kernel.

The error ends that request. The engine keeps serving, and a request from another client in the
same batch finishes normally. The invokes of one trace are not separate that way: they are one
block, and it raises as a whole, so a batched sweep loses the invokes that were fine along with
the one that was not. The refusal reaches you as a `RuntimeError` whose message begins with the
original `ValueError:` line, so match on the message rather than on the class.

### `logits` and `samples`

These are vLLM-specific served values on the model — `eproperty` descriptors, the same mechanism behind a module's `.output`/`.input` — not on a vanilla `vllm.LLM`, and only meaningful inside a trace. Read them for the logits and sampled ids of each step; for the finished request as a whole, read `tracer.result`.

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
outputs[3].saves["hidden"]        # see Editing a vLLM engine
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

It is served at collect time — after every module, `logits` and `samples` visit of the request,
and the first moment both halves exist, since the block runs in the engine's worker and the
output is assembled by the engine. So it must be the **last** read in the block; a read after it
raises `OutOfOrderError` naming that later value. Two consequences: it carries the generation but
**not** `.saves` (those are attached to the engine's own copy afterwards), and per-step values
still come from `model.logits` / `model.samples` under `tracer.iter`, since `result` is the
request's end state.

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
## Special properties

| Attribute | Description |
|-----------|-------------|
| `model.logits` / `model.samples` | Pre-sampling logits / sampled ids for the step (trace-only, `eproperty`). |
| `model.tokenizer` | The tokenizer vLLM resolved for the checkpoint. Loaded eagerly. |
| `model.vllm_entrypoint` | The underlying `vllm.LLM` (sync) or `AsyncLLM` (async); `None` until dispatch. |
| `model.taps` | The resolved tap set on a `taps=` engine; empty on an eager one. |
| `model.dispatched` | Whether the engine is built (`MetaMixin`). |

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

- **A tap can be an operation inside a forward.** `"model.layers.10.self_attn.source.qkv_split_0.output"` taps a `.source` op: the worker installs the source-instrumented forward for exactly the modules such taps name, before the graphs are recorded, and the op is served on replay like any module location (reads bitwise-equal to the eager engine, in-place edits land). An op name the forward does not have is refused while the engine builds; the caller gets
  `RuntimeError: Engine core initialization failed`, and the message that names the ops the forward
  *does* have is in the EngineCore subprocess's output above it (see [When the engine builds, or
  does not](#when-the-engine-builds-or-does-not)). Ops inside a fused kernel are still not locations.
- **Only taps are reachable.** A location you didn't declare is never visited by a replayed step, so a block that reads one parks until the request ends and the request's error says `'...' is not a tap on this engine`. Declare it, or drop `taps` for the eager engine. `model.taps` lists the resolved set after dispatch. Keep the set small: each tap splits the graph, and a break at every module would cost what replay bought.
- **Edits land in place.** The next kernel reads the tap's tensor from a fixed address, so an in-place edit (`output[0][:] += v`, `output[0][:, i] = 0`) is exactly right and a replacement (`output = t`) is copied back into that memory, so it has to keep the rows it
  replaces — see [Replacing a value](#replacing-a-value-keeps-its-rows).
- **Clone what you keep.** The value served at a tap *is* the graph's memory, rewritten next step. A tensor you `.save()` or append under `tracer.iter` aliases it; call `.clone()` if you read it after the step. The un-cloned list still comes back as N separate tensors — each is copied at collect time — but the decode entries all hold the last step's values, and nothing warns. (The prefill entry is a different buffer, sized to the prompt, so it survives; every decode entry aliases the same one-row tensor.) The eager engine has the same rule: a served value is the model's live buffer, and the next layer's fused add+norm (or DeepSeek's MLA attention, which rotates the `q_proj` output in place) rewrites it after the module returns — `.clone()` makes a read a read.
- **`torch.compile` is off.** Breakable graphs keep replay and drop the compiled path; that is most of the throughput, not all of it. The flag is process-wide, so one process holds either graph engines or compiled ones.
## Where an error comes from

Your block runs in vLLM's EngineCore subprocess, and that decides where you read about a failure.

**Inside a trace, errors come home.** An exception in the block is re-raised in your process as a
`RuntimeError` carrying the original type, its message and an "Intervention traceback" pointing at
your line — `1/0` in a block arrives as `RuntimeError: ZeroDivisionError: division by zero`.
Because the type is carried in the message rather than in the class, catch `RuntimeError` and match
on the text. The engine keeps serving afterwards, and requests from other clients in the same batch
are unaffected.

**Warnings do not.** A `warnings.warn` raised while the block runs is emitted by the EngineCore
process, so it prints to that process's output rather than to yours, and a
`warnings.catch_warnings()` around the trace records nothing. A `tracer.iter[:N]` that asks for more
steps than the request generates is the case you are most likely to meet: the loop is cut short,
nothing after it runs, and the note saying so is in the engine's output. Hold the run to the count
you loop over — `min_new_tokens=N`, or `ignore_eos=True` when the model would otherwise stop
early — rather than relying on seeing the warning.

### When the engine builds, or does not

Everything a `VLLM(...)` constructor does in the worker fails the same shape:

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

"Above" is literal. The real message is in the `(EngineCore pid=...)` lines further up the output,
usually many screens of vLLM logging back. Two causes are common enough to recognize:

- a `.source` op named in `taps=` that the forward does not have. (A tap naming no *module* is
  caught client-side, against the meta tree, and raises `ValueError` where you wrote it; the op is
  checked in the worker, where the forward is instrumented, and the EngineCore line names the ops
  it *does* have.)
- `AssertionError: Error in memory profiling. Initial free memory ... current free memory ...` on a
  shared GPU, when another process frees memory while vLLM profiles. Nothing is wrong with the
  trace: build again, or use a card nothing else is holding.

`Chunked prefill is enabled with max_num_batched_tokens=...` is printed on every construction and
does not describe the engine you get. It comes from the meta tree nnsight builds first; the real
engine's arguments are logged a few lines later as `non-default args: {... 'enable_chunked_prefill':
False ...}`.

## Limitations

- **`enforce_eager=True` is forced unless you declare `taps`** — see [CUDA graphs with taps](#cuda-graphs-with-taps).
- **One prompt per invoke** — no `tracer.invoke(["a", "b"])`.
- **No `tracer.barrier(n)`** — each invoke is its own request and the engine schedules them independently, so the blocks never run against the same forward. Calling it raises rather than hanging.
- **No backward / gradients, and no source tracing inside a fused CUDA kernel** — the kernel's
  inputs and outputs are locations; its interior is not Python.
- **No `.scan()`.** It propagates shapes by running the model's own forward under a fake-tensor
  mode, and there is no forward on this side to run. `model.scan(...)` raises
  `NotImplementedError: scan is unavailable on vLLM: ... Trace a prompt and read the shapes off the
  activations it serves.`
- **Text prompts only** — image/video inputs are not accepted; vision-language checkpoints load and their language trunk traces normally.
- **Version sensitivity** — nnsight targets vLLM's V1 engine and imports its internals directly
  (`vllm.tokenizers`, `vllm.v1.worker.gpu_model_runner`, `vllm.v1.engine.async_llm`). The `vllm`
  extra carries no upper bound, so `pip install "nnsight[vllm]"` takes the current release; a
  release that moves one of those names fails at `import nnsight.modeling.vllm` with a
  `ModuleNotFoundError` rather than a version message. Everything on this page was run on 0.27.1.

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
- **A replacement keeps the rows it replaces** — `layer.output = t` with a different leading
  dimension is refused before the model sees it (see [Replacing a
  value](#replacing-a-value-keeps-its-rows)).
- **Mode is fixed at construction.** Build with `mode="async"` if you want streaming.
- **`tracer.backend` is iterable only in async mode.** In sync mode, results land in your local variables when the block exits.
- **`logits` / `samples` don't exist on a vanilla `vllm.LLM`** — use them only inside a trace.
- **Per-invoke kwargs override trace-level ones**, including when the value an invoke names happens to be vLLM's own default (`temperature=1.0`, `max_tokens=16`). Trace-level settings only fill in what an invoke did not name.
- **`gpu_memory_utilization` defaults to 0.9** — lower it for small/shared GPUs (tests use 0.1).
- **Async saves are finished-only** — for per-step saves, accumulate inside `tracer.iter[:]`.

## Related

- [docs/models/vllm-editing.md](vllm-editing.md) — `model.edit()` on the engine
- [docs/models/vllm-serving.md](vllm-serving.md) — async mode, `nnsight-serve`, remote
- [docs/models/vllm-parallelism.md](vllm-parallelism.md) — tensor parallelism, MoE, hybrid trunks
- [docs/models/index.md](index.md) — decision tree
- [docs/models/transformers-model.md](transformers-model.md) — the text-only HF alternative
- `src/nnsight/modeling/vllm/vllm.py` — `VLLM`
- `src/nnsight/modeling/vllm/tracer.py` — `VLLMTracer`, the per-request block
- `src/nnsight/modeling/vllm/interleaver.py` — `VLLMInterleaver`, the worker-side handoff
- `src/nnsight/modeling/vllm/batching.py` — `VLLMBatcher`, one request's rows out of the slab
- `src/nnsight/modeling/vllm/fragments.py` — `VLLMFragments`, the tensor-parallel gather/scatter
- `src/nnsight/modeling/vllm/collect.py` — `merge_shared_saves`, values home from the worker
- `tests/vllm/` — runnable examples (tracing, async, tensor parallelism, mixture-of-experts, requests, serve)