---
title: Editing a vLLM engine
one_liner: model.edit() installs a block on the engine itself, so it runs for every request the engine serves — including requests from clients that never heard of nnsight.
tags: [models, vllm, editing, serving]
related: [docs/models/vllm.md, docs/models/vllm-serving.md, docs/models/index.md]
sources: [src/nnsight/modeling/vllm/vllm.py, src/nnsight/modeling/vllm/registration.py, src/nnsight/modeling/vllm/collect.py, tests/vllm/]
---

# Editing the engine

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

`edit(inplace=False)` — a copy of the model carrying the edit, which is what it means on
`TransformersModel` — is refused here: the block lands on the engine every caller shares, and
there is no copy to put it on instead. Trace the requests you want changed, or install the edit
and choose it per request with `edits=` (below).

## Named edits, and choosing them per request

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

## On an async engine

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

## When to edit instead of trace

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

> **Prefix caching must be off — for an edit.** A prefix-cached token is served
> from the KV cache without a forward pass, so no hook fires and an installed
> block sees a short activation with no error. A trace asks for its own request
> to be recomputed and so needs nothing; an edit rides requests it did not create
> and cannot ask. Build with `enable_prefix_caching=False` — editing an engine
> that has it on warns.
