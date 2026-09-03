---
title: vLLM async mode and serving
one_liner: mode="async" streams RequestOutputs from a trace; nnsight-serve holds one engine behind HTTP for GPU-less clients.
tags: [models, vllm, async, serving]
related: [docs/models/vllm.md, docs/models/vllm-editing.md, docs/remote/index.md]
sources: [src/nnsight/modeling/vllm/async_backend.py, src/nnsight/modeling/vllm/serve/, tests/vllm/]
---

# Async mode and serving


Construct with `mode="async"`; a trace then streams `RequestOutput`s. Iterate `tracer.backend` (an attribute, not a call):

```python
import asyncio
from nnsight.modeling.vllm import VLLM

async def main():
    model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True, mode="async")

    with model.trace("The Eiffel Tower is located in the city of",
                     temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.save()

    async for output in tracer.backend:
        print(output.finished, output.outputs[0].text)
        if output.finished:
            print("saves:", list(output.saves.keys()))

if __name__ == "__main__":
    asyncio.run(main())
```

Both halves of that shape are required. `AsyncLLM` binds to the loop that built it, where it keeps its output handler and its per-request futures, so build the model **inside** the coroutine you will await it from: an engine built at import and then used from two `asyncio.run()` calls never answers the second, and does not error either. The `__main__` guard is what makes the file runnable at all — see [the engine page](vllm.md).

Saves are attached **only to the finished output** (`output.finished == True`), fetched from the worker via `collect_nnsight` at that point (`async_backend.py`). Intermediate yields carry no saves — accumulate per-step values inside `tracer.iter[:]` instead.

Await the backend to drain the stream and get just the last output:

```python
last = await tracer.backend
print(last.saves["logits"].shape)
```

**Read saves off the output, not out of your variables.** The async path is the one backend that
does not push saved names back into the calling frame: `logits` is still unbound after the stream,
and the tensor is `last.saves["logits"]`, exactly as written above. The name you save under is the
key you read.

## Async notes

- Async tracing takes a **single prompt** (one invoke or a direct input) — several invokes raise `NotImplementedError` (`async_backend.py`).
- The stream is consumed once. A second `await tracer.backend` returns `None` rather than raising, and the `AttributeError` lands wherever you use it; fire a new trace per generation.
- A stream closed before it finishes aborts the request and frees its worker (`async_backend.py`).
- Errors in the block surface when you iterate the stream (a `1/0` raises `RuntimeError: ...ZeroDivisionError`).
- `model.generate(...)` on an async engine returns a coroutine; `await` it.
- `remote=True` skips async-backend injection (`vllm.py`).

## Remote / serve

- `trace(..., remote=True)` runs on NDIF. The model key is the repo id (`vllm.py`).
- `trace(..., serve=url, api_key=...)` runs the trace on a standalone **nnsight-serve** engine: the block is written against a **GPU-less** meta model, serialized like the NDIF path, sent to the server, and its saved values pushed back into your frame — so reading a `.save()`d variable after the block works exactly as locally.

Start a server (holds one dispatched async engine):

```bash
nnsight-serve gpt2 --port 8000 --enable-prefix-caching False [--api-key SECRET] [--gpu-memory-utilization 0.1]
```

`--help` lists only the server's own options (`--host`, `--port`, `--api-key`); every other
`--flag value` is forwarded to vLLM's `EngineArgs` as `flag=value` (`--max-model-len 4096`,
`--tensor-parallel-size 2`). Three limits of that forwarding, all quiet:

- **Booleans take a literal.** `--enable-prefix-caching False`, not vLLM's
  `--no-enable-prefix-caching`, which arrives as `no_enable_prefix_caching=True` and stops the
  server with `TypeError: EngineArgs.__init__() got an unexpected keyword argument`.
- **Only long flags are read.** `-tp 2` prints `Ignoring unknown argument: -tp` to stderr and the
  server comes up at `tensor_parallel_size=1`. Spell it `--tensor-parallel-size 2`.
- **A value is always a scalar**, so `taps=` has no spelling here: `--taps model.layers.*.output`
  arrives as a string, which the engine iterates one character at a time and refuses with
  `ValueError: Tap 'm' names no module`. Build a tapped engine in Python.

Prefix caching must be off if you will `edit(serve=...)` (below). Poll `GET /health` for `{"status": "ok"}`
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
