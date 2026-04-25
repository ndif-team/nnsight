---
title: Async vLLM
one_liner: Stream RequestOutput objects token-by-token from vLLM's AsyncLLM with intervention saves attached to each output.
tags: [remote, vllm, async, streaming]
related: [docs/remote/index.md]
sources: [src/nnsight/modeling/vllm/vllm.py:68, src/nnsight/modeling/vllm/vllm.py:445, src/nnsight/modeling/vllm/async_backend.py:19, src/nnsight/modeling/vllm/README.md]
---

# Async vLLM

## What this is for

vLLM's `AsyncLLM` engine streams generated tokens as they're produced. The async path on `nnsight.modeling.vllm.VLLM` bridges that streaming into nnsight's tracing model: you write a normal `with model.trace(...)` block, and after the block exits you `async for output in tracer.backend()` to receive a `RequestOutput` per step. Each output has a `.saves` attribute containing the saved values at that step.

> Naming note: this is the **vLLM async engine**, not "async NDIF". Async vLLM runs locally (or on a server you control); it doesn't use NDIF. It's documented here because it's the other way nnsight does non-blocking, streamed remote-style execution.

## When to use / when not to use

- Use for chat / interactive UIs that need partial output as it's generated.
- Use to monitor intervention state evolving across generation steps in real time.
- Use to batch many concurrent requests (vLLM's continuous batcher handles them all).
- Don't use for one-shot scoring — sync `mode="sync"` is simpler.
- Don't use with `remote=True` — `VLLM.trace` explicitly disables async backend injection when `remote` is set (`src/nnsight/modeling/vllm/vllm.py:449`).

## Canonical pattern

```python
import asyncio
from nnsight.modeling.vllm import VLLM

# mode="async" goes on the constructor, NOT on trace().
model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True, mode="async")

async def main():
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.save()

    # tracer.backend() returns the AsyncVLLMBackend; iterate it for streamed outputs.
    async for output in tracer.backend():
        print(
            f"finished={output.finished}, "
            f"text={output.outputs[0].text!r}, "
            f"saves={list(output.saves.keys())}"
        )

asyncio.run(main())
```

How invocation works:

- `mode="async"` is the **only** entry point. It's a kwarg to `VLLM.__init__` (`src/nnsight/modeling/vllm/vllm.py:70`).
- Setting `mode="async"` flips `self._async_engine = True` and at load time builds an `AsyncLLM` instead of a sync `LLM` (`src/nnsight/modeling/vllm/vllm.py:185`).
- `VLLM.trace()` detects `self._async_engine` and injects `AsyncVLLMBackend` as the trace's backend (`src/nnsight/modeling/vllm/vllm.py:445`):

  ```python
  def trace(self, *inputs, **kwargs):
      if (
          self._async_engine
          and kwargs.get("backend") is None
          and not kwargs.get("remote")
      ):
          from .async_backend import AsyncVLLMBackend
          kwargs["backend"] = AsyncVLLMBackend(self)
      return super().trace(*inputs, **kwargs)
  ```
- There is **no** `mode="async"` argument to `trace()`. The mode is fixed at model construction time.

## What .saves looks like on each output

`AsyncVLLMBackend.__aiter__` (`src/nnsight/modeling/vllm/async_backend.py:77`) iterates the underlying `AsyncLLM.generate(...)` generator. For every `RequestOutput`:

1. If the output is `finished`, the worker is asked via `collective_rpc("collect_nnsight", ...)` to finalize the mediator and gather all saved values.
2. The returned bytes are zstd-decompressed and unpickled into a dict.
3. The dict is attached as `output.saves`.

So `output.saves` is a `dict[str, Any]` keyed by the saved variable name in your trace:

```python
async for output in tracer.backend():
    if "logits" in output.saves:
        print(output.saves["logits"].shape)
```

In practice, intermediate (non-finished) outputs may have `output.saves` without the trace-shared values populated — only finished outputs run mediator finalization. See [vllm/README.md "Streaming Saves"](../../src/nnsight/modeling/vllm/README.md) for the worker-side semantics.

## Multi-prompt streaming

Each invoke is one vLLM request. With async, all requests are submitted at once and stream concurrently:

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

vLLM's continuous batcher dynamically batches all active requests on the GPU, so adding more prompts doesn't linearly slow generation.

## Awaiting a single result

If you don't care about streaming and just want the final output, await the backend directly. `AsyncVLLMBackend.__await__` proxies the underlying generator's `__await__` (`src/nnsight/modeling/vllm/async_backend.py:74`):

```python
async def main():
    with model.trace("Hello", max_tokens=3) as tracer:
        logits = model.logits.save()

    final_output = await tracer.backend()
    print(final_output.saves["logits"].shape)
```

(For most use cases, `async for` is more useful — it gives you per-step output.)

## Async + Ray

Async vLLM works with Ray-distributed tensor parallelism by passing `distributed_executor_backend="ray"`:

```python
model = VLLM(
    "meta-llama/Llama-3.1-70B",
    tensor_parallel_size=4,
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.85,
    dispatch=True,
    mode="async",
)
```

`VLLM._load` automatically swaps in `NNsightRayExecutor` and pre-initializes Ray when both async and Ray are requested (`src/nnsight/modeling/vllm/vllm.py:185`). See `src/nnsight/modeling/vllm/README.md` § Ray Distributed Executor.

## Gotchas

- `mode="async"` must be on the `VLLM(...)` constructor. Setting it on `trace()` does nothing.
- `remote=True` is incompatible with `mode="async"` — `VLLM.trace` explicitly skips the async-backend injection if `remote` is passed (`src/nnsight/modeling/vllm/vllm.py:449`). NDIF currently runs the sync vLLM path.
- `.saves` is a regular Python dict on the `RequestOutput`; saved values are copied per output. Holding references to many outputs holds references to many save dicts.
- The underlying generator is a one-shot. Once you've iterated it to completion, calling `tracer.backend()` again won't restart generation.
- AsyncLLM uses dynamic batching — order of `output.request_id`s in the stream is **not** the order of your invokes. Match by `request_id` if order matters.
- Pipeline parallelism (`pipeline_parallel_size > 1`) is not supported by the integration; saves from non-rank-0 stages are lost. See `src/nnsight/modeling/vllm/README.md` § Limitations.

## Related

- [src/nnsight/modeling/vllm/README.md](../../src/nnsight/modeling/vllm/README.md) — full architecture (sync vs async paths, batch group management, Ray support).
- [docs/remote/index.md](./index.md) — when to pick sync remote vs async vLLM.
