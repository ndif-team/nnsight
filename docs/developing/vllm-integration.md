---
title: vLLM Integration Internals
one_liner: How NNsight injects interventions into vLLM's worker processes, hooks the forward / logits / sampling phases, and transports mediators across process boundaries.
tags: [internals, dev]
related: [docs/developing/batching-internals.md, docs/developing/serialization.md, docs/developing/backends.md, docs/developing/adding-a-new-runtime.md]
sources: [src/nnsight/modeling/vllm/vllm.py:1, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:1, src/nnsight/modeling/vllm/workers/GPUWorker.py:1, src/nnsight/modeling/vllm/engines/engine.py:1, src/nnsight/modeling/vllm/async_backend.py:1, src/nnsight/modeling/vllm/sampling.py:1, src/nnsight/modeling/vllm/batching.py:1, src/nnsight/modeling/vllm/README.md]
---

# vLLM Integration Internals

## What this covers

The vLLM integration is the most invasive runtime in NNsight. vLLM owns the model, runs forward/logits/sampling in worker subprocesses, uses a flat `[total_tokens, hidden]` tensor format, and continuously batches requests across generation steps. NNsight injects interventions at three points (forward, logits, samples), serializes mediator code across process boundaries via `SamplingParams.extra_args`, and uses a custom `Batcher` to handle both per-mediator slicing and tensor-parallel gather/scatter.

This document focuses on the **internals** — process boundaries, interleaver entry points, batch group lifecycle, and async streaming. The user-facing surface (`VLLM("gpt2", dispatch=True)` and `with model.trace(...)`) is covered in [`vllm/README.md`](../../src/nnsight/modeling/vllm/README.md), and that file is the canonical narrative reference. Read it alongside this one.

## Architecture / How it works

### Two-process layout

The `VLLM` Envoy class exists in two processes:

- **User process.** Created by `VLLM("gpt2", dispatch=True)` (`src/nnsight/modeling/vllm/vllm.py:43`). Holds the meta model (built from a `DummyModelLoader` at `src/nnsight/modeling/vllm/vllm.py:135`), the tokenizer, and the `vllm_entrypoint` (`vllm.LLM` or `AsyncLLM`). Used for tracing — it captures source, builds mediators, and submits requests.
- **Worker process(es).** vLLM spawns these via `worker_cls="nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"` (`src/nnsight/modeling/vllm/vllm.py:200,209`). Each worker creates a second `VLLM` Envoy in `NNsightGPUModelRunner.load_model` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:343`), wrapping the actually-loaded model. This worker-side instance owns the `Interleaver` and `VLLMBatcher` and is where interventions actually run.

vLLM 0.15+ uses `spawn` multiprocessing, so the worker doesn't inherit env vars set after import time. Distributed setup (`init_distributed_environment`) happens in the user process for compatibility — see `src/nnsight/modeling/vllm/vllm.py:78` — and is destroyed before vLLM creates its own workers.

### `NNsightGPUModelRunner` — three interleaver entry points

`NNsightGPUModelRunner` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:29`) extends vLLM's `GPUModelRunner` and is monkey-patched in via `NNsightGPUWorker.__init__` (`src/nnsight/modeling/vllm/workers/GPUWorker.py:14`). It enters the interleaver at three places:

1. **Forward pass** (`execute_model`, `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:385`). `Globals.enter()` then `with self.nnsight_model.interleaver:` then `super().execute_model(...)`. After the forward returns, `unflatten()` switches batch groups from token-level to prompt-level for the next phase.
2. **Logits wrap** (`sample_tokens`, `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:402`). Inside another interleaver context, the runner calls `type(self.nnsight_model).logits.provide(self.nnsight_model, self.execute_model_state.logits)` to feed logits through the `logits` eproperty so mediators can observe / modify them.
3. **Sampling** (`_sample`, `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:426`). Wraps `super()._sample(...)`. After sampling, `sampler_output.sampled_token_ids = type(self.nnsight_model).samples.provide(self.nnsight_model, sampler_output.sampled_token_ids)` feeds samples through the `samples` eproperty.

Across all three, mediator threads stay alive — they do not restart per phase. The interleaver context is entered/exited multiple times within a single generation step, but mediator threads continue waiting for whichever value they are blocked on.

### `NNsightSamplingParams` + `extra_args` mediator transport

The intervention code is compiled in the user process but must execute in the worker process. NNsight uses vLLM's built-in `SamplingParams.extra_args` dict — a feature that survives both pickle (Ray) and msgpack (multiprocessing) serialization — to carry serialized mediator bytes plus per-trace metadata.

`NNsightSamplingParams` (`src/nnsight/modeling/vllm/sampling.py:4`) is a thin subclass of `vllm.SamplingParams`. The mediator is **not** stored as a custom field — it lives in `extra_args`:

```python
param.extra_args = {
    "nnsight_mediator": serialize(mediator),
    "nnsight_trace_id": trace_id,
    "nnsight_trace_idx": idx,
    "nnsight_saved_names": saved_names,
    "nnsight_expected_count": count,
}
```

`VLLM._serialize_mediators` (`src/nnsight/modeling/vllm/vllm.py:349`) builds this dict, with one mediator per input invoke. The `trace_id` (uuid4) groups all mediators from the same `model.trace(...)` block; `nnsight_saved_names` is the list of parent-frame variable names that were `.save()`d at trace scope, so the worker knows which globals to gather later.

Deserialization happens in `NNsightRequestHelper.process_new_reqs` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:59`):

- For the first mediator of a `trace_id`, its `__globals__` becomes the canonical reference. Saved-variable IDs are added to `Globals.saves` on the worker side so the worker recognizes them as save targets.
- For subsequent mediators of the same trace, the saved-variable entries from canonical globals are grafted into the new mediator's `__globals__`, so all mediators in one trace share the same Python objects (e.g. a shared list that every invoke appends to).

### `NNsightLLMEngine.collect_nnsight()` — sync result collection

`NNsightLLMEngine.step()` (`src/nnsight/modeling/vllm/engines/engine.py:18`) wraps `super().step()`, detects finished requests, and calls `engine_core.collective_rpc("collect_nnsight", args=(finished_req_ids, finished_req_ids))`. The first argument tells the worker which requests to collect saves from; the second tells it which to also finalize and clean up. Rank-0 returns zstd-compressed pickled bytes, other ranks return `None`.

The result is attached to each finished `RequestOutput` as `.saves`. Back in `VLLM.__call__` (`src/nnsight/modeling/vllm/vllm.py:409`), the user-process side reads these and pushes them to the user's frame via `push_variables`.

### `AsyncVLLMBackend` dual-call pattern

`AsyncVLLMBackend` (`src/nnsight/modeling/vllm/async_backend.py:19`) handles streaming. It exists because vLLM exposes `AsyncLLM.generate()` as an async generator, and a normal `Backend` runs synchronously inside `Tracer.__exit__`. The dual-call pattern:

1. **First call: `__call__(tracer)`** at `__exit__` time (`src/nnsight/modeling/vllm/async_backend.py:36`). Compiles the traced function via `Backend.__call__(self, tracer)`, runs it via `tracer._setup_interleaver(fn)` (which sets up mediators **without** triggering generation), then calls `self.model._serialize_mediators(...)` and submits the request via `vllm_entrypoint.generate(...)`. Stores the resulting async generator on `self._generator`.
2. **Second call: `__aiter__`** (`src/nnsight/modeling/vllm/async_backend.py:77`). Iterates the stored generator. **Saves are collected only on the final, `finished == True` output for a request id** — not on every streamed output. When `output.finished` becomes true, `collective_rpc("collect_nnsight", args=([output.request_id], [output.request_id]))` runs on the workers to collect saves, the returned bytes are zstd-decompressed and pickled into `output.saves`. The user does `async for output in tracer.backend()` and only the final yielded `RequestOutput` for each request carries `saves`.

This pattern requires that `VLLM.trace()` swap in the right tracer class. `VLLM.trace` (`src/nnsight/modeling/vllm/vllm.py:445`) checks `self._async_engine` and injects `AsyncVLLMBackend` plus `tracer_cls=AsyncInterleavingTracer`, bypassing `RemoteableMixin.trace` (which hard-codes `RemoteInterleavingTracer`).

#### Demos and reference implementations

Two external repos demonstrate the integration end-to-end:

- [`nnsight-vllm-demos`](https://github.com/ndif-team/nnsight-vllm-demos) — async chat with SAE-based steering, plus other demo apps.
- [`nnsight-vllm-lens-comparison`](https://github.com/ndif-team/nnsight-vllm-lens-comparison) — reference implementation comparing logit-lens variants on top of vLLM. Useful as a template for new lens-style applications.

### Cross-process serialization

Mediators travel across process boundaries via `serialize(mediator)` (`src/nnsight/intervention/serialization.py:1083`, the source-based pickler), embedded in `extra_args`. Since `extra_args: dict[str, Any] | None` is a built-in `SamplingParams` field, vLLM's transport layer handles it transparently — pickle for Ray, msgpack for multiprocessing.

Crucially, the mediator's `intervention` function is serialized by source code (see [serialization.md](./serialization.md)), so the worker process can recompile it against its own Python interpreter. The `__globals__` dict is captured too, including any user variables referenced in the intervention.

The result direction is simpler: saves are zstd-compressed pickled bytes, returned from `collect_nnsight` via `collective_rpc`. `engines/engine.py:7` and `model_runners/GPUModelRunner.py:7` instantiate `_ZSTD_DECOMPRESSOR` / `_ZSTD_COMPRESSOR(level=1)`.

### Tensor parallelism: `VLLMBatcher` gather / scatter

When `tensor_parallel_size > 1`, vLLM shards tensors across GPUs via `ColumnParallelLinear` and `RowParallelLinear`. Intervention code must see complete (unsharded) tensors.

`VLLMBatcher.wrap(model)` (`src/nnsight/modeling/vllm/batching.py:33`) registers four PyTorch hooks on every parallel module:

| Hook | mediator_idx | Purpose |
|------|--------------|---------|
| pre_input_hook | -inf | Track `current_module` and whether input is sharded |
| post_input_hook | +inf | Re-shard input back to per-rank chunks before vLLM resumes |
| pre_output_hook | -inf | Track whether output is sharded |
| post_output_hook | +inf | Re-shard output back |

The `mediator_idx = +/- inf` markers ensure vLLM's gather/scatter wrapping happens **before** any user mediator hook sees the value and **after** all mediator modifications. See [interleaver-internals.md](./interleaver-internals.md) for how `mediator_idx` is sorted.

`VLLMBatcher.check_gathered()` (`src/nnsight/modeling/vllm/batching.py:124`) is invoked from `narrow` / `swap` (overridden at `src/nnsight/modeling/vllm/batching.py:158,164`):

| Layer | Access | Gather op |
|-------|--------|-----------|
| `ColumnParallelLinear` | output | `tensor_model_parallel_all_gather` on last dim |
| `RowParallelLinear` | input | `tensor_model_parallel_all_gather` on last dim |
| `RowParallelLinear` | output | `tensor_model_parallel_all_reduce` then divide by `tp_size` |

Re-sharding in post-hooks is the inverse: split for column-parallel, multiply by `tp_size` for row-parallel.

`VLLMBatcher.wrap` is only called when `get_tp_group().world_size > 1` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:356`). With TP=1, no gather/scatter happens — the wrap is pure overhead.

### Continuous batching: flat → per-request batch groups

vLLM concatenates all tokens from all in-flight requests into a single `[total_tokens, hidden]` tensor. NNsight maps each mediator to a `[start_token, num_tokens]` batch group during the forward pass.

`NNsightRequestHelper.process_batch_groups` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:162`) iterates `self.input_batch.req_ids` (not the scheduler's dict order — this matters because `condense()` and `_may_reorder_batch()` can reorder requests after the scheduler builds `num_scheduled_tokens`). For each request:

- If `num_tokens_scheduled[req_id]` is `None`, skip (request not actually scheduled this step).
- If no mediator is registered, advance `batch_start += num_tokens` and skip.
- Otherwise set `mediator.batch_group = [batch_start, num_tokens]` and advance.

After the forward pass, logits and sampled tokens are per-prompt, not per-token. `unflatten()` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:132`) walks the same `batch_req_ids` ordering and rewrites each mediator's `batch_group` to `[batch_start, 1]` (one row per scheduled request).

Batch groups are recomputed every generation step. A mediator whose request isn't scheduled this step gets `batch_group = None` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:174`), so persistent cache hooks reading the live value report "no slice" rather than out-of-range indices.

### Trace-shared saves: deferred cleanup

A mediator can save a value into `f_locals` (per-invoke) or into a parent-trace-scope variable shared across all invokes. `NNsightRequestHelper.collect_saves` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:261`) handles both:

- Per-invoke: walks `mediator.info.frame.f_locals` and gathers anything in `Globals.saves`.
- Trace-shared: only collected when `received_count == expected_count` and `pending_req_ids` is empty for that `trace_id`. This deferred cleanup avoids premature collection when the scheduler completes one request before another is even scheduled.

`cleanup_finished` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:304`) drops finished mediator entries, removes saved IDs from `Globals.saves`, and deletes completed `trace_contexts`.

## Key files / classes

- `src/nnsight/modeling/vllm/vllm.py:43` — `VLLM` (user-facing class, both processes)
- `src/nnsight/modeling/vllm/vllm.py:349` — `_serialize_mediators` (mediator → `extra_args`)
- `src/nnsight/modeling/vllm/vllm.py:445` — `VLLM.trace` override (async backend injection)
- `src/nnsight/modeling/vllm/sampling.py:4` — `NNsightSamplingParams`
- `src/nnsight/modeling/vllm/batching.py:15` — `VLLMBatcher` (TP gather/scatter + slicing)
- `src/nnsight/modeling/vllm/engines/engine.py:10` — `NNsightLLMEngine` (sync result collection)
- `src/nnsight/modeling/vllm/workers/GPUWorker.py:6` — `NNsightGPUWorker` (monkey-patches model runner)
- `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:29` — `NNsightGPUModelRunner` (forward / logits / sample interleavers)
- `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:38` — `NNsightRequestHelper` (mediator transport, batch groups)
- `src/nnsight/modeling/vllm/async_backend.py:19` — `AsyncVLLMBackend`
- `src/nnsight/modeling/vllm/async_tracer.py` — `AsyncInterleavingTracer`
- `src/nnsight/modeling/vllm/executors/ray_workaround.py` — `LazyRayWorkerWrapper` + `NNsightRayExecutor`
- `src/nnsight/modeling/vllm/README.md` — full architectural narrative
- `src/nnsight/modeling/vllm/DISCUSSION.md`, `IDEAS.md` — design notes and TODOs

## Lifecycle (sync, end-to-end)

1. User: `with model.trace("Hello", temperature=0.0, max_tokens=3): logits = model.logits.save()`.
2. `VLLM.__call__(prompts, params, lora_requests, **kwargs)` — `_serialize_mediators` writes mediators into `params[i].extra_args`.
3. `vllm_entrypoint.generate(prompts, sampling_params=params, ...)` runs vLLM's normal pipeline.
4. vLLM scheduler builds `scheduler_output` and forwards to worker.
5. `NNsightGPUModelRunner._update_states` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:359`) calls `process_new_reqs` (deserializes mediators) and `process_batch_groups` (computes token-level slices).
6. `execute_model` runs `super().execute_model` inside an interleaver context. Mediator threads observe / modify intermediate activations. After return, `unflatten()` switches to prompt-level batch groups.
7. `sample_tokens` provides logits through `model.logits` eproperty (mediators observe / modify).
8. `_sample` runs `super()._sample` inside an interleaver context, then provides `sampled_token_ids` through `model.samples` eproperty.
9. Steps 5–8 repeat per generation step.
10. `NNsightLLMEngine.step` detects finished requests, calls `collect_nnsight(finished_ids, finished_ids)` via `collective_rpc`, attaches `.saves` to `RequestOutput`.
11. Back in `VLLM.__call__`, saves are pushed to the user's frame.

For async, step 2 is replaced by `AsyncVLLMBackend.__call__(tracer)` setting up the request, and steps 10–11 happen on every streamed output via `_stream` / `__aiter__`.

## Extension points

- **New interleaver phase.** Add a method on `NNsightGPUModelRunner` that wraps a vLLM stage with `Globals.enter()` + `with self.nnsight_model.interleaver:` + the call. Use a new eproperty on `VLLM` to expose the value to user code.
- **New eproperty.** Decorate a method on `VLLM` with `@eproperty(description=..., iterate=True)`. Inside the runner, call `type(self.nnsight_model).<name>.provide(self.nnsight_model, value)`.
- **Custom tensor layout (e.g. PP).** Subclass `VLLMBatcher` and override `narrow` / `swap` for the new layout. Note: PP is currently unsupported because `collect_nnsight` only collects from `get_pp_group().rank == 0` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:461`).
- **Different distributed executor.** See `src/nnsight/modeling/vllm/executors/ray_workaround.py` for the Ray pattern. The integration is executor-agnostic at the `worker_cls` / `collective_rpc` level.

## Related

- [`vllm/README.md`](../../src/nnsight/modeling/vllm/README.md) — canonical architectural narrative
- [`vllm/DISCUSSION.md`](../../src/nnsight/modeling/vllm/DISCUSSION.md) — design rationale notes
- [`vllm/IDEAS.md`](../../src/nnsight/modeling/vllm/IDEAS.md) — open ideas / TODOs
- [batching-internals.md](./batching-internals.md) — `VLLMBatcher` in the broader batching context
- [serialization.md](./serialization.md) — how mediators survive process boundaries
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — using vLLM as a reference for new runtimes
