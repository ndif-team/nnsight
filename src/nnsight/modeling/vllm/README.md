# vLLM Integration

This document details the design and implementation of NNsight's vLLM integration. It is written for contributors working on this code and assumes familiarity with NNsight's core concepts (tracing, interleaving, mediators, envoys) but is otherwise self-contained.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Key Classes](#key-classes)
4. [Execution Flow](#execution-flow)
5. [Model Loading](#model-loading)
6. [Mediator Transport via extra_args](#mediator-transport-via-extra_args)
7. [Batch Group Management](#batch-group-management)
8. [Multiple Interleaving Phases](#multiple-interleaving-phases)
9. [Tensor Parallelism](#tensor-parallelism)
10. [Continuous Batching](#continuous-batching)
11. [Multi-Token Generation](#multi-token-generation)
12. [Async Engine](#async-engine)
13. [Ray Distributed Executor](#ray-distributed-executor)
14. [Multi-Node Support](#multi-node-support)

---

## Overview

The vLLM integration enables NNsight interventions (observing and modifying intermediate activations) on models served through vLLM's high-performance inference engine. This is one of the most complex integrations in NNsight because vLLM's architecture differs substantially from standard PyTorch model execution:

- **Separate processes**: vLLM runs model execution in worker processes, not the user's process. Intervention code must be serialized and transported across process boundaries.
- **Flat tensor format**: vLLM concatenates all tokens from all prompts into a single `[total_tokens, hidden]` tensor rather than the standard `[batch, tokens, hidden]` format.
- **Continuous batching**: Requests can join and leave the batch between generation steps.
- **Tensor parallelism**: When using multiple GPUs, tensors are sharded. Intervention code must see complete, unsharded tensors.
- **Phased execution**: The forward pass, logit computation, and sampling are separate stages that NNsight must hook into independently.

The integration solves each of these by subclassing vLLM's engine, worker, and model runner classes, injecting NNsight's interleaving machinery at key points.

---

## File Structure

```
vllm/
├── __init__.py                    # Exports VLLM class
├── vllm.py                        # VLLM model wrapper (user-facing class)
├── async_tracer.py                # AsyncInterleavingTracer — deferred execution for async engine
├── async_backend.py               # AsyncVLLMBackend — dual-call backend that streams via AsyncLLM
├── sampling.py                    # NNsightSamplingParams — thin SamplingParams subclass
├── batching.py                    # VLLMBatcher — tensor-parallel gather/split + flat-batch slicing
├── engines/
│   ├── __init__.py
│   └── engine.py                  # NNsightLLMEngine — collects saved results after requests finish (sync)
├── executors/
│   ├── __init__.py
│   └── ray_workaround.py          # LazyRayWorkerWrapper + NNsightRayExecutor for Ray support
├── workers/
│   ├── __init__.py
│   └── GPUWorker.py               # NNsightGPUWorker — monkey-patches model runner at init
├── model_runners/
│   ├── __init__.py
│   └── GPUModelRunner.py           # NNsightGPUModelRunner — core: interleaves interventions with vLLM execution
└── examples/
    └── multi_node_with_ray/        # Docker-based multi-node Ray example and tests
        ├── README.md
        ├── Dockerfile
        ├── docker-compose.yml
        └── test_multinode.py
```

### File Responsibilities

**`vllm.py`** — The `VLLM` class that users instantiate. Handles:
- Meta/real model loading via mixin inheritance (`RemoteableMixin -> MetaMixin -> LoadableMixin -> NNsight`)
- Input preparation (`_prepare_input`) — normalizes strings, token ID lists, and HuggingFace tokenizer dicts
- Batching multiple invokes together (`_batch`)
- Forwarding calls to the vLLM engine (`__call__`)
- Creating wrapper modules for `logits`, `samples`, and `generator`
- Automatic Ray executor substitution when `distributed_executor_backend="ray"`
- `trace()` override that injects `AsyncVLLMBackend` and `AsyncInterleavingTracer` when `mode="async"`

**`async_tracer.py`** — `AsyncInterleavingTracer` extends `RemoteInterleavingTracer`. Overrides `execute()` to serialize mediators into sampling params and store the prepared `(prompts, params, kwargs)` on the tracer instance (instead of running synchronous generation). The `AsyncVLLMBackend` reads this prepared data after the trace context exits.

**`async_backend.py`** — `AsyncVLLMBackend` extends `Backend` with a dual-call pattern:
- `__call__(tracer)`: Called from `__exit__`, compiles and executes the traced function to prepare generation data.
- `__call__()`: Called by user code, returns an async generator that streams `RequestOutput` objects from `AsyncLLM`. On each streamed output, calls `collect_nnsight` via `collective_rpc` to retrieve current saves from the worker. Saves are attached as `output.saves` on every output (not just the final one).

**`sampling.py`** — `NNsightSamplingParams` is a thin subclass of vLLM's `SamplingParams` used for type identification in `_prepare_input`. Mediator data is transported via the built-in `extra_args` dict field on `SamplingParams`, not on a custom field.

**`batching.py`** — `VLLMBatcher` extends NNsight's base `Batcher` to handle tensor parallelism. Registers pre/post hooks on all modules to track which module is currently executing and whether its tensors are sharded. When intervention code requests a value, the batcher transparently gathers sharded tensors; when intervention code returns a modified value, the batcher re-shards before passing back to vLLM.

**`engines/engine.py`** — `NNsightLLMEngine` extends vLLM's `LLMEngine`. Used by the sync path only. After each engine step, checks for finished requests and calls `collect_nnsight()` on the model executor to collect saved intervention results.

**`workers/GPUWorker.py`** — `NNsightGPUWorker` extends vLLM's `Worker`. Its only job is to monkey-patch `GPUModelRunner` with `NNsightGPUModelRunner` before vLLM's init runs, and to expose `collect_nnsight()`.

**`model_runners/GPUModelRunner.py`** — `NNsightGPUModelRunner` is the core of the integration. It:
- Creates a second `VLLM` wrapper around the model loaded by vLLM (inside the worker process)
- Deserializes mediators from incoming requests via `extra_args`
- Manages batch group mappings (flat token-level during forward, prompt-level after)
- Enters the interleaver at three phases: forward pass, logit wrapping, and sampling
- Collects saved values via `collect_nnsight()`, which delegates to helper methods on `NNsightRequestHelper`: `match_req_ids()`, `finalize_mediators()`, `collect_saves()`, `cleanup_finished()`

**`executors/ray_workaround.py`** — Contains `LazyRayWorkerWrapper` and `NNsightRayExecutor` for Ray distributed executor support. See [Ray Distributed Executor](#ray-distributed-executor) for details.

**`examples/multi_node_with_ray/`** — Docker-based example for multi-node tensor parallelism with Ray. Includes a Dockerfile, docker-compose config, test script, and detailed README. See [Multi-Node Support](#multi-node-support) for details.

---

## Key Classes

### VLLM (vllm.py)

The user-facing class. Exists in two contexts:

1. **User process**: Created by the user (`model = VLLM("gpt2", dispatch=True)`). Handles tracing, input preparation, and dispatching to the vLLM engine.
2. **Worker process**: Created by `NNsightGPUModelRunner.load_model()` to wrap the model that vLLM loaded. This instance has the interleaver and batcher attached.

Key attributes:
- `vllm_entrypoint` — The actual `vllm.LLM` or `AsyncLLM` instance (user process only)
- `tokenizer` — vLLM's tokenizer
- `logits` — `WrapperModule` envoy for intercepting logits
- `samples` — `WrapperModule` envoy for intercepting sampled tokens
- `generator` — `WrapperModule` envoy for generation output
- `_async_engine` — Boolean flag (derived from `mode="async"`); when `True`, `trace()` injects async backend/tracer

### AsyncInterleavingTracer (async_tracer.py)

Custom tracer for the async path. Extends `RemoteInterleavingTracer` and overrides `execute()` to prepare generation data without triggering synchronous generation. The key difference from the sync path: instead of calling `model.interleave()` (which runs the full generate loop), it serializes mediators into sampling params and stores the result as `self.prepared`. The `AsyncVLLMBackend` reads this after the trace context exits.

The `VLLM.trace()` override bypasses `RemoteableMixin.trace()` (which hard-codes `tracer_cls=RemoteInterleavingTracer`) by calling `Envoy.trace()` directly with `tracer_cls=AsyncInterleavingTracer`.

### AsyncVLLMBackend (async_backend.py)

Backend with a dual-call pattern:
- **First call** `__call__(tracer)`: Invoked by `Tracer.__exit__`. Compiles the user's intervention code and calls `tracer.execute(fn)` to set up mediators and prepare generation data.
- **Second call** `__call__()`: Invoked by user code via `tracer.backend()`. Returns an async generator (`_stream()`) that submits to `AsyncLLM.generate()` and yields `RequestOutput` objects with saves attached.

On every streamed output, `_stream()` calls `collect_nnsight` via `collective_rpc` to retrieve the current saves from the worker. When the request finishes, the worker also finalizes the mediator (runs result handler, cancels, cleans up).

### NNsightSamplingParams (sampling.py)

Thin subclass of `vllm.SamplingParams`. The mediator is not stored as a field on this class — instead, serialized mediator bytes are placed in the built-in `SamplingParams.extra_args` dict, which survives vLLM's internal msgpack serialization in multiprocessing mode.

### VLLMBatcher (batching.py)

Extends NNsight's `Batcher`. Handles two concerns:

1. **Batch slicing**: `narrow(batch_group)` extracts a mediator's slice from the flat batch; `swap(batch_group, value)` puts a modified value back. When `batch_group` is `None` (empty invoke), the full batch is returned/replaced.

2. **Tensor parallelism**: Tracks the current module and whether its tensors are sharded. `check_gathered()` gathers sharded tensors before intervention code sees them. Post-hooks re-shard after intervention.

### NNsightGPUModelRunner (model_runners/GPUModelRunner.py)

The most complex class. Contains an inner `NNsightRequestHelper` that manages:
- Deserializing mediators from new requests' `extra_args`
- Mapping request IDs to batch groups (token-level start position and count)
- Switching batch groups from flat (token-level) to unflattened (prompt-level) after the forward pass

Key methods:
- `load_model()` — Creates the worker-side `VLLM` wrapper and `VLLMBatcher`
- `_update_states(scheduler_output)` — Processes new/finished requests, updates batch groups
- `execute_model(scheduler_output, ...)` — Runs the forward pass inside an interleaver context, wraps logits
- `_sample()` — Runs sampling inside an interleaver context, wraps sampled tokens
- `collect_nnsight(req_ids, finished_req_ids)` — Collects saves from mediators. Called on every streamed output (async) or on finished requests (sync). Delegates to `NNsightRequestHelper` helper methods:
  - `match_req_ids()` — matches engine-reported IDs to stored mediators (handles vLLM's hash suffix via `rsplit`)
  - `finalize_mediators()` — runs result handler and cancels finished mediators
  - `collect_saves()` — gathers per-invoke saves from frame locals and trace-shared saves from canonical globals
  - `cleanup_finished()` — removes from `Globals.saves`, deletes completed trace contexts, drops mediator entries

### NNsightLLMEngine (engines/engine.py)

Thin extension of vLLM's engine. Used by the sync path only. After each `step()`, checks for finished requests and delegates to `collect_nnsight()` on the executor to gather saved results. In the async path, `collect_nnsight` is called directly by the `AsyncVLLMBackend` via `collective_rpc` on every streamed output.

### NNsightGPUWorker (workers/GPUWorker.py)

Thin extension of vLLM's worker. Monkey-patches the model runner class before init, and exposes `collect_nnsight()` which delegates to the model runner.

### NNsightRayExecutor (executors/ray_workaround.py)

Custom `RayDistributedExecutor` subclass passed as `distributed_executor_backend` when Ray is requested. Swaps in `LazyRayWorkerWrapper` before creating Ray actors, and handles connecting to existing Ray clusters (including remote ones). See [Ray Distributed Executor](#ray-distributed-executor).

---

## Input Model: One Prompt Per Invoke

### Why One Prompt Per Invoke?

Unlike `LanguageModel` where a single invoke can contain a batch of prompts (`tracer.invoke(["Hello", "World"])`), vLLM requires **exactly one prompt per invoke**.

This is an architectural constraint, not a limitation of NNsight. In vLLM, each prompt becomes an independent **request** with its own:
- Request ID and lifecycle (scheduled, running, finished independently)
- Scheduling decisions (may be chunked, preempted, or batched differently per step)
- Sampling parameters (temperature, top_p, max_tokens, stop conditions)
- Finish condition (each request stops independently when it hits its stop token or max_tokens)

Each invoke produces one `Mediator`, and each mediator maps 1:1 to a `SamplingParams` which maps 1:1 to a vLLM request. If one invoke could contain multiple prompts, a single mediator would need to track multiple independent requests that may be scheduled at different times, finish at different times, and have different token counts per step — breaking batch group management entirely.

`_prepare_input()` enforces this and raises `ValueError` if multiple prompts are passed to a single invoke.

### Batching Pattern

Use a loop of invokes to process multiple prompts. Each invoke runs its intervention code independently, but all prompts are batched together by vLLM's engine for efficient execution:

```python
prompts = ["Prompt A", "Prompt B", "Prompt C"]

with model.trace(max_tokens=512) as tracer:
    # Shared state defined at trace scope
    out_ids = [list() for _ in range(len(prompts))].save()

    for i, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            # Each invoke = one prompt = one vLLM request
            with tracer.all():
                out_ids[i].append(model.samples.output.item())

# Access results after the trace
for i, ids in enumerate(out_ids):
    print(model.tokenizer.decode(ids))
```

Key points:
- **Each `tracer.invoke(prompt)` becomes one vLLM request** — vLLM batches them internally for GPU efficiency
- **Shared state across invokes** (like `out_ids` above) works via globals grafting on the worker (see [Cross-Invoke Shared State](#cross-invoke-shared-state))
- **Per-invoke sampling params**: pass kwargs to individual invokes to control each request independently (e.g., `tracer.invoke(prompt, temperature=0.8)`)
- **`tracer.all()`** applies the intervention body to every generation step (equivalent to `tracer.iter[:]` but recursive)
- **Empty invokes** (`tracer.invoke()` with no args) still work — they see the full batch across all requests, useful for batch-wide observations

---

## Execution Flow

### End-to-End: From User Trace to Saved Values

**1. User enters trace context:**
```python
with model.trace("Hello", temperature=0.0, max_tokens=3) as tracer:
    logits = model.logits.output.save()
```

NNsight captures, parses, and compiles the intervention code into a `Mediator`.

**2. `VLLM.__call__()` is invoked:**
- `_prepare_input()` normalizes the input (tokenizes strings, etc.)
- `_batch()` combines inputs from all invokes
- Computes `saved_names` (parent-scope variables in `Globals.saves`) and generates a `trace_id`
- Each invoke's mediator is serialized independently via `serialize()` and stored in `param.extra_args` with trace metadata (`nnsight_mediator`, `nnsight_trace_id`, `nnsight_trace_idx`, `nnsight_saved_names`, `nnsight_expected_count`)
- `vllm_entrypoint.generate(prompts, sampling_params)` is called

**3. vLLM schedules the request:**
- The engine passes the request through its scheduler
- The worker's `_update_states()` is called with the scheduler output

**4. `NNsightGPUModelRunner._update_states()`:**
- Calls `process_new_reqs()` — deserializes each mediator from `extra_args["nnsight_mediator"]`; for the first mediator of a `trace_id`, stores its `__globals__` as canonical; subsequent mediators get shared variables grafted from the canonical globals
- Calls `process_batch_groups()` — computes each mediator's `[start_token, num_tokens]` batch group based on scheduled token counts
- Registers mediators with the interleaver

**5. `NNsightGPUModelRunner.execute_model()`:**
- Enters `Globals` context (NNsight thread-local state)
- Enters interleaver context (`with self.nnsight_model._interleaver:`)
  - This starts mediator worker threads
- Calls `super().execute_model()` — vLLM's forward pass runs, module hooks fire, mediators interleave
- After forward pass: calls `unflatten()` to switch batch groups from token-level to prompt-level
- Wraps logits through `model.logits(logits, hook=True)` — mediators can observe/modify logits
- Updates `execute_model_state` with the (potentially modified) logits

**6. `NNsightGPUModelRunner._sample()`:**
- Enters `Globals` context and interleaver context
- Calls `super()._sample()` — vLLM samples next tokens
- Wraps sampled token IDs through `model.samples(token_ids, hook=True)` — mediators can observe/modify samples

**7. Steps 4-6 repeat for each generation step** (if `max_tokens > 1`).

**8. When requests finish (sync path):**
- `NNsightLLMEngine.step()` detects finished requests
- Calls `collect_nnsight(finished_req_ids, finished_req_ids)` on the executor -> worker -> model runner
- Model runner finalizes mediators (result handler, cancel), extracts saved values, cleans up
- Returns saves dict (pickled to bytes so it survives msgpack transport in multiprocessing mode), which gets attached to the `RequestOutput`

**8. On every streamed output (async path):**
- `AsyncVLLMBackend._stream()` calls `collect_nnsight(req_ids, finished_req_ids)` via `collective_rpc` on every output
- When `finished_req_ids` is `None` (intermediate outputs): saves are collected but mediators are not finalized
- When `finished_req_ids` contains the request ID (final output): mediators are also finalized and cleaned up
- Saves are attached as `output.saves` on every `RequestOutput`

**9. Back in user process (sync):**
- `VLLM.__call__()` receives the `RequestOutput` with attached saves
- Saved values are pushed back into the user's local variables

**9. Back in user process (async):**
- User iterates `async for output in tracer.backend()` to receive streamed outputs
- Each output has `.saves` with the current saved values

---

## Model Loading

The `VLLM` class uses `MetaMixin` for lazy/eager loading.

### Meta Loading (`_load_meta`)

When `dispatch=False` (default), the model is loaded with meta tensors (no real weights allocated). This uses vLLM's `DummyModelLoader` with `device="meta"`. The purpose is to build the Envoy tree (module hierarchy) so users can write intervention code referencing `model.transformer.h[0].output` etc. without allocating GPU memory.

### Real Loading (`_load`)

When `dispatch=True` or when `interleave()` auto-dispatches:
- Destroys any existing distributed environment
- If `distributed_executor_backend="ray"`, replaces it with `NNsightRayExecutor` class (see [Ray Distributed Executor](#ray-distributed-executor))
- If `mode="async"`: creates an `AsyncLLM` via `AsyncLLM.from_engine_args()` with `AsyncEngineArgs`. Pre-initializes Ray if using Ray backend.
- If `mode="sync"` (default):
  - Creates a `vllm.LLM` instance with `enforce_eager=True`
  - After creation, monkey-patches the engine class to `NNsightLLMEngine`
- Both paths use `worker_cls="nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"`

### Worker-Side Loading

Inside the worker process, `NNsightGPUModelRunner.load_model()`:
- Calls vLLM's normal `load_model()` (loads real weights)
- Creates a new `VLLM` wrapper around the loaded model
- Creates a `VLLMBatcher` and attaches it to the interleaver
- Calls `batcher.wrap(model)` to register tensor-parallelism hooks on all modules

This means there are **two VLLM instances**: one in the user process (for tracing/input prep) and one in the worker process (for interleaving).

---

## Mediator Transport via extra_args

The core challenge: intervention code is compiled into a `Mediator` in the user process, but must execute in the worker process. Additionally, multiple invokes within a single trace may share parent-scope variables (e.g., a saved list that each invoke appends to).

### How It Works

1. During tracing, each invoke produces a `Mediator` containing the compiled intervention function.
2. `VLLM.__call__()` computes `saved_names` — parent-frame variable names whose values are in `Globals.saves` (i.e., variables defined at trace scope with `.save()`). It also generates a `trace_id` UUID to group all mediators from the same trace.
3. Each mediator is serialized independently via `serialize()` and stored in `param.extra_args` along with trace metadata:
   ```python
   param.extra_args = {
       "nnsight_mediator": serialize(mediator),  # per-mediator bytes
       "nnsight_trace_id": trace_id,             # groups mediators from same trace
       "nnsight_trace_idx": idx,                 # ordering within the trace
       "nnsight_saved_names": saved_names,        # shared variable names
       "nnsight_expected_count": count,           # total mediators in this trace
   }
   ```
4. `SamplingParams.extra_args` is a built-in `dict[str, Any] | None` field that survives vLLM's internal msgpack serialization when passing to worker processes (both multiprocessing and Ray).
5. In the worker, `process_new_reqs()` deserializes each mediator independently. The first mediator to arrive for a given `trace_id` has its `__globals__` stored as the canonical reference. Subsequent mediators for the same trace have the saved variable entries in their `__globals__` replaced with references from the canonical globals, so all mediators share the same Python objects for cross-invoke state.

### Cross-Invoke Shared State

When users define variables in the parent trace scope and reference them inside multiple invokes (e.g., a shared list that each invoke appends to), the worker-side globals grafting ensures all mediators operate on the same Python objects:

```python
with model.trace(max_tokens=3) as tracer:
    out_ids = [list() for i in range(len(prompts))].save()  # parent-scope saved var
    for i, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            with tracer.all():
                out_ids[i].append(model.logits.output)  # mutates shared list
```

On the worker:
- Mediator 0 arrives first; its `__globals__["out_ids"]` becomes the canonical object
- Mediator 1 arrives later; its `__globals__["out_ids"]` is replaced with mediator 0's copy
- Both mediators now mutate the same list in-place
- Shared saves are collected only after ALL mediators for the trace have been received and completed (`received_count == expected_count` and `pending_req_ids` is empty), preventing premature cleanup if the scheduler completes one request before another is even scheduled

### Why extra_args?

vLLM already passes `SamplingParams` through its entire pipeline — from engine to scheduler to worker to model runner. The `extra_args` dict is a built-in field on `SamplingParams` that survives all serialization boundaries (pickle for Ray, msgpack for multiprocessing). This avoids creating a separate transport mechanism and doesn't require a custom `__reduce__()` implementation.

---

## Batch Group Management

### The Problem: Flat Tensor Format

Standard NNsight uses `[batch, tokens, hidden]` tensors. vLLM concatenates all tokens into a flat `[total_tokens, hidden]` tensor for efficiency.

```
Standard NNsight:
  Prompt "Hello World" (5 tokens):  [1, 5, 768]  -> batch_group = [0, 1]
  Prompt "Hi" (2 tokens):           [1, 2, 768]  -> batch_group = [1, 1]

vLLM (flat):
  All tokens concatenated:          [7, 768]      -> batch_group = [0, 5] for "Hello World"
                                                     batch_group = [5, 2] for "Hi"
```

### Token-Level vs Prompt-Level Batch Groups

During the forward pass, batch groups are **token-level**: `[start_token_index, num_tokens]`. This allows `narrow()` to slice the correct tokens for each invoke's intervention code.

After the forward pass (for logits and sampling), batch groups switch to **prompt-level**: `[start_prompt_index, num_prompts]`. This is because logits and sampled tokens are per-prompt, not per-token.

`NNsightRequestHelper` manages this transition:
- `process_batch_groups()` computes token-level batch groups from the scheduler's `num_scheduled_tokens`
- `unflatten()` switches to prompt-level batch groups after the forward pass

### Batch Group Updates Per Step

Because vLLM uses continuous batching, batch groups are recomputed every generation step via `process_batch_groups()`. The scheduler may schedule different numbers of tokens per request at each step (e.g., full prompt on prefill, single token on decode).

---

## Multiple Interleaving Phases

vLLM separates execution into distinct stages. NNsight enters the interleaver at each:

### Phase 1: Forward Pass (`execute_model`)

The interleaver context wraps `super().execute_model()`. Module hooks fire as the model runs, and mediator threads interleave to observe/modify intermediate activations. Batch groups are token-level during this phase.

After the forward pass completes, `unflatten()` switches batch groups to prompt-level.

### Phase 2: Logits

Still inside the same `execute_model()` call, logits are wrapped through `model.logits(logits, hook=True)`. This fires the logits envoy's hooks, letting mediators observe/modify logits before sampling. The user accesses this as `model.logits.output`.

### Phase 3: Sampling (`_sample`)

A separate interleaver context wraps `super()._sample()`. After sampling, the sampled token IDs are wrapped through `model.samples(token_ids, hook=True)`. The user accesses this as `model.samples.output`.

### Phase 4: Collect (`collect_nnsight`)

When requests complete (sync) or on every streamed output (async), `collect_nnsight` is called. For finished requests, the interleaver handles the `"result"` provider, letting mediators interact with the final generation output. Saved values are extracted from mediator frames and trace-shared globals. The method delegates to helper methods on `NNsightRequestHelper` for matching, finalizing, collecting, and cleanup.

### Shared Mediator Threads

The same mediator threads persist across all phases within a generation step. The interleaver context is entered/exited multiple times, but mediator threads are not restarted — they continue waiting for the next value from wherever they left off in the user's intervention code.

---

## Tensor Parallelism

When `tensor_parallel_size > 1`, vLLM shards tensors across GPUs using `ColumnParallelLinear` and `RowParallelLinear` layers. Intervention code must see complete, unsharded tensors.

### VLLMBatcher's Role

`VLLMBatcher.wrap(model)` registers PyTorch hooks (not NNsight hooks) on every module:

- **Pre-input hooks**: Track the current module and whether its input is sharded (`RowParallelLinear` with `input_is_parallel=True`)
- **Pre-output hooks**: Track whether the output is sharded (`ColumnParallelLinear` with `gather_output=False`)

When a mediator requests a value via `narrow()`, `check_gathered()` is called first:

| Layer Type | Access | Gather Operation |
|------------|--------|------------------|
| `ColumnParallelLinear` | output | `tensor_model_parallel_all_gather()` on last dim |
| `RowParallelLinear` | input | `tensor_model_parallel_all_gather()` on last dim |
| `RowParallelLinear` | output | `tensor_model_parallel_all_reduce()` then divide by `tp_size` |

After intervention code runs and returns (potentially modified) values, post-hooks re-shard:

| Layer Type | Access | Re-shard Operation |
|------------|--------|--------------------|
| `ColumnParallelLinear` | output | `split_tensor_along_last_dim` -> take `tp_rank` shard |
| `RowParallelLinear` | output | Divide by `tp_size` (to undo the all-reduce) |

Every GPU runs the **same intervention code** on the **same complete tensor**, ensuring consistency across the distributed system.

---

## Continuous Batching

vLLM uses continuous batching: new requests can join and finished requests can leave the batch between generation steps.

### How NNsight Handles This

**New requests**: `process_new_reqs()` deserializes each mediator from `extra_args["nnsight_mediator"]`. The first mediator for a `trace_id` establishes the canonical `__globals__`; subsequent mediators have their shared variables grafted from the canonical copy (see [Cross-Invoke Shared State](#cross-invoke-shared-state)).

**Batch group updates**: `process_batch_groups()` recomputes batch groups every step based on what the scheduler has actually scheduled. Only currently-scheduled requests are reflected in batch groups.

**Finished requests**: `collect_nnsight()` handles two kinds of saves:
1. **Per-invoke saves**: Variables `.save()`-ed inside an invoke are collected from each mediator's `info.frame.f_locals`.
2. **Trace-shared saves**: Variables `.save()`-ed at trace scope are collected from the canonical `__globals__` only after ALL mediators for the trace have been received and completed (`received_count == expected_count` and `pending_req_ids` is empty).

The deferred cleanup prevents premature collection when the scheduler completes one request before another is even scheduled. In the sync path, `NNsightLLMEngine.step()` attaches saves to all finished request outputs. In the async path, saves are collected on every streamed output — intermediate outputs collect saves without finalizing mediators, while the final output also runs finalization and cleanup.

---

## Multi-Token Generation

When `max_tokens > 1`, the execute/sample cycle repeats for each token:

```
Step 0 (prefill):
  _update_states() -> process new requests, compute batch groups
  execute_model()  -> forward pass, wrap logits
  _sample()        -> sample, wrap samples

Step 1 (decode):
  _update_states() -> update batch groups (now 1 token per request)
  execute_model()  -> forward pass, wrap logits
  _sample()        -> sample, wrap samples

Step 2 (decode):
  ...same pattern...

All requests in group finish:
  collect_nnsight() -> collect saves, finalize, cleanup
```

### Iteration Tracking

The interleaver appends iteration suffixes to provider strings: `model.layer.output.i0`, `.i1`, `.i2`, etc. This disambiguates the same module being called multiple times across generation steps.

In user code, `tracer.iter[:]` or `tracer.iter[0:3]` iterates over generation steps:

```python
with model.trace("Hello", max_tokens=3) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.logits.output)
```

Each iteration of the loop corresponds to one generation step. The mediator's iteration counter advances, matching the interleaver's provider iteration suffix.

### Batch Group Differences: Prefill vs Decode

During prefill (step 0), a request's batch group covers all prompt tokens: `[start, num_prompt_tokens]`. During decode (steps 1+), it covers a single token: `[start, 1]`. The `unflatten()` call after the forward pass normalizes these back to prompt-level regardless.

---

## Async Engine

The async engine enables streaming token-by-token output with NNsight interventions using vLLM's `AsyncLLM`. This is useful for chat applications and any scenario where you want to process results incrementally rather than waiting for the full generation to complete.

### Usage

```python
from nnsight.modeling.vllm import VLLM
import asyncio

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True, mode="async")

async def main():
    with model.trace("The Eiffel Tower is in", temperature=0.0, max_tokens=5) as tracer:
        logits = model.logits.output.save()

    async for output in tracer.backend():
        print(f"finished={output.finished}, saves={list(output.saves.keys())}")

asyncio.run(main())
```

### Architecture

The async path introduces three new components that work together to defer generation to after the trace context exits:

```
┌─────────────────────────────────────────────────────────────────┐
│  User Code                                                      │
│  with model.trace("Hello", ...) as tracer:                      │
│      logits = model.logits.output.save()                        │
│                                                                 │
│  async for output in tracer.backend():  # <-- streaming here    │
│      print(output.saves)                                        │
└──────┬──────────────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────────────┐
│  VLLM.trace() override                                          │
│  - Detects mode="async"                                         │
│  - Injects AsyncVLLMBackend and AsyncInterleavingTracer         │
│  - Bypasses RemoteableMixin.trace() → calls Envoy.trace()       │
└──────┬──────────────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────────────┐
│  AsyncInterleavingTracer.execute()                              │
│  - Runs compiled user code to set up mediators                  │
│  - Serializes mediators into sampling params                    │
│  - Stores prepared (prompts, params, kwargs) on self.prepared   │
│  - Does NOT call model.interleave() (no sync generation)        │
└──────┬──────────────────────────────────────────────────────────┘
       │
       v
┌─────────────────────────────────────────────────────────────────┐
│  AsyncVLLMBackend                                               │
│  __call__(tracer): Reads tracer.prepared                        │
│  __call__():       Returns _stream() async generator            │
│                                                                 │
│  _stream():                                                     │
│    async for output in AsyncLLM.generate(...):                  │
│      collective_rpc("collect_nnsight", ...)  # get saves        │
│      output.saves = saves                                       │
│      yield output                                               │
└─────────────────────────────────────────────────────────────────┘
```

### How It Differs from the Sync Path

| Aspect | Sync | Async |
|--------|------|-------|
| Engine class | `vllm.LLM` | `vllm.v1.engine.async_llm.AsyncLLM` |
| Tracer | `RemoteInterleavingTracer` | `AsyncInterleavingTracer` |
| Backend | Default (runs `model.interleave()`) | `AsyncVLLMBackend` |
| Generation trigger | `VLLM.__call__()` via `model.interleave()` | `AsyncLLM.generate()` via `tracer.backend()` |
| Save collection | On finished requests only (`NNsightLLMEngine.step()`) | On every streamed output via `collective_rpc` |
| Engine patching | `NNsightLLMEngine` replaces engine class | No engine patching (async path uses `collective_rpc` directly) |
| Result delivery | Saves pushed to user's local variables | Saves attached as `output.saves` on each `RequestOutput` |
| Ray support | Yes (executor swap in `_load()`) | Yes (executor swap + pre-`ray.init()` in `_load()`) |

### Streaming Saves

A key design choice: saves are collected on **every** streamed output, not just when the request finishes. This enables real-time monitoring of intervention state during generation.

The `collect_nnsight` method on the worker accepts two parameters:
- `req_ids`: All request IDs to collect current saves from
- `finished_req_ids`: Subset that are finished and should be finalized

For intermediate outputs (not finished), `finished_req_ids` is `None` — saves are collected from frame locals but the mediator is not finalized and its saves are not removed from `Globals.saves` (so they can be re-collected on the next step). For the final output, the mediator is finalized (result handler + cancel) and cleaned up.

### Why AsyncInterleavingTracer Bypasses RemoteableMixin

`RemoteableMixin.trace()` hard-codes `tracer_cls=RemoteInterleavingTracer`. The async path needs `AsyncInterleavingTracer` instead, so `VLLM.trace()` bypasses it by calling `Envoy.trace()` directly:

```python
def trace(self, *inputs, **kwargs):
    if self._async_engine and kwargs.get('backend') is None and not kwargs.get('remote'):
        kwargs['backend'] = AsyncVLLMBackend(self)
        return Envoy.trace(self, *inputs, tracer_cls=AsyncInterleavingTracer, **kwargs)
    return super().trace(*inputs, **kwargs)
```

This preserves the `remote=True` path unchanged (still uses `RemoteInterleavingTracer`).

---

## Ray Distributed Executor

vLLM supports multiple executor backends for distributing tensor-parallel workers across GPUs. The default is `"mp"` (multiprocessing), which spawns workers as local subprocesses. The `"ray"` backend uses Ray actors instead, enabling multi-node inference where TP workers can run on different machines.

### The Problem

vLLM v0.15.1 + Ray 2.53.0 have a compatibility issue where Ray actor processes crash during construction. When Ray creates a `RayWorkerWrapper` actor, it imports the module `vllm.v1.executor.ray_utils` in the actor process. This triggers transitive module-level imports:

```
ray_utils.py
  -> worker_base.py
    -> from vllm.multimodal import MULTIMODAL_REGISTRY
      -> (heavy initialization of multimodal registries, torch ops, etc.)
```

These imports conflict with Ray's internal gRPC event engine (specifically grpcio's `cygrpc` C extension) during the actor construction phase, causing the actor process to die with a segfault before it is fully constructed. There is no Python traceback — the crash occurs at the C level.

The same imports work fine when they happen during actor **method execution** (after the actor is fully constructed and Ray's gRPC connection is stable).

### The Fix: LazyRayWorkerWrapper + NNsightRayExecutor

`LazyRayWorkerWrapper` is a thin drop-in replacement for `RayWorkerWrapper` that has **no heavy module-level imports**. It defers all vLLM imports to `__init__` time, which runs during actor method execution rather than actor construction:

```python
class LazyRayWorkerWrapper:
    def __init__(self, *args, **kwargs):
        # This import happens AFTER actor construction,
        # when Ray's gRPC connection is stable.
        from vllm.v1.executor.ray_utils import RayWorkerWrapper
        self._w = RayWorkerWrapper(*args, **kwargs)

    # All methods explicitly defined (Ray actors can't use __getattr__)
    def execute_method(self, method, *args, **kwargs):
        return self._w.execute_method(method, *args, **kwargs)
    # ... etc
```

`NNsightRayExecutor` is a subclass of `RayDistributedExecutor` that swaps in `LazyRayWorkerWrapper` and handles Ray cluster initialization before creating workers:

```python
class NNsightRayExecutor(RayDistributedExecutor):
    def _init_executor(self) -> None:
        import os, ray, subprocess
        import vllm.v1.executor.ray_utils as ray_utils
        import vllm.v1.executor.ray_executor as ray_exec

        # Swap in lazy wrapper to avoid actor crash
        ray_utils.RayWorkerWrapper = LazyRayWorkerWrapper
        ray_exec.RayWorkerWrapper = LazyRayWorkerWrapper
        self.forward_dag = None

        # Three-way Ray initialization:
        if not ray.is_initialized():
            ray_address = os.environ.get("RAY_ADDRESS")
            try:
                ray.init(address="auto")           # (1) local Ray already running
            except (ConnectionError, ValueError, RuntimeError):
                if ray_address:
                    subprocess.run(                 # (2) join remote cluster as driver-only node
                        ["ray", "start", f"--address={ray_address}",
                         "--num-gpus=0", "--num-cpus=0"],
                        check=True, capture_output=True,
                    )
                    ray.init(address="auto")
                else:
                    ray.init()                      # (3) start fresh local cluster

        # ... placement group creation, VLLM_HOST_IP fix, _init_workers_ray ...
```

### How It's Integrated

When the user passes `distributed_executor_backend="ray"`, `VLLM._load()` replaces the string with the `NNsightRayExecutor` class before passing it to both sync (`LLM`) and async (`AsyncLLM`) entrypoints:

```python
_uses_ray = kwargs.get("distributed_executor_backend") == "ray"
if _uses_ray:
    from .executors.ray_workaround import NNsightRayExecutor
    kwargs["distributed_executor_backend"] = NNsightRayExecutor

if self._async_engine:
    # AsyncLLM spawns EngineCore in a subprocess. Pre-initialize Ray
    # so the subprocess can connect via ray.init(address="auto").
    if _uses_ray:
        import ray
        if not ray.is_initialized():
            ray.init()
    # ... create AsyncLLM ...
else:
    # ... create LLM ...
```

vLLM's `EngineArgs.distributed_executor_backend` accepts `str | type[Executor]`, so passing a class directly is supported. This is cleaner than external monkey-patching because:

1. **Works with multiprocessing mode**: vLLM pickles the executor class to the EngineCore subprocess. `NNsightRayExecutor._init_executor()` runs inside that subprocess, where it swaps in `LazyRayWorkerWrapper` before any Ray actors are created. No need to force `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
2. **Self-contained**: The workaround is entirely within `NNsightRayExecutor` — no global state or env var overrides.
3. **Transparent to users**: `VLLM("gpt2", distributed_executor_backend="ray")` just works.

### Async + Ray

The async engine (`mode="async"`) works with the Ray executor. The key difference is that `AsyncLLM` spawns the EngineCore as a subprocess via `multiprocessing`, and that subprocess creates the `NNsightRayExecutor`. For Ray to work in the subprocess, a Ray cluster must already be running — the subprocess connects to it via `ray.init(address="auto")`.

`VLLM._load()` handles this automatically: when both `mode="async"` and `distributed_executor_backend="ray"` are set, it calls `ray.init()` in the main process before creating `AsyncLLM`, ensuring the subprocess has a cluster to connect to.

```python
model = VLLM(
    "gpt2",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
    gpu_memory_utilization=0.1,
    dispatch=True,
    mode="async",  # async + Ray
)
```

### Ray Initialization Behaviors

`NNsightRayExecutor._init_executor()` supports three scenarios:

| Scenario | Condition | Behavior |
|----------|-----------|----------|
| **Local Ray running** | `ray.init(address="auto")` succeeds | Connect to it |
| **Remote cluster** | `RAY_ADDRESS` env var set (e.g. `head:6379`) | Join as driver-only node via `ray start --num-gpus=0 --num-cpus=0`, then connect |
| **Standalone** | No Ray running, no `RAY_ADDRESS` | `ray.init()` starts a fresh local cluster |

**Important**: `RAY_ADDRESS` must be a GCS address (`host:port`), **not** a Ray Client address (`ray://host:port`). vLLM v1 uses compiled DAGs which require direct GCS access — Ray Client protocol does not support the `.bind()` method needed for compiled DAGs.

When connecting to a remote cluster, the executor also sets `VLLM_HOST_IP` to the head node's IP. This prevents vLLM's IP validation from failing when the driver machine's IP differs from the cluster nodes' IPs.

### Multi-Node Support

For multi-node tensor parallelism (TP workers on different machines), the `NNsightRayExecutor` handles everything automatically — just set `RAY_ADDRESS` to point at an existing Ray cluster, and vLLM will place workers across the available nodes.

See [`examples/multi_node_with_ray/`](examples/multi_node_with_ray/) for a complete Docker-based example that simulates multi-node on a single machine, including:
- Docker Compose setup with separate head/worker containers
- NCCL configuration for cross-node communication
- Test script validating interventions across nodes

### Why the Existing Integration Is Executor-Agnostic

The NNsight hooks (`worker_cls`, `collective_rpc`, `execute_model`) work identically across Ray and multiprocessing:

- **`worker_cls`**: Both `MultiprocExecutor` and `RayDistributedExecutor` resolve the `worker_cls` string (`"nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"`) and instantiate it. The monkey-patching of `GPUModelRunner -> NNsightGPUModelRunner` happens inside each worker process/actor.
- **`execute_model`**: `RayWorkerWrapper.execute_model_ray()` calls `self.worker.model_runner.execute_model()` — the same path as multiprocessing. Since the model runner is `NNsightGPUModelRunner`, interventions execute correctly.
- **`collective_rpc("collect_nnsight")`**: `RayDistributedExecutor.collective_rpc()` calls `worker.execute_method.remote()` which delegates to `self.worker.collect_nnsight()`. Return values (pickled bytes) survive Ray serialization.
- **Mediator transport**: `SamplingParams.extra_args` with serialized mediator bytes passes through Ray's compiled DAG via pickle.

### NNsightGPUWorker `init_device()` Override

When the executor backend is passed as a class (`NNsightRayExecutor`) rather than a string (`"ray"`), vLLM's worker-side `init_device()` encounters the class object in `parallel_config.distributed_executor_backend` instead of the expected `"ray"` string. This causes `local_world_size` assertions to fail.

`NNsightGPUWorker.init_device()` normalizes this: if the backend is a class that's a subclass of `RayDistributedExecutor`, it replaces it with the string `"ray"` before calling `super().init_device()`.

### Limitations

- **Pipeline parallelism (PP > 1)** is not supported with Ray. `collect_nnsight` only collects saves from `get_pp_group().rank == 0` (first pipeline stage). With PP > 1, saves from later stages are lost.
- **Version sensitivity**: The actor crash is a grpcio/Ray compatibility issue. Tested with vLLM 0.15.1 + Ray 2.53.0 + grpcio 1.76.0. Future versions may fix the underlying crash, at which point `ray_workaround.py` can be simplified.
- **Ray Client protocol**: Not supported. Must use direct GCS connection (`host:port`) rather than `ray://host:port`.
