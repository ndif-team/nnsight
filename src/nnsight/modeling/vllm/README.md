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
12. [Ray Distributed Executor](#ray-distributed-executor)
13. [Multi-Node Support](#multi-node-support)

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
├── sampling.py                    # NNsightSamplingParams — thin SamplingParams subclass
├── batching.py                    # VLLMBatcher — tensor-parallel gather/split + flat-batch slicing
├── engines/
│   ├── __init__.py
│   └── engine.py                  # NNsightLLMEngine — collects saved results after requests finish
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

**`sampling.py`** — `NNsightSamplingParams` is a thin subclass of vLLM's `SamplingParams` used for type identification in `_prepare_input`. Mediator data is transported via the built-in `extra_args` dict field on `SamplingParams`, not on a custom field.

**`batching.py`** — `VLLMBatcher` extends NNsight's base `Batcher` to handle tensor parallelism. Registers pre/post hooks on all modules to track which module is currently executing and whether its tensors are sharded. When intervention code requests a value, the batcher transparently gathers sharded tensors; when intervention code returns a modified value, the batcher re-shards before passing back to vLLM.

**`engines/engine.py`** — `NNsightLLMEngine` extends vLLM's `LLMEngine`. After each engine step, checks for finished requests and calls `finish_nnsight()` on the model executor to collect saved intervention results.

**`workers/GPUWorker.py`** — `NNsightGPUWorker` extends vLLM's `Worker`. Its only job is to monkey-patch `GPUModelRunner` with `NNsightGPUModelRunner` before vLLM's init runs, and to expose `finish_nnsight()`.

**`model_runners/GPUModelRunner.py`** — `NNsightGPUModelRunner` is the core of the integration. It:
- Creates a second `VLLM` wrapper around the model loaded by vLLM (inside the worker process)
- Deserializes mediators from incoming requests via `extra_args`
- Manages batch group mappings (flat token-level during forward, prompt-level after)
- Enters the interleaver at three phases: forward pass, logit wrapping, and sampling
- Collects saved values when requests finish

**`executors/ray_workaround.py`** — Contains `LazyRayWorkerWrapper` and `NNsightRayExecutor` for Ray distributed executor support. See [Ray Distributed Executor](#ray-distributed-executor) for details.

**`examples/multi_node_with_ray/`** — Docker-based example for multi-node tensor parallelism with Ray. Includes a Dockerfile, docker-compose config, test script, and detailed README. See [Multi-Node Support](#multi-node-support) for details.

---

## Key Classes

### VLLM (vllm.py)

The user-facing class. Exists in two contexts:

1. **User process**: Created by the user (`model = VLLM("gpt2", dispatch=True)`). Handles tracing, input preparation, and dispatching to the vLLM engine.
2. **Worker process**: Created by `NNsightGPUModelRunner.load_model()` to wrap the model that vLLM loaded. This instance has the interleaver and batcher attached.

Key attributes:
- `vllm_entrypoint` — The actual `vllm.LLM` instance (user process only)
- `tokenizer` — vLLM's tokenizer
- `logits` — `WrapperModule` envoy for intercepting logits
- `samples` — `WrapperModule` envoy for intercepting sampled tokens
- `generator` — `WrapperModule` envoy for generation output

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
- `finish_nnsight(finished_requests)` — Collects saved values from finished mediators

### NNsightLLMEngine (engines/engine.py)

Thin extension of vLLM's engine. After each `step()`, checks for finished requests and delegates to `finish_nnsight()` on the executor to gather saved results.

### NNsightGPUWorker (workers/GPUWorker.py)

Thin extension of vLLM's worker. Monkey-patches the model runner class before init, and exposes `finish_nnsight()` which delegates to the model runner.

### NNsightRayExecutor (executors/ray_workaround.py)

Custom `RayDistributedExecutor` subclass passed as `distributed_executor_backend` when Ray is requested. Swaps in `LazyRayWorkerWrapper` before creating Ray actors, and handles connecting to existing Ray clusters (including remote ones). See [Ray Distributed Executor](#ray-distributed-executor).

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
- Each invoke's mediator is serialized to bytes via `serialize()` and stored in `param.extra_args["nnsight_mediator"]`
- Subsequent prompts in the same invoke get `param.extra_args["nnsight_batch_member"] = True`
- `vllm_entrypoint.generate(prompts, sampling_params)` is called

**3. vLLM schedules the request:**
- The engine passes the request through its scheduler
- The worker's `_update_states()` is called with the scheduler output

**4. `NNsightGPUModelRunner._update_states()`:**
- Calls `process_new_reqs()` — checks `extra_args` for `"nnsight_mediator"` bytes and deserializes them; uses `"nnsight_batch_member"` to associate subsequent prompts with the same mediator
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

**8. When all requests in an invoke group finish:**
- `NNsightLLMEngine.step()` detects finished requests
- Calls `finish_nnsight(finished_requests)` on the executor -> worker -> model runner
- Model runner enters interleaver, calls `interleaver.handle("result", outputs)` — mediators can interact with final output
- Extracts saved values from mediator frames (any variable marked with `.save()`)
- Returns saves dict (pickled to bytes so it survives msgpack transport in multiprocessing mode), which gets attached to the `RequestOutput`

**9. Back in user process:**
- `VLLM.__call__()` receives the `RequestOutput` with attached saves
- Saved values are pushed back into the user's local variables

---

## Model Loading

The `VLLM` class uses `MetaMixin` for lazy/eager loading.

### Meta Loading (`_load_meta`)

When `dispatch=False` (default), the model is loaded with meta tensors (no real weights allocated). This uses vLLM's `DummyModelLoader` with `device="meta"`. The purpose is to build the Envoy tree (module hierarchy) so users can write intervention code referencing `model.transformer.h[0].output` etc. without allocating GPU memory.

### Real Loading (`_load`)

When `dispatch=True` or when `interleave()` auto-dispatches:
- Destroys any existing distributed environment
- If `distributed_executor_backend="ray"`, replaces it with `NNsightRayExecutor` class (see [Ray Distributed Executor](#ray-distributed-executor))
- Creates a `vllm.LLM` instance with `enforce_eager=True`
- Sets the worker class to `NNsightGPUWorker` via `worker_cls` kwarg
- After creation, monkey-patches the engine class to `NNsightLLMEngine`

### Worker-Side Loading

Inside the worker process, `NNsightGPUModelRunner.load_model()`:
- Calls vLLM's normal `load_model()` (loads real weights)
- Creates a new `VLLM` wrapper around the loaded model
- Creates a `VLLMBatcher` and attaches it to the interleaver
- Calls `batcher.wrap(model)` to register tensor-parallelism hooks on all modules

This means there are **two VLLM instances**: one in the user process (for tracing/input prep) and one in the worker process (for interleaving).

---

## Mediator Transport via extra_args

The core challenge: intervention code is compiled into a `Mediator` in the user process, but must execute in the worker process.

### How It Works

1. During tracing, each invoke produces a `Mediator` containing the compiled intervention function.
2. `VLLM.__call__()` serializes the mediator to bytes via `serialize()` and stores it in `param.extra_args = {"nnsight_mediator": <bytes>}`. Subsequent prompts in the same invoke get `param.extra_args = {"nnsight_batch_member": True}`.
3. `SamplingParams.extra_args` is a built-in `dict[str, Any] | None` field that survives vLLM's internal msgpack serialization when passing to worker processes (both multiprocessing and Ray).
4. In the worker, `process_new_reqs()` checks `extra_args` for `"nnsight_mediator"` bytes and deserializes using `load()`, passing the worker-side model's persistent objects for reference resolution.
5. Batch members (prompts after the first in an invoke) are identified by the `"nnsight_batch_member"` marker and associated with the most recently deserialized mediator.

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

### Phase 4: Finish (`finish_nnsight`)

When all requests in an invoke group complete, the interleaver handles the `"result"` provider. This lets mediators interact with the final generation output (accessed via `tracer.result`). Saved values are then extracted from mediator frames.

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

**New requests**: `process_new_reqs()` checks `extra_args` for mediator bytes and deserializes them. Batch members are associated with the most recently deserialized mediator via the `"nnsight_batch_member"` marker.

**Batch group updates**: `process_batch_groups()` recomputes batch groups every step based on what the scheduler has actually scheduled. Only currently-scheduled requests are reflected in batch groups.

**Finished requests**: When all requests belonging to an invoke group are finished (per-invoke-group, not per-request), `finish_nnsight()`:
1. Enters the interleaver and handles the `"result"` provider
2. Extracts saved values from the mediator's frame locals
3. Cancels the mediator
4. Returns saved values (pickled to bytes), which get attached to `RequestOutput`

After finishing, remaining active requests have their batch groups re-computed for subsequent steps.

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
  finish_nnsight() -> collect saves, cleanup
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

When the user passes `distributed_executor_backend="ray"`, `VLLM._load()` replaces the string with the `NNsightRayExecutor` class before passing it to `vllm.LLM()`:

```python
if kwargs.get("distributed_executor_backend") == "ray":
    from .executors.ray_workaround import NNsightRayExecutor
    kwargs["distributed_executor_backend"] = NNsightRayExecutor
```

vLLM's `EngineArgs.distributed_executor_backend` accepts `str | type[Executor]`, so passing a class directly is supported. This is cleaner than external monkey-patching because:

1. **Works with multiprocessing mode**: vLLM pickles the executor class to the EngineCore subprocess. `NNsightRayExecutor._init_executor()` runs inside that subprocess, where it swaps in `LazyRayWorkerWrapper` before any Ray actors are created. No need to force `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
2. **Self-contained**: The workaround is entirely within `NNsightRayExecutor` — no global state or env var overrides.
3. **Transparent to users**: `VLLM("gpt2", distributed_executor_backend="ray")` just works.

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
- **`collective_rpc("finish_nnsight")`**: `RayDistributedExecutor.collective_rpc()` calls `worker.execute_method.remote()` which delegates to `self.worker.finish_nnsight()`. Return values (pickled bytes) survive Ray serialization.
- **Mediator transport**: `SamplingParams.extra_args` with serialized mediator bytes passes through Ray's compiled DAG via pickle.

### NNsightGPUWorker `init_device()` Override

When the executor backend is passed as a class (`NNsightRayExecutor`) rather than a string (`"ray"`), vLLM's worker-side `init_device()` encounters the class object in `parallel_config.distributed_executor_backend` instead of the expected `"ray"` string. This causes `local_world_size` assertions to fail.

`NNsightGPUWorker.init_device()` normalizes this: if the backend is a class that's a subclass of `RayDistributedExecutor`, it replaces it with the string `"ray"` before calling `super().init_device()`.

### Limitations

- **Pipeline parallelism (PP > 1)** is not supported with Ray. `finish_nnsight` only collects saves from `get_pp_group().rank == 0` (first pipeline stage). With PP > 1, saves from later stages are lost.
- **Version sensitivity**: The actor crash is a grpcio/Ray compatibility issue. Tested with vLLM 0.15.1 + Ray 2.53.0 + grpcio 1.76.0. Future versions may fix the underlying crash, at which point `ray_workaround.py` can be simplified.
- **Ray Client protocol**: Not supported. Must use direct GCS connection (`host:port`) rather than `ray://host:port`.
