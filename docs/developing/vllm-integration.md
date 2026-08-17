---
title: vLLM Integration Internals
one_liner: How NNsight injects interventions into vLLM's worker processes via SamplingParams.extra_args, hooks forward/logits/sampling, and streams sync/async/served results.
tags: [internals, dev]
related: [docs/developing/serialization.md, docs/developing/batching-internals.md, docs/developing/adding-a-new-runtime.md]
sources: [src/nnsight/modeling/vllm/vllm.py, src/nnsight/modeling/vllm/batching.py, src/nnsight/modeling/vllm/async_backend.py, src/nnsight/modeling/vllm/tracer.py, src/nnsight/modeling/vllm/engines/engine.py, src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py, src/nnsight/modeling/vllm/workers/GPUWorker.py, src/nnsight/modeling/vllm/serve/]
---

# vLLM Integration Internals

## What this covers

vLLM is the most invasive runtime NNsight wraps. vLLM owns the model, runs
forward/logits/sampling in worker subprocesses, uses a flat `[total_tokens, hidden]`
activation layout, and continuously batches requests across steps. The core idea:
**the intervention travels to the model.** Each invoke's intervention greenlet is
serialized into its request's `SamplingParams.extra_args` and rides vLLM's own
request pipeline into the worker, where the nnsight model runner deserializes it,
runs it against the real module, and ships saved values back.

> **Deltas from the old integration.** There is no `sampling.py` /
> `NNsightSamplingParams` — interventions ride the **stock** `vllm.SamplingParams`
> `extra_args` dict under the key `"nnsight_mediator"`. There is no
> `executors/ray_workaround.py` (Ray works through vLLM's stock
> `distributed_executor_backend="ray"`), no local `DummyModelLoader` (nnsight
> monkeypatches vLLM's own), and the `README/DISCUSSION/IDEAS` files are gone.
> `async_tracer.py` is now `tracer.py` (`VLLMTracer`). `logits`/`samples` are
> `eproperty` descriptors served with `.provide` (the old separate
> `NNsightSamplingParams.provide` hook is gone). A new `serve/` package adds an HTTP
> `nnsight-serve` path.

## Two-process layout

`VLLM(Remotable)` (`vllm.py:36`) exists in two processes:

- **Client process.** `VLLM("gpt2")` builds a **meta** model (via vLLM's
  `DummyModelLoader` with `load_weights` patched to a no-op, `vllm.py:155`) plus the
  tokenizer. Its Envoy tree is read only for structure — a client never runs a
  forward. Used for writing traces and serializing them.
- **Worker process(es).** vLLM spawns these with `worker_cls =
  "nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"` (`vllm.py:189`). The
  worker builds a *second* `VLLM` Envoy over the actually-loaded module
  (`GPUModelRunner.load_model`), which owns the `Interleaver`, its `VLLMFragments` and the `VLLMBatcher`, and
  is where interventions run.

Model-parallel init happens in the client process before the meta build (vLLM builds
real rank tensors and calls `.tolist()` on them, which a meta tensor can't serve),
then is torn down before the real engine starts (`_init_distributed`/
`_cleanup_distributed`, `vllm.py:74`/`:96`). The engine runs `enforce_eager=True` —
hooks can't fire inside a captured CUDA graph.

### `mode="sync"` vs `mode="async"`

The only switch is the constructor kwarg: `VLLM(..., mode="sync")` (default) or
`mode="async"` (`vllm.py:57`, sets `self._async_engine`). Sync builds an `LLM` whose
engine class is rebound to `NNsightLLMEngine` (`vllm.py:205`); async builds an
`AsyncLLM` (`vllm.py:222`) with no engine subclass — collection happens in the
streaming backend instead.

## The transport: `extra_args["nnsight_mediator"]`

The intervention is compiled in the client but must run in the worker. NNsight uses
vLLM's built-in `SamplingParams.extra_args` — a dict field that survives both Ray
(pickle) and multiprocessing (msgpack) — so no `SamplingParams` subclass is needed.

`VLLM._attach_mediators(params, **kwargs)` (`vllm.py:403`) is the serialization
boundary. For each mediator with a non-`None` `batch_group`:

```python
param.extra_args = {"nnsight_mediator": dumps(mediator)}
```

`dumps` is the source-based pickler (see [serialization.md](./serialization.md)), so
the mediator's block recompiles against the worker's Python; its `__globals__`
(including referenced user variables) travel with it. Trace-level sampling kwargs
fill in any `SamplingParams` field still at its default.

On the worker, `Requests.add` (`GPUModelRunner.py:115`) deserializes each new
request's mediator with `loads(..., persistent_objects=model._remoteable_persistent_objects())`
(the tokenizer resolves by persistent ID). Requests with no nnsight payload — other
tenants sharing the engine — are skipped.

## The three interleaver entry points

`NNsightGPUModelRunner(GPUModelRunner)` (`GPUModelRunner.py:324`) is installed by
`NNsightGPUWorker`, which rebinds `gpu_model_runner.GPUModelRunner` to the nnsight
subclass **before** `Worker.__init__` resolves it (`GPUWorker.py:24`) — vLLM's own
startup is not patched. The runner enters the interleaver at three points:

1. **Forward** — `execute_model` (`GPUModelRunner.py:378`): runs
   `super().execute_model(...)` inside `with interleaver:`, so the module's hooks
   serve parked workers. Afterward `Requests.unflatten` switches each worker's batch
   group from per-token to per-row.
2. **Logits** — `sample_tokens` (`:437`): offers the logits via
   `type(model).logits.provide(model, original)`; if edited, rebuilds the state with
   the new tensor. Then captures all workers' saves in one pass (`record_saves`)
   while still on the workers' own thread.
3. **Sampling** — `_sample` (`:466`): after `super()._sample(...)`, offers
   `sampler_output.sampled_token_ids = type(model).samples.provide(model, ...)`.

`logits`/`samples` are surfaced with the `eproperty` extension pattern — `VLLM.logits`
(`vllm.py:144`) is an `@eproperty(description=...)` whose stub is an identity
preprocess of the served value; the client reads it (parking on
`Mediator.value("model.logits")`) and the runner's
`type(model).logits.provide(model, original)` is the produce side, forwarding to
`model.interleaver.handle("model.logits", ...)` at the eproperty's own location. The
`description` is what makes `.logits`/`.samples` show up in the model's repr (see
[extending-envoy.md](./extending-envoy.md)).

`_still_running()` (`:458`) filters to `mediator.alive` workers so re-entering the
interleaver after the forward doesn't restart completed blocks. Errors are deferred
per request (`interleaver.defer_exceptions = True`) and surfaced to the client via
`raise_deferred`; a `tracer.stop()` is silent control flow. `_finish_erred` (`:430`)
retires an erred/stopped request by forcing its next token to EOS.

## Continuous batching & the `Requests` helper

vLLM concatenates all in-flight tokens into one `[total_tokens, hidden]` slab.
`Requests` (`GPUModelRunner.py:97`) maps each mediator onto its own token span,
recomputed every step:

- `add(new_reqs, model)` (`:115`) deserializes new mediators from `extra_args`.
- `scope(model)` (`:132`) sets each scheduled worker's `batch_group = [start,
  tokens]`, `mediator.start(interleaver)` on first schedule, and keeps finished
  workers scheduled only if they still hold caches or an exception. It orders by
  `list(self.input_batch.req_ids)` — **input-batch order, not scheduler order**,
  which `condense`/reorders can diverge from.
- `unflatten(model)` (`:188`) re-points each scheduled worker to `[row, 1]` for the
  per-request logits/samples tensors.
- `match(request_ids)` (`:209`) reconciles engine ids with worker ids — vLLM appends
  a content hash, so engine `"0"` maps to worker `"0-<hash>"`.
- `record_saves()` (`:229`) / `saves`/`error` (`:308`/`:316`) capture and read back
  each worker's saved names and any error, on the workers' thread (required because
  final collection may run on a different thread under Ray).
- `finish_dangling(worker_id)` (`:258`) throws a still-parked worker at request end
  into a `ValueError` (barrier) or `OutOfOrderError` (the run already ran past its
  location); an over-iterated `tracer.iter` only warns.

## Tensor parallelism: `VLLMFragments`

When `tensor_parallel_size > 1`, parallel linears shard tensors across GPUs;
intervention code must see the whole tensor. `VLLMFragments(Fragments)`
(`fragments.py`) says which locations are a rank's piece and how to reassemble
them; the `Interleaver` brackets the gather (see
[`nnsight.intervention.fragments`][nnsight.intervention.fragments]).

- `instrument` records `location -> (module, side)` as the Envoy tree is built,
  for the layers that really shard: `ColumnParallelLinear` output unless it
  gathers its own, `RowParallelLinear` input when `input_is_parallel`, its output
  unless `reduce_results`, and a `FusedMoE` output that defers its combine.
- `whole` all-gathers or all-reduces per layer kind; `fragment` is the inverse,
  dividing an MoE write-back by `tp_size * ep_size` so the block's own reduce sums
  it exactly once. With TP=1 nothing is recorded and `enabled` stays `False`.
- `VLLMBatcher` keeps only the row math: its `batching` property is always `True`,
  because a request's tokens sit alongside others in the slab, so even a lone
  invoke must be narrowed to its own span.

This used to live in `VLLMBatcher` with two extra pairs of forward hooks per
parallel layer, installed either side of building the tree so they bracketed the
interleaver's. It needed them because `Batcher.narrow` runs once per *parked
worker*, so the gather had to be memoized and explicitly released — several
workers reading one value would otherwise have run several collectives and
deadlocked the ranks. On the interleaver the bracket is already once-per-visit,
so none of that is needed. See [batching-internals.md](./batching-internals.md).

## Sync result collection

`NNsightLLMEngine(LLMEngine)` (`engine.py:17`) overrides only `step()`: after
`super().step()`, for any finished request it calls
`engine_core.collective_rpc("collect_nnsight", args=(finished, finished))`, picks the
first non-`None` payload (only rank 0 holds sampled output), `pickle.loads` it, and
attaches `output.saves` / `output.nnsight_error` to each `RequestOutput`.
`VLLM._collect` (`vllm.py:382`) then `mark`s each saved value and writes it into the
mediator's locals, and `raise_deferred`s any error.

`collect_nnsight(request_ids, finished_request_ids)` (`GPUModelRunner.py:471`) runs
on the worker: it matches ids, drains `finish_dangling` for finished ones, builds
`{engine_id: {"saves", "error"}}`, pops finished mediators, `torch.cuda.synchronize`s,
and returns `pickle.dumps(collected)`. `nnsight_request_count()` (`:526`) is a leak
gauge — it should return to 0.

## Async streaming

`VLLMTracer(InterleavingTracer)` (`tracer.py:24`) splits the base tracer's
`execute` in two: `prepare(code)` (`:27`) builds the invoke workers and assembles the
call input **without running the model**, returning `(mediators, args, kwargs)`.

`AsyncVLLMBackend(Backend)` (`async_backend.py:40`), injected by `VLLM.trace` when
`mode="async"`:

- `__call__(tracer)` (`:48`) runs on `__exit__` while the frame is live: dispatches,
  `tracer.prepare(...)`, `_attach_mediators`, then starts
  `model.vllm_entrypoint.generate(prompt, param, request_id)` and stores the async
  generator. One prompt only (`NotImplementedError` otherwise).
- `__aiter__` (`:82`) — `async for output in tracer.backend:` yields each step's
  `RequestOutput`; on `output.finished` it collects saves via `collective_rpc(
  "collect_nnsight", ...)`, attaches `output.saves`, and `raise_deferred`s. If the
  consumer stops early, `_free_worker` (`:101`) frees the aborted request's worker.
- `__await__` (`:109`) — `await tracer.backend` drains the stream and returns the last
  output.

This mirrors `AsyncRemoteBackend`'s await/async-iterate shape. Sync differs by
collecting inside `NNsightLLMEngine.step()`; async has no `step()` and collects
per-request in the stream.

## Registered blocks — the `registration.py` module

`model.register()` is the persistent counterpart of the per-request transport
above: instead of a mediator per request in `extra_args`, one block is sent to
every rank via `collective_rpc("nnsight_register", ...)` and kept there.

- `RegisteringTracer` (`registration.py`) captures the block like `EditingTracer`
  but `execute` ships it rather than storing it on the envoy — an edit is
  replayed by the envoy that holds it, which on vLLM leaves it in the client
  where there are no weights.
- `Requests.register` deserializes **once** and keeps `(code, glbls, lcls,
  presaved)`; `Requests.add` builds a fresh `Mediator` per arriving request from
  those pieces, so the source is compiled once rather than per request. That is
  the whole performance argument for registering.
- Registered copies are scoped, started, unflattened and `record_saves`-ed
  alongside traced ones (`Requests.scope` runs them *first*, so a trace on the
  same request sees what they left).
- `Requests.harvest` moves a finished request's saves into `harvested`, driven by
  `scheduler_output.finished_req_ids` in `_update_states` — **not** by
  `collect_nnsight`, because a request nobody traced never triggers a collect.
- `Registration.collect` merges across ranks (a registered block runs wherever
  its layers live, so under PP the values are split across stages).

Worker RPCs are exposed on `NNsightGPUWorker`, not the runner —
`collective_rpc` resolves method names on the worker.

**Prefix caching.** A cached token runs no forward, so no hook fires for it. A
trace sets `skip_reading_prefix_cache` on its own request (`_attach_mediators`);
a registration rides requests it did not create, so it cannot, and
`_warn_if_prefix_caching` says so at register time.

## Serving over HTTP — the `serve/` package

`serve/` (new) exposes `model.trace(..., serve=url)`: a GPU-less client holds only
the meta model, and a server holds one dispatched async `VLLM`.

- `serve/cli.py` — `nnsight-serve <model> [--host] [--port] [--api-key] [vLLM args]`
  (registered in `pyproject.toml` as `nnsight-serve`). Builds `VLLM(model,
  mode="async", dispatch=True, ...)`, optionally guards requests with an
  `ndif-api-key` header, and `uvicorn.run`s the app. Default host is loopback; it
  warns on `0.0.0.0` since it runs client-sent code.
- `serve/server.py` — a FastAPI app. `POST /v1/nnsight/generate` takes a
  `RequestModel.serialize(tracer)` blob, `_build_tracer` deserializes it (like
  `RequestModel.deserialize`, plus restoring `tracer.node`), calls `prepare` +
  `_attach_mediators`, runs each invoke through the engine, collects saves via
  `collect_nnsight`, and `torch.save`s `{"saves", "error"}` back (saved values only;
  a build or runtime error rides `error` with its real type + traceback). Worker
  building is synchronous (all before the first `await`), so concurrent requests
  never interleave on the shared interleaver. `GET /health` reports readiness.
- `serve/backend.py` — `LocalServeBackend(Backend)`, the client side of `serve=url`:
  serializes the trace, POSTs it, `torch.load`s the result, `raise_deferred`s any
  error, `mark`s the saves, and `push`es them into the caller's frame — so reading a
  `.save()`d variable after the block works exactly as locally.

## Lifecycle (sync, end-to-end)

1. `with model.trace("Hello", max_tokens=3): logits = model.logits.save()`.
2. On `__exit__`, `VLLM._call` → `_attach_mediators` writes each mediator into its
   `SamplingParams.extra_args["nnsight_mediator"]`, then
   `vllm_entrypoint.generate(prompts, sampling_params=params, ...)`.
3. The scheduler forwards to the worker; `_update_states` → `Requests.add`
   (deserialize mediators) + `scope` (token spans).
4. `execute_model` runs the forward under the interleaver (hooks serve parked
   greenlets); `unflatten` switches to per-row spans.
5. `sample_tokens` serves `logits`; `_sample` serves `samples`; `record_saves`
   captures saves.
6. `NNsightLLMEngine.step` detects finished requests, `collect_nnsight` pulls saves,
   attaches `RequestOutput.saves`.
7. `VLLM._collect` marks saves into the user's variables and re-raises deferred errors.

Async replaces steps 2 and 6-7 with `AsyncVLLMBackend` streaming.

## Developer gotchas & invariants

Preserve these when changing the integration — each has bitten someone (and has a
test guarding it):

- **A saved value returns by name; bind it.** `record_saves` maps a `.save()`d value
  back to a variable by scanning the block's locals, so a bare `model.logits.save()`
  with no assignment produces no named save and `output.saves["logits"]` is empty.
  Write `logits = model.logits.save()`. (This is the single most common way a "saves
  don't come back" bug is actually a mis-written trace, not an engine bug.)
- **One prompt per invoke.** Each `tracer.invoke(...)` becomes exactly one vLLM
  request — batch several prompts with several invokes, never a list in one invoke
  (that raises). An empty `tracer.invoke()` has no request to ride: a do-nothing body
  is a dropped no-op, but a body with interventions raises (they'd vanish silently).
- **`trace`/`invoke` keyword arguments are sampling params, not data.** They flow into
  `SamplingParams`; an unknown one (a typo) raises rather than being silently ignored.
- **The save-scope must open on the forward thread.** `execute_model` calls `inc()` so
  a block's `.save()` passes its depth guard; this must be the thread the greenlet runs
  on, which is *not* `load_model`'s under Ray. Don't move it back to `load_model`.
- **Cross-invoke shared mutable state does not survive.** Each invoke's mediator is
  serialized separately, so a list declared outside the invokes and appended inside
  each does not merge back (unlike the in-process local path). Each invoke saves its
  own values.
- **`defer_exceptions` must drop a worker on all TP ranks or none**, and the lazy
  gather in `VLLMFragments` assumes every rank runs identical mediators in lockstep — a
  rank-divergent error or a rank-local read hangs the collective.
- **`tracer.result` is not served on vLLM.** Read generated tokens via `model.logits`/
  `model.samples` (or the streamed `RequestOutput` in async), never `tracer.result`
  (a worker would park on it forever).

## Testing

Single-GPU tests always run; TP/Ray/async-Ray tests skip below 2 GPUs but should be
verified on a multi-GPU box before shipping runner/batcher changes. The suite sets
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` (needed for the `collective_rpc` request-state
tests, and for the mediator payload to ride Ray's transport). See
[testing.md](./testing.md) for the run commands.

## Key files

- `src/nnsight/modeling/vllm/vllm.py` — `VLLM` (`:36`), `_attach_mediators` (`:403`),
  `_collect` (`:382`), `trace` override (`:330`), `logits`/`samples` eproperties (`:144`)
- `src/nnsight/modeling/vllm/fragments.py` — `VLLMFragments`
- `src/nnsight/modeling/vllm/batching.py` — `VLLMBatcher` (row scoping only)
- `src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py` — runner (`:324`),
  `Requests` (`:97`), `collect_nnsight` (`:471`)
- `src/nnsight/modeling/vllm/engines/engine.py:17` — `NNsightLLMEngine` (sync)
- `src/nnsight/modeling/vllm/async_backend.py:40` — `AsyncVLLMBackend`
- `src/nnsight/modeling/vllm/tracer.py:24` — `VLLMTracer`
- `src/nnsight/modeling/vllm/workers/GPUWorker.py:21` — `NNsightGPUWorker`
- `src/nnsight/modeling/vllm/serve/` — `cli.py`, `server.py`, `backend.py`

## Related

- [serialization.md](./serialization.md) — how mediators survive process boundaries
- [batching-internals.md](./batching-internals.md) — the `Batcher`/`VLLMBatcher` contract
- [fragments-proposal.md](./fragments-proposal.md) — the seam both distributed runtimes share
- [extending-envoy.md](./extending-envoy.md) — how `logits`/`samples` are exposed
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — vLLM as the reference runtime
- `tests/vllm/` — the reference test suite (needs GPU + `vllm`)
