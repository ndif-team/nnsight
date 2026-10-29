---
title: vLLM Integration Internals
one_liner: How NNsight injects interventions into vLLM's worker processes via SamplingParams.extra_args, serves forward/logits/sampling values to them, and streams sync/async/served results.
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

## Two-process layout

`VLLM(Remotable)` (`vllm.py`) exists in two processes:

- **Client process.** `VLLM("gpt2")` builds a **meta** model (via vLLM's
  `DummyModelLoader` with `load_weights` patched to a no-op, `vllm.py`) plus the
  tokenizer. Its Envoy tree is read only for structure — a client never runs a
  forward. Used for writing traces and serializing them.
- **Worker process(es).** vLLM spawns these with `worker_cls =
  "nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"` (`vllm.py`). The
  worker builds a *second* `VLLM` Envoy over the actually-loaded module
  (`GPUModelRunner.load_model`), which owns the `Interleaver`, its `VLLMFragments` and the `VLLMBatcher`, and
  is where interventions run.

Model-parallel init happens in the client process before the meta build (vLLM builds
real rank tensors and calls `.tolist()` on them, which a meta tensor can't serve),
then is torn down before the real engine starts (`_init_distributed`/
`_cleanup_distributed` in `vllm.py`). The engine runs `enforce_eager=True` —
a captured CUDA graph replays kernels without calling the module controllers —
unless the client declared `taps`. Then graphs are on and the worker's interleaver is a
`VLLMInterleaver` (`interleaver.py`): at each tap it registers its own `handle`
with vLLM's breakable graph capture (`add_eager`), so the handoff is replayed
with the graph. The tap set reaches the worker in
`vllm_config.additional_config["nnsight_taps"]`; an edit at a tap is copied back
into the recording's tensor, since the callable's return is discarded.

### `mode="sync"` vs `mode="async"`

The only switch is the constructor kwarg: `VLLM(..., mode="sync")` (default) or
`mode="async"` (`vllm.py`, sets `self._async_engine`). Sync builds an `LLM` whose
engine class is rebound to `NNsightLLMEngine` (`vllm.py`); async builds an
`AsyncLLM` (`vllm.py`) with no engine subclass — collection happens in the
streaming backend instead.

## The transport: `extra_args["nnsight_mediator"]`

The intervention is compiled in the client but must run in the worker. NNsight uses
vLLM's built-in `SamplingParams.extra_args` — a dict field that survives both Ray
(pickle) and multiprocessing (msgpack) — so no `SamplingParams` subclass is needed.

`VLLM._attach_mediators(params, **kwargs)` (`vllm.py`) is the serialization
boundary; for each mediator with a non-`None` `batch_group`:

```python
param.extra_args = {"nnsight_mediator": dumps(mediator)}
```

`dumps` is the source-based pickler (see [serialization.md](./serialization.md)), so
the mediator's block recompiles against the worker's Python; its `__globals__`
(including referenced user variables) travel with it. Trace-level sampling kwargs
fill in any `SamplingParams` field still at its default.

On the worker, `Requests.add` (`GPUModelRunner.py`) deserializes each new
request's mediator with `loads(..., persistent_objects=model._remoteable_persistent_objects())`
(the tokenizer resolves by persistent ID). Requests with no nnsight payload — other
tenants sharing the engine — are skipped.

## The three interleaver entry points

`NNsightGPUModelRunner(GPUModelRunner)` (`GPUModelRunner.py`) is installed by
`NNsightGPUWorker`, which rebinds `gpu_model_runner.GPUModelRunner` to the nnsight
subclass **before** `Worker.__init__` resolves it (`GPUWorker.py`) — vLLM's own
startup is not patched. The runner enters the interleaver at three points:

1. **Forward** — `execute_model` (`GPUModelRunner.py`): runs
   `super().execute_model(...)` inside `with interleaver:`, so the module
   controllers serve parked workers. Afterward `Requests.unflatten` switches each worker's batch
   group from per-token to per-row.
2. **Logits and sampling** — `_sample`, which vLLM's `sample_tokens` calls with the
   step's logits (after a structured-output request's grammar mask): offers them via
   `type(model).logits.provide(model, logits)`, samples from what comes back, then
   offers `sampler_output.sampled_token_ids = type(model).samples.provide(model, ...)`.
   Then captures all workers' saves in one pass (`record_saves`) while still on the
   workers' own thread.

`logits`/`samples` are `eproperty` descriptors on `VLLM`, and these `.provide` calls
are their produce side — the worked example in
[extending-envoy.md](./extending-envoy.md#serving-a-value-that-isnt-a-module-output-vllm-logitssamples).

Re-entering the interleaver after the forward restarts nothing:
`Interleaver.__enter__` skips any worker that already has a greenlet, and a finished
one is parked nowhere, so only blocks still waiting are served. Errors are deferred
per request (`interleaver.defer_exceptions = True`) and surfaced to the client via
`raise_deferred`; a `tracer.stop()` is silent control flow. `_finish_erred`
retires an erred/stopped request by forcing its next token to EOS.

## Continuous batching & the `Requests` helper

vLLM concatenates all in-flight tokens into one `[total_tokens, hidden]` slab.
`Requests` (`GPUModelRunner.py`) maps each mediator onto its own token span,
recomputed every step:

- `add(new_requests, persistent_objects)` deserializes new mediators from `extra_args`.
  A request vLLM preempted and resumed comes back through the same call and is
  skipped: its workers continue, because the engine replays the tokens they already
  saw inside one recompute step, and a fresh block would be short by exactly those
  steps. `scope` notes the interleaver's occurrence counts when a worker's request
  leaves the batch (`Requests.out`) and moves the worker's `base` past the visits it
  sat out when it returns, so its next `tracer.iter` step is the recompute step.
- `scope(model)` sets each scheduled worker's `batch_group = [start,
  tokens]`, `mediator.start(interleaver)` on first schedule, and keeps finished
  workers scheduled only if they still hold caches or an exception. It orders by
  `list(self.input_batch.req_ids)` — **input-batch order, not scheduler order**,
  which `condense`/reorders can diverge from.
- `unflatten(model)` re-points each scheduled worker to `[row, 1]` for the
  per-request logits/samples tensors.
- `match(request_ids)` reconciles engine ids with worker ids — vLLM appends
  a content hash, so engine `"0"` maps to worker `"0-<hash>"`.
- `record_saves()` / `saves` / `error` capture and read back
  each worker's saved names and any error, on the workers' thread (required because
  final collection may run on a different thread under Ray).
- `finish_dangling(worker_id)` throws a still-parked worker at request end
  into a `ValueError` (barrier) or `OutOfOrderError` (the run already ran past its
  location); an over-iterated `tracer.iter` only warns.

## Tensor parallelism: `VLLMFragments`

When `tensor_parallel_size > 1`, parallel linears shard tensors across GPUs;
intervention code must see the whole tensor. `VLLMFragments(Fragments)`
(`fragments.py`) says which locations are a rank's piece and how to reassemble
them; the `Interleaver` brackets the gather (see the module docstring of
`src/nnsight/intervention/fragments.py`).

- `instrument` records `location -> (module, side)` as the Envoy tree is built,
  for the layers that really shard: `ColumnParallelLinear` output unless it
  gathers its own, `RowParallelLinear` input when `input_is_parallel`, its output
  unless `reduce_results`, and a `FusedMoE` output that defers its combine.
- `whole` all-gathers or all-reduces per layer kind; `fragment` is the inverse,
  dividing an MoE write-back by `tp_size * ep_size` so the block's own reduce sums
  it exactly once. With TP=1 nothing is recorded and `enabled` stays `False`.
- `VLLMBatcher` keeps only the row math: its `batching` property is always `True`,
  because a request's tokens sit alongside others in the slab, so even a lone
  invoke must be narrowed to its own span. `_token_dim` picks the axis to narrow
  on — dim 0 for a model vLLM has its own definition for, dim 1 for one served
  through vLLM's Transformers backend, whose decoder layers carry the wrapped
  HuggingFace module's leading singleton batch dim (`[1, total_tokens, hidden]`).
  The base's dim-0-only rule reads `shape[0] == 1 != total` on those and calls the
  activation unbatched, so a block sees every request's tokens and its writes are
  dropped. The graph-replay path is not covered: `VLLMInterleaver.replay`
  (`interleaver.py`) trims a padded tap tensor with `t[:total]`, which is still
  dim-0-only.
- `VLLMBatcher.narrow` also honours `NNSIGHT_VLLM_CLONE_READS`, which serves each
  worker a copy of its span instead of a view. Read from the environment, not
  `CONFIG`, and once per batcher: the one that narrows is built in the worker
  (`GPUModelRunner.load_model`), so a client-side `CONFIG` field would never reach
  it. Taps go through the same handoff and are covered. Nothing then aliases
  engine memory, so `_copy_into` writes back the unchanged value and only a swap
  (through `widen`) lands; the user-facing trade is in
  [models/vllm.md](../models/vllm.md#serving-copies-instead-of-views).

The gather lives on the interleaver, not the batcher, because `Batcher.narrow` runs
once per *parked worker*: a gather there would run one collective per reader, and
since which workers read a value is a property of the block rather than of the
model, the ranks would run different numbers of collectives and deadlock.
`Interleaver.handle` sees each visit exactly once, so one visit is one collective,
however many workers read it. See [batching-internals.md](./batching-internals.md).

## Sync result collection

`NNsightLLMEngine(LLMEngine)` (`engines/engine.py`) overrides only `step()`: after
`super().step()`, for **every** finished request (not only traced ones) it calls
`collective_rpc("collect_nnsight", args=(finished, finished, by_id))` — the third
argument is the finished `RequestOutput`s, which the worker cannot build and needs
in order to serve `tracer.result` — merges the ranks with `merge_collected`, and
`attach`es the result to each output. `VLLM._collect` then `mark`s each saved value
and writes it into the mediator's locals, and `raise_deferred`s any error.

Ranks are merged, **not** first-non-empty: a trace's values come from the reporting
rank while a registered block's come from whichever rank ran the layers it read, and
taking one payload would silently drop the other. Where two ranks report the same
registered name — tensor parallelism, where each gathers the same whole value — the
earliest rank's is kept, so the value lands on the device a traced one would.

`attach` sets `output.saves` (trace and registered values together) and
`output.nnsight_saves` (the trace's own), plus `nnsight_error`. The separation is
load-bearing: `_collect` feeds only `nnsight_saves` to `merge_shared_saves`, which
reads a name saved across several requests as one shared container and would
otherwise fold a sweep's per-request values into one.

`collect_nnsight(request_ids, finished_request_ids, outputs=None)` runs on the
worker and returns
`pickle.dumps({engine_id: {"error", "sequences": {index: {"saves", "registered"}}}})`. In one
loop over one `Request` record per in-flight request (its traced worker, its
registered copies, what they left behind, any deserialization error) it harvests
what finished, takes the registered values, serves `tracer.result`, throws
`finish_dangling` into whatever is still parked, collects the saves, and pops
finished requests.

`sequences` is keyed by sampled-sequence index: `n > 1` fans a request into a child
per sequence (`"{index}_{parent}"`, vLLM's `ParentRequest`), each running its own
copy of the block. `Request.key` resolves a vLLM id to `(engine_id, index)` once, on
arrival, taking the child reading only when the parent it names is one the engine
asked about — so an id that merely starts with digits and an underscore is not
mistaken for a second sequence. `merge_collected` adds flat `saves`/`registered`,
the primary sequence's, once the ranks' payloads are combined; `attach` puts each sequence's on `output.outputs[i].saves` and the trace's own on
`output.nnsight_sequences`, which `VLLM._collect` pushes back as a list.

**Every rank winds up its own workers; only one reports.** Each rank ran the block,
so each holds a worker, a greenlet, and whatever that greenlet captured — all of it
has to be released. Only rank 0's values are reported, since the reads are gathered
and every rank holds the same ones. Only the *reporting* is gated on rank; an early
return on the other ranks would leave their workers in place for the life of the
engine. `nnsight_request_count()` is the leak gauge — it returns to 0 **on every
rank**, which is what `tests/vllm/test_tensor_parallel.py::TestEveryRankWindsUp` pins.

`VLLM.generate` used without a `with` block (detected by `WithBlockNotFoundError`
from `capture()`, as `traceable` does) returns vLLM's `RequestOutput`s — an awaitable
on `mode="async"` — and `_generate_async` does its own collect, because the
streaming backend only runs for traces nnsight submitted.

## Async streaming

`VLLMTracer(InterleavingTracer)` (`tracer.py`) splits the base tracer's
`execute` in two: `prepare(code)` builds the invoke workers and assembles the
call input **without running the model**, returning `(mediators, args, kwargs)`.

`AsyncVLLMBackend(Backend)` (`async_backend.py`), injected by `VLLM.trace` when
`mode="async"`:

- `__call__(tracer)` runs on `__exit__` while the frame is live: dispatches,
  `tracer.prepare(...)`, `_attach_mediators`, then starts
  `model.vllm_entrypoint.generate(prompt, param, request_id)` and stores the async
  generator. One prompt only (`NotImplementedError` otherwise).
- `__aiter__` — `async for output in tracer.backend:` yields each step's
  `RequestOutput`; on `output.finished` it collects saves via `collective_rpc(
  "collect_nnsight", ...)`, attaches `output.saves`, and `raise_deferred`s. If the
  consumer stops early, `_free_worker` frees the aborted request's worker.
- `__await__` — `await tracer.backend` drains the stream and returns the last
  output.

Sync collects inside `NNsightLLMEngine.step()`; async has no `step()` and collects
per request in the stream.

## Registered blocks — the `registration.py` module

`model.edit()` is the persistent counterpart of the per-request transport
above: instead of a mediator per request in `extra_args`, one block is sent to
every rank via `collective_rpc("nnsight_register", ...)` and kept there.

- `RegisteringTracer` (`registration.py`) captures the block like `EditingTracer`
  but `execute` ships it rather than storing it on the envoy — an edit is
  replayed by the envoy that holds it, which on vLLM leaves it in the client
  where there are no weights.
- `Requests.register` deserializes **once** and keeps `(code, glbls, lcls,
  presaved)`; `Requests.add` builds a fresh `Mediator` per arriving request from
  those pieces, so the source is compiled once rather than per request — the
  performance argument for registering.
- Registered copies are scoped, started, unflattened and `record_saves`-ed
  alongside traced ones (`Requests.scope` runs them *first*, so a trace on the
  same request sees what they left).
- `Requests.harvest` moves a finished request's saves into `harvested`, driven by
  `scheduler_output.finished_req_ids` in `_update_states` — **not** by
  `collect_nnsight`, because a request nobody traced never triggers a collect.
- Values are taken at collect, not held: `collect_nnsight` pops them from
  `harvested` into the entry that rides home on the output. It also harvests on
  demand, because the scheduler's own pass happens at the top of the *next* step,
  which on the async path may come later or (for the last request in flight)
  never.
- The registered portion is answered from **every** rank, the traced portion only
  from rank 0 — a registered block runs wherever its layers live, so under PP its
  values are not on rank 0.
- Installing is synchronous on `LLM` and a coroutine on `AsyncLLM`; the latter can
  only be awaited from inside the engine's own loop (a foreign loop on another
  thread times out, and none is exposed), hence `__aenter__`/`__aexit__` and
  `aclear`. `__aenter__` spells out `__enter__`'s body rather than calling it,
  because `capture` reads the caller's frame at a fixed depth.
- `Registration.install`/`uninstall` (and their awaited forms) are the transport
  seam. `ServeRegistration` overrides those four to POST instead, for
  `model.edit(serve=url)` from a client with no engine to `collective_rpc` into;
  the idempotence guard and the `_installed_edits` bookkeeping stay on the base's
  `clear`/`aclear`, so a third transport cannot forget them.
- `VLLM._installed_edits` holds the live handles so `clear_edits()` can reach
  them; `Registration._forget` drops one as it clears.

Worker RPCs are exposed on `NNsightGPUWorker`, not the runner —
`collective_rpc` resolves method names on the worker.

**Which runner.** `NNsightGPUModelRunner` subclasses vLLM's V1 GPU model runner
(`vllm/v1/worker/gpu_model_runner.py`). `VLLM._require_v1_model_runner` forces
`VLLM_USE_V2_MODEL_RUNNER=0` before the engine is built (workers inherit it) and
refuses an explicit `1`, which would give an engine with no instrumentation.

**Prefix caching.** A cached token runs no forward, so no controller sees it. A
trace sets `skip_reading_prefix_cache` on its own request (`_attach_mediators`).
A registration rides requests it did not create, which it cannot flag, and only
vLLM's scheduler (in the engine-core process) decides per request, so `_load`
defaults `enable_prefix_caching=False`; `_warn_if_prefix_caching` warns when an
edit is installed on an engine built with it on.

**Logits and saves.** The sampler scales its logits in place, so when a block is
in the step `_sample` hands it a copy, and a saved `model.logits` stays
what the block read. Pickling a tensor writes its whole storage, and a block's
reads are rows narrowed out of the step's buffers, so `collect_nnsight` ships
each saved view as a copy of itself (`collect.dumps_compact`, a pickler whose
`reducer_override` clones a tensor that views a larger storage).

## Serving over HTTP — the `serve/` package

`serve/` exposes `model.trace(..., serve=url)`: a GPU-less client holds only
the meta model, and a server holds one dispatched async `VLLM`.

- `serve/cli.py` — `nnsight-serve <model> [--host] [--port] [--api-key] [vLLM args]`
  (registered in `pyproject.toml` as `nnsight-serve`). Builds `VLLM(model,
  mode="async", dispatch=True, ...)`, optionally guards requests with an
  `ndif-api-key` header, and `uvicorn.run`s the app. Default host is loopback; it
  warns on `0.0.0.0` since it runs client-sent code.
- `serve/server.py` — a FastAPI app. `POST /v1/nnsight/generate` takes a
  `RequestModel.serialize(tracer)` blob, `_build_tracer` deserializes it
  (`RequestModel.deserialize`, plus restoring `tracer.node`), calls `prepare` +
  `_attach_mediators`, runs each invoke through the engine, collects saves via
  `collect_nnsight`, and `torch.save`s `{"saves", "error"}` back (saved values only;
  a build or runtime error rides `error` with its real type + traceback). Worker
  building is synchronous (all before the first `await`), so concurrent requests
  never interleave on the shared interleaver. `POST /v1/nnsight/register/{id}` and
  `.../clear` install and drop a block on the server's engine — the HTTP form of the
  `collective_rpc` an in-process `model.edit()` makes. `GET /health` reports
  readiness, and `_ready()` is what every endpoint calls first.
- `serve/http.py` — the headers, timeouts, and non-200 handling both client halves
  share: `LocalServeBackend` sending a trace and `ServeRegistration` sending a block.
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
4. `execute_model` runs the forward under the interleaver (the controllers serve
   parked workers); `unflatten` switches to per-row spans.
5. `_sample` serves `logits`, then `samples`; `record_saves` captures saves.
6. `NNsightLLMEngine.step` detects finished requests, `collect_nnsight` pulls saves,
   attaches `RequestOutput.saves`.
7. `VLLM._collect` marks saves into the user's variables and re-raises deferred errors.

Async replaces steps 2 and 6-7 with `AsyncVLLMBackend` streaming.

## Developer gotchas & invariants

Preserve these when changing the integration; each has a test guarding it:

- **A saved value returns by name; bind it.** `record_saves` maps a `.save()`d value
  back to a variable by scanning the block's locals, so a bare `model.logits.save()`
  with no assignment produces no named save and `output.saves["logits"]` is empty.
  Write `logits = model.logits.save()`.
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
- **`tracer.result` is served from `collect_nnsight`, not `interleave`.** The block
  runs in the worker and the `RequestOutput` is assembled by the engine, so the
  collect is the first moment both exist — the engine passes `{request_id: output}`
  alongside the ids and `Requests.serve_result` hands it to a worker parked on
  `"result"`, before `finish_dangling` would throw into it. The worker binds its
  name *after* the run's last `record_saves`, so `serve_result` re-takes that
  snapshot (`Requests.record`) or the value comes home unbound.

## Testing

Single-GPU tests always run; TP/Ray/async-Ray tests skip below 2 GPUs but should be
verified on a multi-GPU box before shipping runner/batcher changes. Run commands and
the `VLLM_ALLOW_INSECURE_SERIALIZATION` note are in [testing.md](./testing.md).

## Related

- [models/vllm.md](../models/vllm.md) — the user-facing behavior.
- `tests/vllm/` — the reference test suite (needs GPU + `vllm`).
