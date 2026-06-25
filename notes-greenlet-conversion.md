# Greenlet Conversion — Status

Branch: `refactor/greenlet` (PR #680 → `dev`). Converts the per-invoke **OS-thread**
worker model to **greenlet** (stackful coroutines). This is Top Win #1 from
`notes.md` §2.1.

## Result

- Whole local test suite green: **288 passed, 2 skipped, 1 xpassed** (tiny, LM,
  transform, envoys, iteration, source, caching, **editing**, serialization, VLM,
  memory cleanup). Backward/gradients, multi-invoke, barriers, early-stop,
  iteration, attached-module editing all pass.
- **~1.6× faster** per trace on a single-access micro-benchmark
  (threads 389 µs → greenlet 245 µs/trace). The per-invoke thread spawn + lock
  handoff is the overhead removed.
- Net `interleaver.py` change *deletes* machinery (the lock-based `Value` queue,
  the response queue, the CUDA-stream propagation, the thread lifecycle) and adds
  only a stdlib `deque(maxlen=1)` slot + a small `resume` helper.

## What changed

`src/nnsight/intervention/interleaver.py`
- `Mediator.Value` (lock-as-Event single-slot queue) **deleted**. The worker's one
  pending event is parked in a `collections.deque(maxlen=1)` (`self.event`);
  the greenlet `switch` provides the synchronization, so no lock is needed.
- `response_queue` **deleted**. The response to an event is simply the return
  value of switching back into the worker.
- `self.worker`: `threading.Thread` → `greenlet`. Added `self.resumer`.
- `start()`: spawn greenlet + first `switch` instead of `Thread.start()` +
  `event_queue.wait()`. **CUDA-stream capture/propagation deleted** — the worker
  now shares the caller's OS thread (and therefore its stream), so the race the
  propagation existed to fix cannot occur.
- The handoff, 1:1 with the old queues:
  - worker `send`/`end`/`exception`: `self.resumer.switch((event, data))`
  - main `respond`/`start`/`cancel`: `self.resume(value)` → `worker.switch`
- `cancel()`: reordered so the drain runs while `self.worker` is still set;
  resumes a stuck worker with `Cancelation()` instead of the queue `put/get` dance.
- `import _thread` / `from threading import Thread` → `from greenlet import greenlet`.

`src/nnsight/intervention/tracing/backwards.py`
- Backward pass now runs under `torch.autograd.set_multithreading_enabled(False)`
  (see fix #1 below).

`pyproject.toml`: added `greenlet>=3.0.0`.

## Two non-obvious fixes (both genuine, not workarounds)

**1. Backward must be single-threaded.** PyTorch's autograd engine runs the CUDA
backward on a dedicated device thread, so grad hooks fire on a *different* thread
than the one that created the worker greenlet → `greenlet: cannot switch to a
different thread`. Fix: wrap the backward in
`torch.autograd.set_multithreading_enabled(False)`, which makes the engine run
grad hooks on the calling thread. Verified empirically. This aligns with nnsight's
single-threaded-by-design interleaver and removes a source of nondeterminism.

**2. Dynamic resumer, not a fixed "main".** A worker can resume a *sibling*
mediator mid-body — e.g. `model.edit()` whose body calls an attached module that
is itself hooked. The sibling's worker must return control to *the greenlet that
resumed it*, not to a fixed main greenlet. So `resumer` is refreshed on every
resume (in `resume`) rather than captured once in `start()`. The thread model got
this for free because the queue handoff is thread-agnostic; greenlet makes the
control-flow topology explicit. This is the `cannot unpack MissedProviderError`
failure that the editing tests caught.

## Known limitation / follow-up

- **greenlet cannot switch across OS threads.** Any path where the model forward
  (and thus hooks) runs on a different thread than mediator creation will hit
  "cannot switch to a different thread." Fixed for backward (above). **Sync vLLM is
  validated** — full `test_vllm.py` on gpt2 passed 28, skipped 8 (tp>1/Ray), 0
  failed, in an isolated env (vLLM 0.15.1, torch 2.9.1+cu128); the interleaver is
  created and entered within `execute_model` on the same thread, so no cross-thread
  switch occurs. **Async vLLM / serve not yet run** (`mode="async"`,
  `test_serve*.py`); it uses a single asyncio event loop so greenlet *should* be
  fine, but it must be exercised before merge. If a cross-thread case appears, the
  same pattern applies: ensure the worker greenlet is created on the thread that
  drives it (or force that section single-threaded).
- `tests/test_remote.py`, `test_serve*.py`, `test_server_errors.py` not run here
  (need NDIF/server infra). The remote serialization path serializes mediators via
  `__getstate__`/`__setstate__`, which were updated and pass under
  `test_local_simulation.py`.
