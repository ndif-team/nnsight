# PP Review TODOs

Status: reviewed 2026-04-04. Covers commit `2ace37c` (PP pull transport + end-to-end interventions).

---

## Workflow

For each issue: (1) explain *why* it's a problem — the underlying dynamics that create the failure mode, (2) propose the fix, (3) explain *why* the fix resolves the problem fundamentally. Get approval before implementing.

---

## Classification Key

- **BUG** — Correctness issue. Can produce wrong results, crashes, or data corruption.
- **STYLE** — Unacceptable pattern. Hardcoded constants, abstraction violations, duplicate code, defensive getattr. Not a bug today but a maintenance trap or future bug.
- **PERF** — Hot-path performance issue. Unnecessary work on every hook/token/forward pass.
- **ROBUSTNESS** — Missing error handling, silent failures, no timeouts. Works in the happy path, breaks ungracefully otherwise.
- **CLEANUP** — Test scaffolding in production, dead code, dev artifacts. Low risk but should be removed.
- **INVESTIGATION** — Needs testing or measurement before deciding on a fix.

---

# BUG

## ~~BUG-1: `pp_hook_buffer.clear()` without holding the condition lock~~ FIXED

~~`GPUModelRunner.py:598` — `collect_nnsight()` calls `self.pp_hook_buffer.clear()` outside the condition lock.~~

**Fixed** as part of the iteration gate redesign. The deeper issue was that `collect_nnsight` could run while mediators were still mid-pull (mediator threads survive `execute_model()` return). The fix:
1. Added an **iteration gate** (`_iter_condition` + `_generation_done` on Interleaver) that blocks mediators at iteration boundaries until the next forward pass starts or generation ends.
2. `collect_nnsight` now calls `interleaver.stop_iteration()` + `thread.join()` before finalizing — all pulls complete before any cleanup.
3. Buffer clear is now under `with self.pp_buffer_condition:` as belt-and-suspenders.

Files changed: `interleaver.py`, `iterator.py`, `GPUModelRunner.py`.

---

## ~~BUG-2: `Globals.enter()` change affects all code paths, not just PP~~ FIXED

~~`globals.py:137` — Changed from `if cur == 0:` to `if cur == 0 and _saves_var.get() is None:`. The `is None` guard prevented cleanup between unrelated traces.~~

**Fixed:** Root cause was that vLLM saves were tracked in the global `Globals.saves` set, which got orphaned when `Globals.enter()` reset it across `execute_model`/`_sample` boundaries. Fix: moved save tracking to per-trace sets stored in `trace_contexts["saves"]`. Each mediator holds a reference (`mediator._trace_saves`), and `_saves_var` is pointed at the trace's set before `mediator.start()`. `collect_saves` and `cleanup_finished` now operate on per-trace sets. `Globals.enter()` reverted to unconditional `if cur == 0: reset`.

Files changed: `GPUModelRunner.py` (serve branch), `globals.py` (reverted in merge).

---

## ~~BUG-3: `_dtype_to_code` / `_code_to_dtype` silently corrupt unsupported dtypes~~ FIXED

~~`pp_listener.py:196,199` — Unknown dtypes silently fall back to `float32` (code 1). A tensor with `torch.float8_e4m3fn` is encoded as float32, the receiver decodes it as float32, and the data is silently corrupt. No error is raised.~~

**Fixed** by eliminating dtype encoding entirely. Root cause: the pull protocol treated cross-rank tensor transfer as a generic "send unknown tensor" problem, when both ranks loaded the same model and know the dtype statically. Fix:
1. Built `pp_module_meta` map (`{module_path: dtype}`) at load time from `model_config.dtype`. Shared to interleaver and PPListener.
2. Receiver resolves dtype locally from the map — no dtype on the wire.
3. Deleted `_dtype_to_code`, `_code_to_dtype`, `_DTYPE_MAP`, `_CODE_MAP`.
4. Buffer now stores narrowed (per-mediator) value instead of full batch — remote ranks pull exactly what they need.
5. Deduplicated `_pp_lazy_output`/`_pp_lazy_input` into `_pp_lazy_access(kind)` (also fixes STYLE-4).
6. Simplified `LazyRemoteTensor` constructor — removed `shape`/`device` params (not known statically).

Files changed: `pp_listener.py`, `envoy.py`, `interleaver.py`, `GPUModelRunner.py`, `lazy_remote_tensor.py`, `test_vllm_pp.py`.

**Discovered:** Latent bug — vLLM layers return `(hidden_states, residual)` tuples, but `_listen_loop` calls `.detach().contiguous().cpu()` on the buffer value, which fails on tuples. Works only because current test coverage doesn't exercise tuple-output pulls. Filed as new issue below.

---

## BUG-4: `get_owning_rank` returns `None` → `LazyRemoteTensor(source_rank=None)`

`envoy.py:352,379` — If `PPModuleMap` doesn't recognize a module path, `get_owning_rank` returns `None`. This flows into `LazyRemoteTensor(source_rank=None, ...)`. On materialization, `pull_from_remote(None, ...)` crashes inside `dist.send` with an unhelpful error.

**Fix:** Raise early in `_pp_lazy_output`/`_pp_lazy_input` when `source_rank is None` with a message naming the unresolved module path.

---

## ~~BUG-5: Missing `collect_saves` LazyRemoteTensor filter~~ FIXED

~~PP_DESIGN.md specifies "Filter out any unmaterialized LazyRemoteTensor in collect_saves as safety net." The actual `collect_saves` (GPUModelRunner.py:285-326) does no such filtering. If a LazyRemoteTensor's `id()` enters `Globals.saves` (e.g., via `nnsight.save()` bypassing the no-op override), pickling fails or produces corrupt data since `__getstate__` nulls `_pull_fn`.~~

**Fixed:** Added `isinstance(value, LazyRemoteTensor)` filter in both collection paths within `collect_saves` (per-invoke frame locals and trace-shared saves). File changed: `GPUModelRunner.py`.

---

## BUG-6: Provider string includes `.output.iN` suffix when passed to `get_owning_rank`

`envoy.py:352` — Passes `"model.layers.5.output.i0"` to `get_owning_rank`, which splits on `.` and searches for layer container names. Works by accident because `"output"` and `"i0"` don't match any name in `_LAYER_CONTAINER_NAMES`. If someone adds `"output"` to any name set, it breaks silently.

**Fix:** Pass the module path (without suffix) to `get_owning_rank`, not the full provider string. Or better: replace PPModuleMap entirely (see STYLE-2).

---

## BUG-7: Pull protocol crashes on tuple outputs (vLLM layers return tuples)

`pp_listener.py:_listen_loop` — calls `tensor.detach().contiguous().cpu()` on the buffer value, but vLLM decoder layers return `(hidden_states, residual)` tuples. The `narrow` in `handle_value_event` maps over the tuple (narrowing each tensor), so the buffer stores a tuple of narrowed tensors. `_listen_loop` then crashes trying to call `.detach()` on a tuple.

Discovered during BUG-3 analysis. Currently not hit because PP test coverage doesn't exercise cross-rank pulls of tuple-output modules.

**Fix:** Either serialize tuples element-by-element in the pull protocol, or store only the first element (hidden states) since that's what users access via `model.layers[i].output[0]`.

---

# STYLE

## STYLE-1: PP attributes bolted on dynamically with defensive `getattr`/`hasattr`

6 PP attributes on `Interleaver` and 4 on `GPUModelRunner` are set dynamically in `load_model()`. All access sites use `getattr(self._interleaver, 'pp_enabled', False)`. If someone renames `pp_enabled`, every `getattr` silently returns the default — PP stops working with no error.

**All access sites (verified complete):**

`getattr`:
- `envoy.py:327` — `getattr(self._interleaver, 'pp_enabled', False)`
- `envoy.py:335` — `getattr(self._interleaver, 'pp_module_map', None)`
- `envoy.py:337` — `getattr(self._interleaver, 'pp_local_rank', None)`
- `interleaver.py:846` — `getattr(self.interleaver, 'pp_enabled', False)`
- `GPUModelRunner.py:475` — `getattr(self, 'pp_listener', None)`
- `GPUModelRunner.py:491` — `getattr(self, 'pp_enabled', False)`
- `GPUModelRunner.py:597` — `getattr(self, 'pp_enabled', False)`

`hasattr`:
- `envoy.py:363` — `hasattr(self._interleaver, 'pp_listener')`
- `envoy.py:389` — `hasattr(self._interleaver, 'pp_listener')`

**Fix:** Define all PP attrs with defaults on `Interleaver.__init__` and `GPUModelRunner.load_model` (before conditional setup). Replace all `getattr` with direct access, all `hasattr` with `is not None` checks.

---

## STYLE-2: PPModuleMap string-pattern guessing duplicates `is_pp_missing` ground truth

`pp.py` has `is_pp_missing(module)` checking the actual runtime module, then `PPModuleMap` (35-101) that reimplements the same via hardcoded name sets (`_LAYER_CONTAINER_NAMES`, `_FIRST_RANK_MODULES`, `_LAST_RANK_MODULES`). Two overlapping systems for the same job. The string-pattern approach breaks on any model that doesn't use these exact names.

The name sets exist to handle WrapperModules (logits/samples) that are real modules on all ranks but only meaningful on the owning rank — `is_pp_missing()` doesn't catch these since they aren't `PPMissingLayer`.

**Fix:** Build a single `{path: owning_rank}` dict at load time from ground truth (walk Envoys + `is_pp_missing` + `get_pp_indices` for WrapperModules). Delete the hardcoded name sets and string-parsing logic.

---

## STYLE-3: Core intervention layer imports vLLM-specific code (abstraction leak)

`envoy.py` imports `nnsight.modeling.vllm.pp.is_pp_missing` and `nnsight.modeling.vllm.lazy_remote_tensor.LazyRemoteTensor` (deferred, but still a dependency). `interleaver.py:846-854` contains vLLM PP buffer clone logic.

The core `intervention/` layer is supposed to be backend-agnostic. PP logic should be injected from the vLLM backend (callback, strategy object on interleaver, or override in a vLLM-specific interleaver subclass).

---

## ~~STYLE-4: `_pp_lazy_output` and `_pp_lazy_input` are near-identical (28 lines duplicated)~~ FIXED

~~`envoy.py:342-393` — These two methods differ only in the suffix string (`".output"` vs `".input"`).~~

**Fixed** as part of BUG-3: merged into single `_pp_lazy_access(kind: str)` method.

---

## STYLE-5: Saves-merge pattern triplicated across collection sites

Identical `for r in results: if r is not None: all_saves.update(pickle.loads(r))` in:
- `async_backend.py:100-103`
- `engines/engine.py:27-30`
- `serve/server.py:142-145`

Any change to merge logic (LazyRemoteTensor filtering, error handling) must be replicated in three places.

**Fix:** Extract `merge_collected_saves(results) -> dict` utility.

---

## STYLE-6: PP iteration tracking bypasses `iterate_requester`

`envoy.py:347-350` manually reads/increments `mediator.iteration_tracker[module_key]` and constructs `f"{key}.i{iteration}"` inline. The existing `Interleaver.iterate_requester` does the same for the non-PP path but also handles `mediator.iteration` (bounded iter contexts).

The PP path ignoring `mediator.iteration` means bounded iteration (`tracer.iter[0:3]`) silently breaks with PP-missing modules.

**Fix:** Refactor a shared helper or call `iterate_requester` from the PP path.

---

## STYLE-7: `contextvars.copy_context()` removal — future ContextVars silently lost

`interleaver.py:720-735` — The old code used `copy_context()` which propagated ALL context vars. The new code manually propagates only `_stack_var` and `_saves_var`. Any future ContextVar (in nnsight or user code) won't be propagated to worker threads without updating this code.

Not a bug today, but a maintenance trap. Add a comment documenting this constraint, or switch back to `copy_context()` once the saves-set sharing issue is solved differently.

---

## STYLE-8: PP state copied to interleaver on every `execute_model`

`GPUModelRunner.py:468-476` — 6 attribute copies per forward pass, including `get_pp_group().rank_in_group` recomputed each time. These values are constant after `load_model()`. Should be set once at setup.

---

# PERF

## PERF-1: Unconditional clone + condition notify on every hook (hot path)

`interleaver.py:846-854` — Every `handle_value_event` for every module for every token does: mutex acquire, GPU tensor clone, `notify_all()`. With 40 local layers, that's 80+ GPU clones per token — most never pulled.

**Fix:** Lazy clone: record a reference, clone only when a pull request arrives. Or: only clone for modules that other ranks might access (check `pp_module_map`).

---

## PERF-2: `_is_pp_missing` recomputed on every `.output`/`.input` access

`envoy.py:325-340` — 3 `getattr` calls, a deferred import, `type().__name__` string comparison, string formatting, `get_owning_rank` (string split + dict iteration) — on every property access for every module. The PP-missing status never changes during a request.

**Fix:** Cache a boolean `self._pp_is_remote` on the Envoy at setup time.

---

## PERF-3: `pp_hook_buffer` unbounded growth during generation

Buffer accumulates `(local_layers * hooks_per_layer * N_tokens)` GPU tensor clones. Cleared only when the entire request finishes in `collect_nnsight`. For long generations, this is an OOM risk.

**Fix:** Evict consumed entries after the pull is served, or after the listener confirms no more pulls for a given iteration.

---

## PERF-4: Per-pull GPU→CPU→gloo→CPU→GPU with 6 round-trips

`pp_listener.py` — Each pull does: GPU→CPU on producer, 6 gloo send/recv, CPU→GPU on consumer. For 40 remote layers, that's 240 gloo operations per token.

**Fix (future):** Batch pulls — send a list of keys, get one concatenated response. Or NCCL for data channel with gloo for control.

---

# ROBUSTNESS

## ROBUST-1: `_pp_wait_for_mediator_readiness` spin-wait with no timeout (partially mitigated)

`GPUModelRunner.py:537-539` — `while mediator.alive and not mediator.event_queue.has_value: time.sleep(0.0001)` spins at 10KHz. If a mediator deadlocks (e.g., pull from a dead rank), this spins forever with no diagnostics.

**Partially mitigated** by the iteration gate: mediators can no longer run into an extra iteration that triggers impossible pulls after the last token. The spin-wait now only covers within-iteration PP-missing processing, which is bounded by the number of layers. Still worth adding a timeout for robustness.

**Remaining fix:** Add a timeout to the spin-wait loop. Log a warning if the timeout is reached.

---

## ROBUST-2: `_listen_loop` catches all exceptions and continues silently

`pp_listener.py:126-128` — Bare `except Exception: traceback.print_exc()` in an infinite loop. Protocol corruption (partial recv, malformed header) causes infinite retry with no escalation. The loop can never exit except by process death.

**Fix:** Distinguish recoverable errors (timeout waiting for buffer) from protocol-level corruption (malformed header, partial recv). Fatal errors should break the loop or set an error flag that callers can check.

---

## ROBUST-3: `local_lookup` timeout is 60s hardcoded, no configurability

`pp_listener.py:60` — If a cross-rank pull takes longer than 60s (overloaded cluster, slow rank), the PP pipeline fails. The error message doesn't say which rank is waiting for which, or which provider string timed out.

**Fix:** Make timeout configurable. Include source rank, destination rank, and provider string in the `TimeoutError` message.

---

## ROBUST-4: No listener shutdown mechanism

`pp_listener.py:82` — `_listen_loop` is `while True` with no stop event. Relies on daemon thread dying with the process. If the gloo group is torn down while the listener is blocked on `dist.recv`, behavior is undefined (hang or crash depending on gloo implementation).

**Fix:** Add a `_stop_event` checked each iteration. Provide a `stop()` method called from `collect_nnsight` cleanup.

---

# CLEANUP

## ~~CLEANUP-1: Test helpers in production GPUModelRunner~~ FIXED

~~`GPUModelRunner.py:618-660` — `test_pp_buffer_put`, `test_pp_pull`, `test_pp_buffer_clear`, `test_pp_profile_pull` are test scaffolding. Passthrough wrappers in `GPUWorker.py:36-46`.~~

**Fixed:** Removed test helpers from production code.

---

## ~~CLEANUP-2: Dead code in `LazyRemoteTensor.__getitem__`~~ FIXED

~~`lazy_remote_tensor.py:130` — `child._pull_fn = self._pull_fn` is immediately overwritten at line 137 by `child._pull_fn = _deferred_pull`. Line 130 is dead.~~

**Fixed:** Removed dead assignment.

---

## ~~CLEANUP-3: Task tracking markers in production code~~ FIXED

~~`GPUModelRunner.py:486,493` — `# --- Task 6: PP readiness check ---` / `# --- End Task 6 ---`. Dev artifacts.~~

Removed as part of the iteration gate change.

---

## ~~CLEANUP-4: Dead `world_size` variable~~ FIXED

~~`GPUModelRunner.py:414` — `world_size = dist.get_world_size()` assigned but never referenced.~~

**Fixed:** Removed dead variable.

---

## CLEANUP-5: Remove unnecessary `__getstate__` from LazyRemoteTensor

`lazy_remote_tensor.py:37-41` — Nulls `_pull_fn` for picklability. But `save()` is a no-op, so LazyRemoteTensors should never enter the saves dict. Dead code guarding against a scenario the design already prevents. (If BUG-5 filter is added, this becomes doubly unnecessary.)

---

# INVESTIGATION

## INVESTIGATE-1: Verify gloo initialization on multi-machine (simulated via Docker)

The existing multi-node TP setup (`examples/ray/`) uses Docker containers to simulate separate machines. PP should be testable the same way.

### What to verify

1. **Does `dist.new_group(ranks=..., backend="gloo")` work across Ray-spawned workers on different nodes?** vLLM's Ray executor initializes `torch.distributed` with NCCL. PyTorch's `new_group(backend="gloo")` should bootstrap from the existing store, but needs verification.

2. **Does gloo TCP transport work between containers?** Potential issues: wrong interface binding (needs `GLOO_SOCKET_IFNAME=eth0`), firewall/namespace isolation, DNS resolution.

3. **Rank addressing correctness across nodes.** `PPListener` uses `group_src`/`group_dst` derived from `pp_group.rank_in_group`. Need to confirm correct mapping when global ranks span machines.

### How to test

- **PP=2**: 2 containers, 1 GPU each.
- **PP=2+TP=2**: 4 containers, 1 GPU each.
- Dockerfile needs updating to install from `pp-design` branch.

### Possible failure modes

- `new_group(backend="gloo")` raises → fix: bootstrap from vLLM's TCPStore, or use standalone TCP sockets.
- Gloo binds to loopback → fix: set `GLOO_SOCKET_IFNAME`.
- Rank mapping mismatch → fix: use explicit rank lists from vLLM's PP group.

---

## INVESTIGATE-2: LazyRemoteTensor shape metadata via meta-device scan

Currently `LazyRemoteTensor._meta` stores `shape=()`, `dtype=float32`, `device=cpu` — all wrong before materialization. If user code calls `.shape` or `.dtype` pre-materialization, it gets garbage.

### Proposed approach

1. After `_load_meta()` (which creates the full model on meta device before `make_layers()` inserts PPMissing stubs), run a fake-tensor forward pass. Record `{module_path: (shape, dtype)}`.
2. Thread into workers via `vllm_config` or shared object.
3. Look up shape/dtype when constructing a `LazyRemoteTensor`.

### Open questions

- How to thread the shape map from driver into workers? (vllm_config, shared file, Ray object store?)
- Lazy vs eager resolution of dynamic dims (batch, seq)?
- `device` should be local consumer GPU — trivial fix independent of shape work.

---

## INVESTIGATE-3: Pull batching for cross-machine performance

If cross-machine pull latency is >10ms, individual per-layer pulls become a bottleneck. Options:
- **Batch pulls**: send list of keys, get concatenated response.
- **Prefetch/push**: producer proactively sends buffered tensors after forward pass.
- **NCCL data channel**: gloo for control, NCCL for tensor data (GPUDirect RDMA).

Depends on INVESTIGATE-1 results.

---

## Environment notes

- Machine: 8x A100 80GB PCIe
- Current PP tests (`test_vllm_pp.py`) run single-machine with `mp` executor
- Multi-node TP Docker setup exists and was validated previously
- vLLM v0.15.1, Ray 2.53.0
