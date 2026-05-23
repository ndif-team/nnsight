# Pipeline Parallelism Design

## Goal

User writes single-GPU-style intervention code. The system handles PP transparently:

```python
model = VLLM("meta-llama/Llama-3.1-405B", tensor_parallel_size=8, pipeline_parallel_size=2)

with model.trace("Hello"):
    hidden_5 = model.layers[5].output[0]           # stage 0
    model.layers[50].output[0][:] = hidden_5 * 2   # stage 1, cross-stage dependency
    logits = model.logits.output.save()

with model.trace("Hello", max_tokens=5) as tracer:
    for step in tracer.iter[:]:
        h = model.layers[5].output[0]
        model.layers[50].output[0][:] = h * 2
        logits = model.logits.output.save()
```

No `if pp_rank == X` guards. No user awareness of stage boundaries. Single-token and multi-token generation both work.

---

## How vLLM PP Works

### Model Structure

With PP=2 on an 80-layer model, vLLM's `make_layers()` creates ALL 80 layers on every rank, but only `start_layer:end_layer` are real. The rest are `PPMissingLayer` (subclass of `nn.Identity`):

```
Stage 0 (rank 0)                         Stage 1 (rank 1)
model.layers = [                         model.layers = [
  [0]:  RealLayer                          [0]:  PPMissingLayer
  ...                                      ...
  [39]: RealLayer                          [39]: PPMissingLayer
  [40]: PPMissingLayer                     [40]: RealLayer
  ...                                      ...
  [79]: PPMissingLayer                     [79]: RealLayer
]                                        ]
start_layer=0, end_layer=40              start_layer=40, end_layer=80
embed_tokens=Real, norm=PPMissing        embed_tokens=PPMissing, norm=Real
```

The model's `forward()` only iterates `islice(self.layers, start_layer, end_layer)`. PPMissingLayer modules are never called during the forward pass.

### Execution Model

Both ranks enter `execute_model()` simultaneously via `collective_rpc`. Each rank runs its forward pass sequentially (rank 0 first, rank 1 after receiving IntermediateTensors):

```
Rank 0                                    Rank 1
──────                                    ──────
execute_model() {                         execute_model() {
  model_runner.execute_model()              (blocked on IT recv)
    interleaver opens
    forward pass: layers 0-39
    interleaver closes
  send IT ─────────────────────────────▶  recv IT
                                            model_runner.execute_model()
                                              interleaver opens
                                              forward pass: layers 40-79
                                              interleaver closes
}                                         }
collect_nnsight() via collective_rpc
```

vLLM calls `execute_model()` **once per decode token**. Multi-token generation = multiple `execute_model()` calls.

### Interleaver and Mediator Lifecycle

- The **interleaver** opens/closes once per `execute_model()` call. It gates hook dispatch (`_interleaving` flag).
- The **mediator** persists across `execute_model()` calls. Its worker thread survives (blocked but alive), frame locals intact, iteration tracker preserved.
- The **iteration tracker** lives on the mediator, not the interleaver. It counts how many times each provider has been seen, persists across interleaver open/close cycles.

---

## Design: Free-Running Mediators with LazyRemoteTensor

### Two Blocking Points

The mediator's worker thread runs the user's intervention function continuously. It only blocks for two reasons:

1. **Local module access** — the mediator posts a VALUE event on `event_queue` and waits on `response_queue`. The response comes from a hook during the forward pass. **Requires the interleaver to be open.**

2. **LazyRemoteTensor materialization** — the mediator calls `_materialize()` which does an RPC pull from the source rank's listener. Blocks until the tensor arrives. **Does NOT require the interleaver.**

Everything else runs freely: PPMissing module access (returns LazyRemoteTensor instantly), no-op writes, no-op saves, pure torch computation.

### PPMissing Access at the Envoy Level

When the mediator accesses `.output` on a PPMissing module, the Envoy **short-circuits** — instead of posting an event on `event_queue` and blocking, it returns a `LazyRemoteTensor` directly:

```python
# In the Envoy's .output access path:
if is_pp_missing(module):
    iteration = mediator.iteration_tracker[module_path]
    provider_string = f"{module_path}.output.i{iteration}"
    mediator.iteration_tracker[module_path] += 1
    return LazyRemoteTensor(
        source_rank=pp_module_map.get_owning_rank(provider_string),
        provider_string=provider_string,
        shape=..., dtype=..., device=...,
    )
# else: normal event_queue path (blocks until hook fires)
```

This bypasses the interleaver entirely for PPMissing modules. The mediator continues running without blocking. The iteration tracker is incremented at the Envoy level.

### Parallel Execution: The Key Win

Because mediators run freely for PPMissing accesses, remote operations overlap with other ranks' forward passes:

```
Rank 0                               Rank 1 mediator (running freely)
──────                               ─────────────────────────────────
forward pass running                  layer 0  → PPMissing → LazyRemoteTensor (instant)
  layer 0 hook → buffer              layer 1  → PPMissing → LazyRemoteTensor (instant)
  layer 1 hook → buffer              ...
  ...                                 layer 39 → PPMissing → LazyRemoteTensor (instant)
  layer 39 hook → buffer              h = lazy_0 * 2 → materialize → RPC pull
                                        ↑ rank 0 listener serves from buffer → unblocks
sends IT to rank 1 ──────────────▶    model.layers[40].output → LOCAL → blocks on event_queue
                                      ────── waiting for forward pass ──────

recv IT
forward pass starts                   layer 40 hook fires → mediator unblocks → continues
  layer 40 hook                       ...
  ...
```

Rank 1's mediator processes all 40 remote layers **while rank 0 is still running its forward pass**. No serialized drain of sequential RPCs. The mediator arrives at its first local module access (`layer 40`) and parks there, ready for the hook.

### Readiness Check

Before firing the forward pass hooks, the interleaver checks that each mediator is ready — its program counter is at a local module access (has a pending VALUE event in `event_queue`):

```python
# At the start of each interleaver session, before forward pass:
for mediator in interleaver.mediators:
    while mediator.alive and not mediator.event_queue.has_value:
        brief_yield()  # mediator is still processing PPMissing accesses
    # Now mediator is parked at a local module — safe to fire hooks
```

Since PPMissing accesses bypass `event_queue`, any event in the queue must be for a local module. The check is cheap in the common case — the mediator has already processed all PPMissing accesses and is waiting.

### Timeline: Multi-Token Generation (PP=2, Rank 0)

```
Token 0:
  interleaver opens
  readiness check: mediator just started, first access is layer 0 (local) → ready
  forward pass: layers 0-39 hooks fire (i0) → mediator processes
  interleaver closes
  ── mediator continues freely ──
  layer 50 → PPMissing → LazyRemoteTensor (instant, no-op write)
  logits  → PPMissing → LazyRemoteTensor (instant, no-op save)
  mediator loops → layer 0 i1 → LOCAL → blocks on event_queue
  send IT to rank 1

Token 1:
  interleaver opens
  readiness check: event_queue has layer 0 i1 → ready
  forward pass: layers 0-39 hooks fire (i1) → mediator processes
  interleaver closes
  ── mediator continues freely ──
  layer 50 → LazyRemoteTensor (instant)
  logits  → LazyRemoteTensor (instant)
  mediator loops → layer 0 i2 → blocks
  send IT to rank 1

...repeats...
```

### Timeline: Multi-Token Generation (PP=2, Rank 1)

```
Token 0:
  recv IT from rank 0
  interleaver opens
  readiness check: mediator just started, first access is layer 0 (PPMissing)
    → Envoy returns LazyRemoteTensor (instant)
    → mediator continues: layers 1-39 (all PPMissing, instant)
    → h = lazy * 2 → materialize → RPC pull from rank 0 (already done) → unblocks
    → layer 40 → LOCAL → posts event → parked
    → event_queue.has_value → ready
  forward pass: layers 40-79 hooks fire (i0) → mediator processes
  interleaver closes
  ── mediator continues freely ──
  logits → local (last rank) → already processed by hook
  mediator loops → layer 0 i1 → PPMissing → LazyRemoteTensor
    → materialize → RPC pull from rank 0
    → blocks until rank 0's token 1 forward pass produces the value
    → ... eventually unblocks ...
    → layers 1-39 → PPMissing → instant
    → layer 40 → LOCAL → blocks on event_queue

Token 1:
  recv IT from rank 0
  interleaver opens
  readiness check: event_queue has layer 40 i1 → ready
  forward pass: layers 40-79 hooks fire (i1)
  ...
```

---

## LazyRemoteTensor

A proxy returned for PPMissing module accesses. Materializes only when the value is actually consumed:

```python
class LazyRemoteTensor:
    def __init__(self, source_rank, provider_string, shape, dtype, device):
        self._meta = {
            'source_rank': source_rank,
            'provider_string': provider_string,
            'shape': shape,
            'dtype': dtype,
            'device': device,
        }
        self._real = None

    def _materialize(self):
        if self._real is None:
            self._real = pull_from_listener(
                self._meta['source_rank'],
                self._meta['provider_string'],
            )
        return self._real

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs):
        args = tree_map(
            lambda x: x._materialize() if isinstance(x, cls) else x, args
        )
        return func(*args, **(kwargs or {}))

    def __setitem__(self, key, value): pass   # absorb writes, zero transfer
    def __getitem__(self, key): return self    # chained indexing
    def save(self): return self                # no-op on non-owning rank

    @property
    def shape(self): return self._meta['shape']
    @property
    def dtype(self): return self._meta['dtype']
    @property
    def device(self): return self._meta['device']
```

**Behavior by operation:**

| Operation | Behavior | Transfer? |
|-----------|----------|-----------|
| `lazy[:] = X` | `__setitem__` → no-op | No |
| `lazy[0][:] = X` | `__getitem__` → self, `__setitem__` → no-op | No |
| `lazy.save()` | no-op (owning rank saves real value) | No |
| `lazy.shape` | returns metadata | No |
| `lazy * 2` | `__torch_function__` → materialize → RPC pull | Yes |
| `torch.cat([lazy, x])` | `__torch_function__` → materialize | Yes |

---

## pp_hook_buffer

Each rank's `NNsightGPUModelRunner` maintains a buffer of cloned hook values:

```python
self.pp_hook_buffer: dict[str, torch.Tensor] = {}
# e.g., "model.layers.5.output.i0" → cloned tensor
```

**Populated** in `handle_value_event()` — when a local hook matches a mediator's request, clone the raw value into the buffer.

**Why clone**: the original tensor lives in the forward pass computation graph and may be overwritten by subsequent layers. A clone survives independently.

**Lifetime**: accumulates across tokens (keyed by `provider.iN`), cleared at request finish in `collect_nnsight()`.

**Stays local**: never sent to other ranks. The listener serves from it on demand.

---

## Listener Thread

Each rank runs a background listener thread for the entire request lifetime. It serves pull requests from other ranks' `LazyRemoteTensor._materialize()` calls.

### Pull Protocol

```
Consumer (rank N)                    Producer listener (rank M)
─────────────────                    ────────────────────────────
send(provider_string)  ───────────►  recv(provider_string)
                                     wait until key in pp_hook_buffer
                                       (threading.Condition)
                                     lookup tensor
recv(shape_meta)       ◄───────────  send(shape, dtype)
recv(tensor)           ◄───────────  send(tensor)
```

The listener waits via `threading.Condition` if the value isn't in `pp_hook_buffer` yet. This handles the case where materialization is requested before the source rank's forward pass produces the value (e.g., rank 1 pulling from rank 0 while rank 0 is mid-forward-pass — the condition variable is notified when the clone-to-buffer happens in `handle_value_event`).

### Lifecycle

- **Starts**: when the first NNsight request arrives (in `process_new_reqs`)
- **Stops**: at `collect_nnsight()` cleanup when the request finishes
- **Thread safety**: `pp_hook_buffer` is written by the main thread (during `handle_value_event`) and read by the listener thread (serving pulls). Protected by the same `threading.Condition` used for wait/notify.

---

## Iteration Tracking

The iteration tracker (`mediator.iteration_tracker`) counts per-module iterations. It's incremented in two places:

1. **Local modules**: incremented in `interleaver.handle()` after the hook fires (`iterate=True`). This is the existing behavior.

2. **PPMissing modules**: incremented at the Envoy level when the LazyRemoteTensor is created. The Envoy reads `mediator.iteration_tracker[module_path]`, constructs the provider string with the current count, then increments.

Both paths keep the tracker in sync. At any point, `iteration_tracker[module_path]` equals the number of times that module has been accessed by this mediator, regardless of whether it's local or PPMissing.

---

## Save Collection

- `LazyRemoteTensor.save()` is a no-op — does not register in `Globals.saves`
- Owning rank saves real tensors via the normal `.save()` mechanism
- `collect_nnsight` via `collective_rpc` gathers saves from ALL ranks
- Merge: `{**rank0_saves, **rank1_saves}` — later ranks override (owning rank wins for duplicates)
- Filter out any unmaterialized `LazyRemoteTensor` in `collect_saves` as safety net

---

## Module-to-PP-Rank Mapping

`PPModuleMap` determines which PP stage owns a module:

- **Layers**: `get_pp_indices(num_hidden_layers, rank, world_size)` returns `(start, end)` per rank. Layer container names: `layers`, `h`, `block`, `blocks`.
- **Non-layer modules**: `embed_tokens`/`wte`/`wpe` → first rank. `norm`/`lm_head`/`ln_f`/`logits`/`samples` → last rank.
- Computed once at model load in `NNsightGPUModelRunner.load_model()`.

---

## PP-Aware Mediator Deserialization

Mediators serialized on the client reference the full meta model. On PP workers, modules on other stages are `PPMissingLayer` stubs without children. `_pp_aware_load` falls back to the nearest `PPMissingLayer` ancestor for missing paths (e.g., `model.h.6.ln_1` → `model.h.6` stub).

---

## What Changes From Current Code

**Remove:**
- END event injection for future-stage PPMissing in `handle_value_event`/`handle_swap_event`
- Eager `pp_hook_buffer` exchange (`send_tensor_dict`/`recv_tensor_dict`) in `GPUWorker`
- `pp_received_buffer` (no eager transfers)
- `make_dummy_tensor()`

**Add:**
- `LazyRemoteTensor` class
- PPMissing short-circuit in Envoy's `.output`/`.input` access path
- Listener thread per rank (serves from local `pp_hook_buffer`, `threading.Condition`)
- Readiness check at start of each interleaver session
- `threading.Condition` notify in `handle_value_event` buffer clone

**Keep:**
- `pp_hook_buffer` with clone-on-consume (data store for listener)
- `PPModuleMap`, `is_pp_missing`
- `_pp_aware_load` (PP-aware deserialization)
- Save collection from all ranks via `collective_rpc`

---

## Non-Determinism

Both ranks run the same mediator code independently. Non-deterministic operations (`torch.randn`, etc.) produce different values on each rank. This is safe:

- **Writes**: only take effect on the owning rank (PPMissing writes are no-ops via LazyRemoteTensor)
- **Saves**: `LazyRemoteTensor.save()` is a no-op; only the owning rank's save is collected
- **Reads**: materialized values come from the owning rank's buffer (deterministic per-rank)

---

## Limitations

- **PP + TP**: TP rank 0 per PP stage handles pulls. Not yet tested.
- **NCCL threading**: listener thread and main thread both do NCCL. May need separate process groups or careful stream management.
- **Buffer growth**: `pp_hook_buffer` accumulates across tokens. For long generations, this grows linearly. Acceptable for decode (small tensors); may need eviction for very long sequences.
