# Mediator Isolation Harness — Implementation Plan

**Status:** Plan (pre-implementation) · **Date:** 2026-06-05
**Builds on:** [mediator-isolation-sandbox.md](mediator-isolation-sandbox.md) (design + threat model),
`prototypes/mediator-sandbox/p2_isolation_poc.py` (the isolation proof-of-concept, passing).
**Goal of this doc:** turn "the jail is safe" into "a real trace runs *through* the jail," in testable
increments, with the exact nnsight seam identified.

---

## 1. The seam (where host and jail divide)

Three call sites in `src/nnsight/intervention/interleaver.py` define the entire boundary. Everything
above them is user code (→ jail); everything below is the model (→ host).

| Role | Symbol | What it does |
|---|---|---|
| **Client hook** | `eproperty.__get__` (`:275`) | `value = interleaver.current.request(requester)` — `.output` read |
| **Client hook** | `eproperty.__set__` (`:317`) | `interleaver.current.swap(...)` — `.output` write |
| **Worker launch** | `Mediator.start._worker_target` (`:981`) | `Thread(target=…)` runs `intervention(self, info, …)` — `self` is the Mediator |
| **Worker→host RPC** | `Mediator.send` (`:1313`) | `event_queue.put((event, requester))` → wait `response_queue` |
| **host→worker reply** | `Mediator.respond` (`:1299`) | `response_queue.put(value)` → wait `event_queue` |
| **the two queues** | `Mediator.Value` (`:823`), fields `:878-879` | lock-based one-slot handoff (the transport) |
| **server narrow/swap** | `Batcher.narrow` (`batching.py:198`), `Batcher.swap` (`:226`) | slice/replace the tenant's rows |

**Key fact:** the compiled intervention fn never calls the model directly. It only ever reaches
`interleaver.current.request/swap/skip` and reads scalar state (`iteration`, `history`, `transform`).
So the jail needs **no weights and no GPU** — only an Envoy *path tree* wired to a client-side mediator
that emits the six events (`VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION`, `Events` `:349`) over a socket.

---

## 2. Target architecture

```
HOST (trusted: GPU + weights)                         JAIL (untrusted: CPU-only, net=none, ro-fs)
  model + Envoy + weights                               deserialize request bytes  ← pickle lives here
  Interleaver + Batcher  ── admission bounds            client Envoy *path tree* (no modules)
  MediatorProxy (per request) ──────── socket ────────  client Interleaver stub  (.current = RemoteMediator)
    · runs handle_value_event/handle_swap_event           RemoteMediator.send() → socket
      against the REAL Batcher, narrowed to the           compiled intervention fn = USER CODE
      host-recorded bounds (NOT worker-supplied)          (torch-CPU on the delivered tensor)
    · D2H + serialize VALUE replies
    · deserialize + H2D SWAP tensors
```

- **`MediatorChannel`** — a new abstraction that replaces the `Mediator.Value` queue pair. Two impls:
  `InProcessChannel` (today's lock+slot) and `SocketChannel` (length-prefixed frames over an
  `AF_UNIX` socket). The Mediator's `send`/`respond` and the worker loop talk to a `MediatorChannel`,
  not to `event_queue`/`response_queue` directly.
- **`RemoteMediator`** (jail side) — implements the subset the intervention fn touches:
  `request/swap/skip/end/exception`, `iteration`, `transform`, `history`. `send()` writes a frame and
  blocks for the reply. **It exposes no `interleaver`, `batcher`, or sibling refs** — the capability
  leaks have nothing to reach.
- **`MediatorProxy`** (host side) — owns the real `Batcher` and the existing `handle_*` logic; per
  socket event it narrows/swaps against **host-recorded admission bounds**, so `batch_group`,
  `narrow(None)`, `[-1,_]`, and widening become unrepresentable from the jail.

---

## 3. What crosses the wire (the codec)

| Frame | Direction | Payload |
|---|---|---|
| `VALUE` req | jail→host | `requester` (provider path str) + iteration |
| `VALUE` reply | host→jail | the **narrowed activation as a CPU tensor** (D2H'd, serialized) |
| `SWAP` | jail→host | `requester` + the **modified CPU tensor** (serialized) |
| `SKIP` | jail→host | `requester` + sentinel/value |
| `BARRIER` | jail→host | participating mediator names |
| `END` / `EXCEPTION` | jail→host | none / serialized exception metadata |

Tensor codec: start with the dumb path (`torch.save`/raw buffer / `safetensors`) over the socket;
a shared-memory ring is a later optimization (out of scope for the harness). Values may be **tuples /
nested structures** (transformer block outputs) — the codec must round-trip `applyn`-style nested
containers, not just bare tensors.

---

## 4. Phased plan (each phase is independently testable)

### Isolation proof-of-concept ✅ DONE
`prototypes/mediator-sandbox/p2_isolation_poc.py` — the 10 escape gadgets run inside the jail and are
inert; socket-fd-into-`net=none`-jail mechanism verified; real PID isolation confirmed.

### `MediatorChannel` seam (in-process, zero behavior change) ✅ DONE
Extract the `event_queue`/`response_queue` handoff in `Mediator.send`/`respond`/`start` and the worker
loop behind a `MediatorChannel` interface; ship `InProcessChannel` as the default.
- **Touch:** `interleaver.py` — added `MediatorChannel`/`InProcessChannel`; all 23 queue call sites now
  route through `self.channel.*`.
- **Acceptance:** the *entire existing nnsight test suite passes unchanged.* — **met** (203 core tests
  green; independent review verdict MATCH).

### Socket transport for the protocol (process boundary, harness) ✅ DONE
The six-event protocol rides an `AF_UNIX` socket between a forked worker and the host, with the real
`Mediator`/`Batcher`. `SocketHostChannel` (host-local event buffer; only `wait_event` reads the socket)
and `SocketWorkerChannel` (`src/nnsight/intervention/transport.py`) + a length-prefixed pickle codec.
- **Proven** (harness `prototypes/mediator-sandbox/phase2_socket_transport.py`, all green):
  codec round-trip of nested tuples/dicts/tensors (the transport itself is **bit-exact**);
  **golden equivalence** — same `h[6]×2` intervention local vs over-socket on gpt2 → logits match within
  multi-threaded CPU forward noise (`allclose atol=1e-3`; residual ~1e-4 is two separate forward passes,
  not the socket — a broken swap is Δ≈30); **real batching**
  (`needs_batching=True`) — the worker receives only its `batch_group` row and its swap lands only on
  that row (no cross-row leak); **out-of-order `restore_event`** exercised at runtime over the socket.
- **Scope honesty:** this proves *the event protocol + the `provider==requester` routing/restore + real
  Batcher narrow/swap over a process boundary*. The requester string is a shared constant in the harness
  (the compare at `interleaver.py:1175` is exercised; independent eproperty-side *production* of that
  string rides in with the `model.trace()` integration). It does **not** yet route through
  `model.trace()` — see the transparent-trace-integration milestone below.

### Transparent `model.trace()` isolated execution (the trace integration)
Make `model.trace()` itself launch the worker in a separate process and return identical results — the
isolation *execution backend*, separated out because it is the bulk of the real feature.
- **`Mediator.start`** gains an isolated path: fork (the trace-integration milestone) / exec-the-jailed-worker (the jail-the-worker stage)
  instead of `Thread`; the host Mediator gets a `SocketHostChannel`, the worker a `SocketWorkerChannel`.
- **Saves transmission (keystone finding):** `.save()` marks ids in `Globals.saves` and the values live
  in the *worker's* address space (`tracing/base.py:567`). The worker must **ship saved values back to
  the host** (e.g. a final `SAVES` frame at END) and the host must inject them into the trace's return —
  otherwise an isolated trace returns empty saves.
- **Lifecycle parity:** `cancel()` must `waitpid`/`kill` the worker process (not drop a `Thread`);
  `check_dangling_mediators`, EarlyStop drain, and EXCEPTION/SKIP over the socket must all hold.
- **Acceptance:** `with model.trace(prompt): h[6].output.save()` under an `isolate_mediators()` context
  returns **bit-identical** saves to the in-process trace, across read / `+=` write / multi-layer /
  multi-token.

### Jail the worker ✅ DONE
Exec the protocol worker inside a `bwrap` jail (CPU-only, `net=none`, ro-fs allowlist), socket fd passed in.
- **Proven** (`prototypes/mediator-sandbox/phase3_jailed_worker.py` + `phase3_jail_transport.py`, green):
  (A) the real Mediator protocol worker, run *inside the jail*, produces matching gpt2 logits
  (`allclose atol=1e-3`; the transport codec is bit-exact, residual is forward nondeterminism);
  (B) the same jailed worker's escape gadgets are inert — `fs_read`→FileNotFoundError,
  `net_egress`→OSError, `os.system` touch creates **no host file**, secret **not** leaked — *while its
  legitimate request/swap protocol still completes bit-identically*. Host-level sibling = the host-level sibling jail (sandbox design §6.5).
- **Note:** the jailed worker runs a fixed script (not yet a host-shipped compiled intervention) and the
  requester is still a constant — shipping the real intervention is the GPU-path deserialize-in-jail work.

### Batcher as authority (cross-tenant / Boundary B) ✅ DONE
The narrow/swap bounds live on the **host** Mediator (the proxy); the jail's Mediator is a separate object
with no `interleaver`/`batcher` reference, so the worker physically cannot mutate the host's bounds or
reach a sibling. The 8 capability leaks (#1-6, #8; #7 pacing is out of scope) are inert **by construction**,
not by a signature change to `narrow`/`swap`.
- **Proven** (`prototypes/mediator-sandbox/phase4_malicious_worker.py` + `phase4_cross_tenant.py`, green):
  a malicious jailed tenant co-batched with a victim row attempts `batch_group=None`/`[-1,0]`/widen and
  walks to `interleaver.batcher`/`interleaver.mediators`. Result: all batch_group claims are
  "set-on-local-mediator-only" (no host effect); `has_interleaver=False`; both walks → `AttributeError`;
  the worker received **only its own row** (sum=12, not 96 → never saw the victim's secret); its poison
  landed **only on its own row** — the victim row stayed uncorrupted.
- **Note:** demonstrated with one malicious jail + a passive victim row (the property is symmetric for two
  jails); BARRIER cross-jail coordination remains deferred (§5).

### Real-model GPU path + measurement ✅ DONE
Real `D2H → socket → torch-CPU → H2D` round-trip with the model on an A100, worker CPU-only.
- **Proven** (`prototypes/mediator-sandbox/phase5_gpu_measure.py`, green): (A) the GPU D2H/H2D socket
  path matches the in-process GPU forward (`max|Δ|=0`, atol 1e-2); (B) **measured per-hook overhead**
  (TOTAL = D2H + sockRTT + H2D) on an A100 80GB PCIe: 1×16×768 (0.03 MB) **0.9 ms**; 1×512×4096 (4.2 MB)
  **30 ms**; 1×2048×4096 (16.8 MB, ~7B) **111 ms**; 1×2048×8192 (33.6 MB, ~70B) **376 ms** — vs a
  **measured** in-process baseline of **0.1–0.25 ms** (zero-copy reference).
- **Finding (the two-tier-decision input):** the cost is **dominated by pickle + socket RTT** (103 ms of the
  111 ms at 16.8 MB; pickle alone ~14 ms), *not* the PCIe D2H/H2D (~3.5 ms each). A 32-layer 7B cache ⇒
  ~3.5 s added. So the naive path is fine for small/few-layer interventions, but big-model per-layer work
  needs **the two-tier approach** (run the common tensor algebra host-side) and/or a **shared-memory ring** wire
  format — confirming §7.
- **Deferred:** deserialize the *actual* serialized intervention request in the jail (untrusted
  unpickle-in-jail). The security property is already shown (the isolation-PoC + jail stages: pickle `__reduce__` gadget inert
  in the jail); shipping the *real compiled intervention* is the trace-integration work.

### Transport breakdown: the cost is the CODEC, not the boundary ✅ DONE
The Phase-5 "sockRTT-dominated" number was really **pickle-dominated**. `prototypes/mediator-sandbox/
phase5b_transport_breakdown.py` isolates it (16.8 MB, CPU↔CPU, echo = pure transport, no user op):
- The socket moves 16.8 MB in **~10 ms**; the naive path pickles the tensor **4× per hook**
  (`dumps` ~14 ms + `loads` ~8 ms each way), and pickle scales **superlinearly** (285 ms at 33.6 MB).
- **Measured per-hook round-trip by wire codec:**

  | codec | 4.2 MB | 16.8 MB (~7B) | 33.6 MB (~70B) | speedup |
  |---|---|---|---|---|
  | **A pickle** (today, `transport.py`) | 12.5 ms | 95.9 ms | 285 ms | 1× |
  | **B raw message** (struct header + raw bytes via `memoryview`/`torch.frombuffer`; = `safetensors`) | 4.3 ms | 24.1 ms | 123 ms | **~4×** |
  | **C shared memory** (`mmap`, only a 1-byte signal on the socket; bulk never travels) | 8.5 ms | 11.5 ms | 28.4 ms | **~8×** (linear) |

- For **read-only** interventions (caching, logit lens — the common case) shmem is even cheaper: the host
  writes the activation once, the worker reads it zero-copy, only a small result returns.
- Separate finding: the worker's `×2` on **bf16 CPU is ~9 ms** by itself (the *user* op, not transport) —
  another argument for **the two-tier approach** (keep common ops on the GPU host-side; neither transport nor the
  slow CPU recompute happens).
- **Fix hierarchy:** (1) swap the codec pickle→`safetensors`/raw in `transport.py` — small, localized, 4×;
  (2) shared-memory ring for big-model per-layer work — 8×, linear; (3) **the two-tier approach** removes transport
  entirely for the common tensor algebra. The 111 ms was a codec artifact, **not** intrinsic to isolation.

### Shared-memory + safetensors transport ✅ DONE (built into `transport.py`)
`ShmArena` (anonymous `memfd`) + `ShmSocketHostChannel`/`ShmSocketWorkerChannel`: tensor bulk rides the
shared memfd, only a small control frame crosses the socket, tensors encoded with **safetensors** (safe,
no-code-exec — also closes the jail→host untrusted-deserialize hole for the bulk).
- **Proven** (`phase6_shm_safetensors.py` + `phase6_jailed_worker.py`, green): correct through the real
  `Mediator` protocol both **forked** and **jailed** (gpt2 golden `max|Δ|=0`; the memfd is passed into the
  bwrap jail via `pass_fds`+`SHM_FD`). Echo round-trip vs pickle: 16.8 MB **87→44 ms (~2×)**, 4.2 MB ~1.8×.
- **Why only ~2× (not the 8× of hand-rolled shmem):** `safetensors` copies on `save` AND `load` (it returns
  owned tensors), so the bulk is memcpy'd ~4×. The hand-rolled `frombuffer` path was zero-copy (a view into
  the arena) but that aliases the shared buffer — needs double-buffering / careful lifecycle. Safe-and-simple
  (copy) vs fastest (zero-copy view) is the next dial; both beat pickle.
- **Still pickle:** the small control skeleton (non-tensor structure + requester string). Hardening that to a
  restricted decoder is the remaining jail→host security item.
- **Bigger lever (open):** true zero-copy needs the data to not move at all — either **the two-tier approach**
  (op runs on the GPU host-side; nothing transported) or **GPU-in-jail + CUDA-IPC** (jail maps the host's GPU
  buffer; no D2H/serialize) — the latter trades the CPU-only security posture; see the GPU-access discussion.

---

## 5. Open design decisions (resolve as they're reached)

- **Client Envoy tree in the jail** — ship a path-only Envoy, or reconstruct from the deserialized
  request (the nnsight request already carries Envoy refs)? Lean: reconstruct from the request, since
  deserialize-in-jail produces it anyway.
- **`cross_invoker` var-sharing** (`send` push/pull, `:1327`/`:1410`) — today two invokes share a Python
  frame in-process. Across jails that frame doesn't exist. v1: **one jail per request, `cross_invoker`
  disabled**; revisit a host-mediated var store later.
- **`BARRIER`** coordinates multiple mediators (= multiple jails); the host-side `MediatorProxy` set
  becomes the coordinator. Defer multi-jail barriers past the cross-tenant stage.
- **CUDA stream propagation** (`start` captures the caller stream, `:967-977`) — irrelevant in a
  CPU-only jail; the host keeps all stream handling.
- **Tensor wire format** — raw/`safetensors` first; shmem ring is a perf follow-up, not correctness.

---

## 6. Risks

- **Per-hook latency** — the accepted cost; quantified in the GPU-path measurement, drives the two-tier approach.
- **Codec correctness** for nested/tuple values (transformer block outputs) — needs a robust
  `applyn`-aware codec, tested against the gotchas (tuple-output handling).
- **Lifecycle parity** — the in-process `cancel()`/worker-join must map cleanly to **SIGKILL the jail**;
  verify END/EXCEPTION/EarlyStop all drain correctly across the socket.
- **Deadlock surface** — the one-event-in-flight invariant must hold over the socket exactly as over the
  one-slot queue, or the "access in forward-pass order" contract breaks.

---

## 7. Test strategy (every phase)

1. **Golden equivalence** — in-process result == isolated result, on varied models/ops (not just gpt2;
   include a non-standard module-named model per the testing rules).
2. **Security** — the jail-the-worker stage runs the 10 escape gadgets as the intervention (inert); the cross-tenant stage runs the 8
   capability-leak payloads across two jails (blocked).
3. **Lifecycle** — exception propagation, early-stop, timeout→SIGKILL, normal drain.

The two existing ndif suites (`tests/security/test_protector_escapes.py`,
`test_mediator_capability.py`) are the security gates; each fix flips a row from 🔓 to blocked.
