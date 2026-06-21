"""Transparent isolated execution of mediators in spawned GPU worker processes.

This is the *outer harness* for the chosen GPU-sandbox design (see
docs/developing/mediator-gpu-trace-integration.md). It does NOT change the
six-event Mediator protocol — it runs the Mediator's intervention in a separate,
GPU-enabled process and routes the protocol over a :class:`CudaIpcChannel`.

Two shared-memory assumptions of the in-process path become explicit harness steps:

- **Host-side hook registration**: the worker has no real module, so when the host
  receives a ``VALUE``/``SWAP``/``SKIP`` for a requester it hasn't seen, it registers
  the matching one-shot hook on the *real* module (resolved from the requester string).
- **Worker→host saves transmission**: ``.save()`` values live in the worker's frame +
  ``Globals.saves``; the worker bundles them into the ``END`` event and the host
  ``push_variables`` them into the real user frame.

**Warm worker pool.** Spawning a worker per request costs ~4 s of process bring-up
(cold ``import torch`` + ``import nnsight`` + CUDA context init — measured, model-
independent; see ``prototypes/.../perf_spawn_cost.py``). To amortize it, a worker is
*generic*: it holds a CUDA context + bounce buffer but no mediator, and receives a
serialized mediator per **job** over the channel it already owns (the ~3 ms payload
is the only per-request serialization). A process-global :class:`_WorkerPool` keeps
warm workers across traces; a worker that ends a job cleanly is reset and recycled,
one that is timed-out/killed/cancelled-mid-protocol is retired and lazily re-warmed.

Each warm worker costs ~0.55 GiB GPU (CUDA context + cuBLAS kernels) **per GPU it
touches**, model-weight-independent and NOT reduced by MPS — so the pool cap is a
real GPU-memory budget, with a cold one-shot fallback past the cap.

Public surface: :func:`isolate_mediators` (context manager, ``pool_size=`` opt-in),
:func:`warm_worker_pool` / :func:`shutdown_worker_pool`, and :func:`isolation_state`.
``Mediator.start`` calls :func:`acquire_isolated_worker` when isolation is on; the
host ``handle`` loop calls :func:`ensure_isolated_provider` (host-side hook
registration); ``Mediator.cancel`` calls :func:`release_isolated_worker`.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from . import serialization
from .interleaver import Mediator
from .transport import CudaIpcHostChannel, CudaIpcWorkerChannel
from ..util import apply

# Types that cross_invoker may ship between workers (data, not framework objects).
_XINVOKE_SCALARS = (int, float, complex, bool, str, bytes, type(None))

# Extra wall-clock allowed for a job's first event beyond the user `timeout`, to cover
# deserializing the mediator on an already-warm worker (the spawn+warm is paid before
# the worker's "ready" ack, so the first event only covers deserialize + run-to-first).
_JOB_STARTUP_MARGIN = 30.0


def _transmittable(v) -> bool:
    """True if ``v`` is cross_invoker-shareable data: a tensor, a basic scalar, or a
    container recursively of those. Framework objects (Barrier/Envoy/model) are not —
    the worker already has them via its own closure/deserialization."""
    if torch.is_tensor(v) or isinstance(v, _XINVOKE_SCALARS):
        return True
    # Note: no `set` — util.apply can't walk sets, so a set holding a CUDA tensor
    # would skip the D2H move and crash on the host's IPC re-share. Lists/tuples ok.
    if isinstance(v, (list, tuple)):
        return all(_transmittable(x) for x in v)
    if isinstance(v, dict):
        return all(_transmittable(k) and _transmittable(x) for k, x in v.items())
    return False


# --------------------------------------------------------------------------- #
# Opt-in surface                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IsoOptions:
    """Per-worker isolation options, in one place for every consumer (the opt-in
    context, the pool key, spawn, and the worker bootstrap — previously four
    hand-copied dicts).

    ``device``/``arena_bytes``/``gpu_mem_fraction``/``lockdown``/``preimport`` are
    **warm-time**: fixed when the worker process spawns, and they define pool
    interchangeability (:attr:`pool_key`). ``timeout`` is **per-job**: re-applied to the
    host channel each trace in ``_wire_host_channel`` (it also sizes the channel's
    first-event budget at spawn).
    """

    device: str = "cuda"
    arena_bytes: int = 64 << 20
    gpu_mem_fraction: float = 0.3
    lockdown: bool = False   # functional-first; seccomp lockdown enabled separately
    # Modules to import at worker warm time, before seccomp lockdown freezes new file
    # opens (import == open()); warm-time, so part of pool_key.
    preimport: tuple = ()
    timeout: float = 60.0    # per-step wall-clock cap on user code (hang containment)
    # Fast lane: confirmed-safe interventions run IN-PROCESS (full model access, no
    # worker, no per-hook channel) instead of in the GPU worker. Without it, isolation
    # cannot run the weight-reading interp majority at all (the worker is weightless).
    fast_lane: bool = True
    trust: str = "local"     # only "local" provenance is fast-lane-eligible
    fast_lane_timeout: float = 120.0  # whole-intervention watchdog bound for the fast lane

    @property
    def pool_key(self) -> tuple:
        # Workers are interchangeable ONLY within the same warm-time signature — the
        # bounce buffer is device- and size-specific, so reusing a worker across
        # devices would copy into the wrong-device buffer (silent corruption).
        return (
            str(self.device),
            int(self.arena_bytes),
            float(self.gpu_mem_fraction),
            bool(self.lockdown),
            tuple(sorted(self.preimport)),
        )


# Generous wait for a COLD spawn+warm (import torch/nnsight + CUDA init, ~4 s typical);
# distinct from the per-job first-event budget (user timeout + deserialize margin).
_WARM_STARTUP_TIMEOUT = 180.0

_STATE: Dict[str, Any] = {
    "on": False,
    "pool_size": 0,  # 0 => cold one-shot worker per trace; >0 => warm pool cap
    "opts": IsoOptions(),
}


def isolation_state() -> Dict[str, Any]:
    return _STATE


@contextmanager
def isolate_mediators(
    arena_bytes: int = 64 << 20,
    gpu_mem_fraction: float = 0.3,
    device: str = "cuda",
    timeout: float = 60.0,
    lockdown: bool = False,
    pool_size: int = 0,
    fast_lane: bool = True,
    trust: str = "local",
    fast_lane_timeout: float = 120.0,
    preimport: tuple = (),
):
    """Run interventions inside ``with model.trace(...)`` in an isolated GPU worker.

    Footguns in user intervention code (infinite loops, OOM allocations, device-side
    asserts, host-object pokes) are contained to the worker; the model server keeps
    serving.

    Args:
        timeout: per-step wall-clock cap on user code; a worker that produces no
            event within ``timeout`` is presumed hung and killed (the host survives).
        pool_size: if > 0, draw workers from a process-global warm pool capped at
            ``pool_size`` per (device, arena_bytes, gpu_mem_fraction, lockdown,
            preimport) signature (auto-grown lazily, persists across traces, falls back
            to a cold one-shot worker past the cap). 0 (default) spawns a cold worker per
            trace — the original behavior. Use :func:`warm_worker_pool` to pre-warm at
            startup. Under ``lockdown=True`` the worker freezes its import set at warm
            time (cold and pooled share the unified worker, which locks down before any
            job deserializes), so a job whose user code triggers a NEW import fails — the
            cold-vs-pool difference is recycle-vs-retire, not lockdown timing.
        fast_lane: if True (default), a per-mediator static classifier
            (:mod:`nnsight.intervention.fastlane`) confirms interventions that use only
            whitelisted ops / host-model access / nnsight primitives and runs THOSE
            in-process (full model + weights, no worker, no per-hook channel), isolating
            only the unconfirmable remainder. This is what lets isolation run the
            weight-reading interp majority at all — the worker holds weightless dummy
            modules. The override only ever moves a mediator from isolate to in-process;
            set ``fast_lane=False`` to force pure isolation.
        trust: only ``"local"`` provenance is fast-lane-eligible; the static gate is a
            footgun selector, not a malice boundary, so any other value disables the fast
            lane wholesale (everything isolates).
        fast_lane_timeout: whole-intervention wall-clock bound for a fast-laned thread
            (a best-effort watchdog restoring loop-containment in-process).
        preimport: module names to import at worker warm time, before seccomp lockdown
            freezes new file opens (import == ``open()``). Lets isolated interventions use
            those modules under ``lockdown=True``, bringing import capability to parity
            with an in-process whitelist; also part of the pool signature.
    """
    prev = dict(_STATE)
    _STATE.update(
        on=True,
        pool_size=pool_size,
        opts=IsoOptions(
            device=device,
            arena_bytes=arena_bytes,
            gpu_mem_fraction=gpu_mem_fraction,
            lockdown=lockdown,
            timeout=timeout,
            fast_lane=fast_lane,
            trust=trust,
            fast_lane_timeout=fast_lane_timeout,
            preimport=tuple(preimport),
        ),
    )
    try:
        yield
    finally:
        _STATE.update(prev)


def fast_lane_enabled() -> bool:
    """True if confirmed-safe mediators should run in-process. Gated on the context
    option, the ``trust="local"`` provenance cordon, and the global config flag (a server
    can force pure isolation without code changes)."""
    opts = _STATE["opts"]
    if not opts.fast_lane or opts.trust != "local":
        return False
    try:
        from .. import CONFIG

        return bool(getattr(CONFIG.APP, "FAST_LANE", True))
    except Exception:  # noqa: BLE001 — config unavailable => default-on
        return True


def classify_for_fast_lane(mediator):
    """Run the static classifier on ``mediator`` (host-side, before any serialization).
    Returns a :class:`fastlane.Verdict`. Cached on the intervention code object so a
    re-run of the same trace pays the walk once."""
    from . import fastlane

    code = getattr(mediator.intervention, "__code__", None)
    cache = _FASTLANE_VERDICT_CACHE
    if code is not None and code in cache:
        return cache[code]
    verdict = fastlane.classify(mediator)
    if code is not None:
        cache[code] = verdict
    return verdict


# Verdict cache keyed by the intervention code object identity (per-trace-shape, stable
# across re-runs of the same trace); bounded implicitly by the number of distinct traces.
_FASTLANE_VERDICT_CACHE: Dict[Any, Any] = {}


# --------------------------------------------------------------------------- #
# Host side — worker handle + pool                                             #
# --------------------------------------------------------------------------- #
class _PooledWorker:
    """Host-side handle to a warm, *generic* GPU worker process.

    Holds the live process, its bounce buffer, and the host channel — all reused
    across jobs. Also carries the per-job host-side hook-registration state
    (``registered`` / ``path2envoy``) that :func:`ensure_isolated_provider` reads,
    and a ``clean`` flag gating recycle (set when this job's END is consumed).
    """

    def __init__(self, proc, buf, conn, channel, poolable: bool, key: tuple = None):
        self.proc = proc
        self.buf = buf
        self.conn = conn
        self.channel = channel
        self.poolable = poolable        # False => one-shot cold worker (never recycled)
        self.key = key                  # base-opts signature (device, ...) for pool keying
        self.registered: set = set()    # requesters whose host-side hook is registered
        self.path2envoy: Optional[dict] = None
        self.clean = False              # True once this job ended via a consumed END

    def send_job(self, payload, extras, opts) -> None:
        self.conn.send(("job", payload, extras, opts))

    def reset_for_release(self) -> None:
        """Drop per-job references so an idle worker doesn't pin them (the hook set,
        the path→envoy map, and — via ``channel.reset()`` — the meta/push callbacks
        closing over the last trace's interleaver). The *authoritative* per-job reset
        happens at acquire time in ``_wire_host_channel``; this is memory hygiene."""
        self.registered = set()
        self.path2envoy = None
        self.channel.reset()

    def close(self) -> None:
        """Stop the worker, escalating SIGTERM->SIGKILL for a wedged CUDA/C call,
        then release the host-side pipe fd + GPU bounce buffer (else they linger
        until GC)."""
        try:
            self.conn.send("stop")
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()  # SIGTERM
            self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()  # SIGKILL — wedged in a non-interruptible CUDA/C call
            self.proc.join(timeout=5)
        try:
            self.channel.close()  # closes the pipe fd
        except Exception:  # noqa: BLE001
            pass
        self.buf = None  # drop the host ref so the GPU arena can be reclaimed


class _WorkerPool:
    """Process-global pool of warm generic workers, persisting across traces.

    Thread-safe (mediator starts are sequential today, but a server may run several
    traces); ``acquire`` hands out an idle worker or lazily grows the pool up to the
    cap, with a cold one-shot worker as the past-cap fallback so correctness never
    blocks on the budget.
    """

    def __init__(self):
        # Keyed by IsoOptions.pool_key — the warm-time signature (device, arena_bytes,
        # gpu_mem_fraction, lockdown, preimport): workers are interchangeable ONLY within
        # it. The bounce buffer is device- and size-specific (reusing across devices would
        # copy into the wrong-device buffer — silent corruption), and a worker's frozen
        # import set must match the requested preimport list. See IsoOptions.
        self._idle: Dict[tuple, deque] = defaultdict(deque)
        self._all: Dict[tuple, set] = defaultdict(set)
        self._lock = threading.Lock()
        self._shutting_down = False

    def warm(self, n: int, opts: IsoOptions) -> None:
        key = opts.pool_key
        with self._lock:
            need = max(0, n - len(self._all[key]))
        # Spawn outside the lock (each ~4 s); register under it.
        fresh = [_spawn_worker(opts, poolable=True) for _ in range(need)]
        with self._lock:
            for w in fresh:
                self._all[key].add(w)
                self._idle[key].append(w)

    def acquire(self, opts: IsoOptions, cap: int) -> _PooledWorker:
        key = opts.pool_key
        dead: list = []
        live = None
        placeholder = None
        with self._lock:
            idle, allset = self._idle[key], self._all[key]
            # Skip workers that died while idle (OOM-killed by a neighbor, crashed):
            # forget them now and close below, so a dead worker is never handed out.
            while idle and live is None:
                w = idle.popleft()
                if w.proc.is_alive():
                    live = w
                else:
                    allset.discard(w)
                    dead.append(w)
            if live is None and len(allset) < cap:
                # Reserve the slot under the lock so concurrent acquires can't grow
                # past the cap (the spawn itself happens outside the lock).
                placeholder = object()
                allset.add(placeholder)
        for w in dead:
            w.close()
        if live is not None:
            return live
        if placeholder is not None:
            try:
                w = _spawn_worker(opts, poolable=True)
            except BaseException:
                with self._lock:
                    self._all[key].discard(placeholder)
                raise
            with self._lock:
                self._all[key].discard(placeholder)
                self._all[key].add(w)
            return w
        # At cap with none idle: a cold one-shot worker so the trace never blocks.
        return _spawn_worker(opts, poolable=False)

    def put_idle(self, w: _PooledWorker) -> None:
        close_it = False
        with self._lock:
            if self._shutting_down or w not in self._all[w.key]:
                # Released during shutdown, or forgotten while checked out: don't
                # re-pool — close it so the process can't leak (close is idempotent if
                # it was already torn down).
                close_it = True
            else:
                self._idle[w.key].append(w)
        if close_it:
            w.close()

    def forget(self, w: _PooledWorker) -> None:
        with self._lock:
            self._all[w.key].discard(w)
            try:
                self._idle[w.key].remove(w)
            except ValueError:
                pass

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            workers = [w for s in self._all.values() for w in s]
            self._all.clear()
            self._idle.clear()
        for w in workers:
            w.close()
        with self._lock:
            self._shutting_down = False


_POOL = _WorkerPool()


def warm_worker_pool(
    size: int,
    device: str = "cuda",
    arena_bytes: int = 64 << 20,
    gpu_mem_fraction: float = 0.3,
    lockdown: bool = False,
    timeout: float = 60.0,
    preimport: tuple = (),
) -> None:
    """Pre-warm ``size`` generic workers (blocks until each is ready).

    Call once at server startup so the first request pays no spawn cost. The base
    options here fix the pool's per-worker configuration. Each worker costs ~0.55 GiB
    GPU per GPU it touches — size the pool as a GPU-memory budget. ``preimport`` and
    ``lockdown`` must match the values later passed to :func:`isolate_mediators` or the
    pre-warmed workers won't match its pool signature and fresh ones are spawned.
    """
    _POOL.warm(
        size,
        IsoOptions(
            device=device,
            arena_bytes=arena_bytes,
            gpu_mem_fraction=gpu_mem_fraction,
            lockdown=lockdown,
            timeout=timeout,
            preimport=tuple(preimport),
        ),
    )


def shutdown_worker_pool() -> None:
    """Stop and free all pooled workers (e.g. at server shutdown)."""
    _POOL.shutdown()


def _spawn_worker(opts: IsoOptions, poolable: bool) -> _PooledWorker:
    """Spawn a generic worker, wait for its one-time ``ready`` ack, wire the channel."""
    ctx = mp.get_context("spawn")  # CUDA requires spawn, not fork
    buf = torch.empty(opts.arena_bytes, dtype=torch.uint8, device=opts.device)
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_pool_worker_main, args=(child_conn, buf, opts), daemon=True
    )
    proc.start()
    # The worker warms CUDA + imports (~4 s) then sends exactly one "ready"; consume
    # it before the channel starts reading protocol frames on the same pipe. This poll
    # covers the cold spawn+warm, so it stays generous.
    if not parent_conn.poll(_WARM_STARTUP_TIMEOUT):
        proc.terminate()
        raise TimeoutError(
            f"isolated worker failed to warm up within {_WARM_STARTUP_TIMEOUT}s"
        )
    msg = parent_conn.recv()
    if msg != "ready":
        proc.terminate()
        raise RuntimeError(f"unexpected isolated-worker handshake: {msg!r}")
    # The channel's first-event budget covers only a job's deserialize + run-to-first-
    # request (spawn+warm already happened above), so it is the user timeout plus a
    # deserialize margin — NOT the cold 180 s, which would defeat hang-containment on a
    # job that hangs before its first event on an already-warm worker.
    chan = CudaIpcHostChannel(
        parent_conn,
        buf,
        timeout=opts.timeout,
        startup_timeout=opts.timeout + _JOB_STARTUP_MARGIN,
    )
    return _PooledWorker(
        proc, buf, parent_conn, chan, poolable=poolable, key=opts.pool_key
    )


def _build_job(mediator) -> tuple:
    """Serialize ``mediator`` into a job message ``(payload, extras, worker_opts)``.

    Module:* and Interleaver are synthesized on the worker; ship the rest
    (Tokenizer/Processor) so the deserialized graph resolves them. Only remoteable
    models (LanguageModel/VLM/...) carry those extras; a plain NNsight(module) has none.
    """
    model = mediator.interleaver.tracer.model
    from ..modeling.mixins.remoteable import RemoteableMixin

    extras = {}
    if isinstance(model, RemoteableMixin):
        real_map = model._remoteable_persistent_objects()
        extras = {
            k: v
            for k, v in real_map.items()
            if not k.startswith("Module:") and k != "Interleaver"
        }

    # The tracer attaches source during ITS __getstate__; per-mediator serialization
    # must do the same first (else source is unavailable).
    mediator.intervention.__source__ = "".join(mediator.info.source)
    payload = serialization.dumps(mediator)

    worker_opts = {
        # default_all (= generate's max_new_tokens) bounds an open-ended iter[:] on
        # the worker; it is set AFTER spawn so the live value is also piggybacked on
        # each response (meta), but seed the job with the value known now.
        "default_all": mediator.interleaver.default_all,
        # Mediator.start already applied the in-process cross_invoker gate (multiple
        # invokes + config) before acquiring the worker; reuse its decision. The
        # worker can't share a frame, so it pushes/pulls through the host store.
        "cross_invoker": bool(mediator.cross_invoker),
        # `with tensor.backward()` detection — the single decision point for BOTH
        # sides: the host gates real-activation retention, the worker gates
        # delivered-clone tagging. Prefer the fast-lane classifier's closure-aware flag
        # (it resolves through build()/capture() closures the substring is blind to);
        # fall back to the source substring when the classifier did not run.
        "backward_active": _backward_active(mediator),
    }
    return payload, extras, worker_opts


def _backward_active(mediator) -> bool:
    """Closure-aware `with tensor.backward()` detection for the isolated job, preferring
    the fast-lane classifier's verdict (which walks through user closures) over the
    source substring (blind to a backward hidden in a build()/capture() closure)."""
    verdict = getattr(mediator, "_fastlane_verdict", None)
    if verdict is not None:
        return verdict.differentiate
    return ".backward(" in mediator.intervention.__source__


def _wire_host_channel(mediator, iso: _PooledWorker, worker_opts: dict) -> None:
    """Point the (possibly recycled) worker's host channel at THIS mediator."""
    chan = iso.channel
    chan.reset()                            # fresh single-slot buffer + startup-timeout
    chan._timeout = _STATE["opts"].timeout  # per-trace user-code cap
    # default_all is set by generate() AFTER the worker is acquired (LanguageModel
    # ._execute), so a snapshot is stale. Piggyback the LIVE value + the cross_invoker
    # var store on each response; the worker reads default_all before bounding its
    # iter[:] loop and pulls the store before each access.
    chan.meta_provider = lambda: {
        "default_all": mediator.interleaver.default_all,
        "xinvoke_store": mediator.interleaver._xinvoke_store,
    }
    chan.on_push = mediator.interleaver._xinvoke_store.update
    mediator.channel = chan
    mediator.worker = iso.proc
    mediator._iso = iso
    # `with tensor.backward()`: if the trace differentiates, the host must retain each
    # delivered (real, on-graph) activation so handle_backward_event can run the real
    # backward. The decision was made once in _build_job (shared with the worker);
    # start with a fresh retention map.
    mediator._iso_backward = worker_opts["backward_active"]
    mediator._iso_grad_reals = {}
    # Fresh per-job host-side hook-registration state.
    iso.registered = set()
    iso.path2envoy = None
    iso.clean = False

    # Host-side iteration tracking. The worker's tracer.iter[...] loop runs on DUMMY
    # modules, so its iteration_tracker never advances; the HOST must bump ITS tracker
    # per forward pass so the per-step host-registered hooks fire on the right
    # generation step (multi-token). Harmless for single-forward traces (one bump).
    # Registered on mediator.hooks => torn down by cancel/remove_hooks.
    from .tracing.iterator import register_iter_hooks

    register_iter_hooks(mediator, mediator.interleaver.tracer.model)


def acquire_isolated_worker(mediator) -> None:
    """Acquire a worker (pool or cold), ship the job, and wire the host channel.

    Sets ``mediator.channel`` (host end), ``mediator.worker`` (the process, so
    ``alive`` is True), and ``mediator._iso`` (the handle used by host-side hook
    registration + cancel/release).
    """
    payload, extras, worker_opts = _build_job(mediator)
    pool_size = _STATE["pool_size"]

    def _acquire():
        if pool_size > 0:
            return _POOL.acquire(_STATE["opts"], pool_size)
        return _spawn_worker(_STATE["opts"], poolable=False)

    iso = _acquire()
    try:
        iso.send_job(payload, extras, worker_opts)
    except (BrokenPipeError, EOFError, OSError):
        # The worker died between the liveness check and dispatch (tiny race). Forget
        # it and retry once through the normal acquisition path with a fresh worker;
        # a second failure is a real problem and propagates.
        _POOL.forget(iso)
        iso = _acquire()
        iso.send_job(payload, extras, worker_opts)
    _wire_host_channel(mediator, iso, worker_opts)


def release_isolated_worker(iso: _PooledWorker, dirty: bool) -> None:
    """Recycle a cleanly-ended pooled worker; retire everything else.

    Recyclable only if the worker ended a job cleanly (``clean``, set when its END was
    consumed), is poolable, and is still alive. ``dirty`` (the host drained it
    mid-protocol with a Cancelation, leaving the pipe unbalanced), a timeout/death
    (``clean`` never set), or a one-shot cold worker => retire it; the pool re-warms
    lazily on the next acquire.
    """
    if (not dirty) and iso.poolable and iso.clean and iso.proc.is_alive():
        iso.reset_for_release()
        _POOL.put_idle(iso)
    else:
        iso.close()
        _POOL.forget(iso)


def path_to_envoy(mediator) -> dict:
    """The ``{path: envoy}`` map for this job's model, built lazily and cached on the
    worker handle (fresh per job — ``_wire_host_channel`` clears it). Shared by
    host-side hook registration and ``handle_cache_event``'s target resolution."""
    iso = mediator._iso
    if iso.path2envoy is None:
        model = mediator.interleaver.tracer.model
        iso.path2envoy = {e.path: e for e in model.modules()}
    return iso.path2envoy


def ensure_isolated_provider(mediator, requester: str) -> None:
    """Host-side hook registration: register the one-shot hook for ``requester`` on the *real* module.

    Parses ``"<path>.<output|input>.i<N>"``, resolves the envoy on the host, and
    registers the existing ``output_hook``/``input_hook`` with the host mediator for
    the **specific step N** parsed from the requester — the worker's iteration counter
    lives in another process, so the step must come from the wire, not from the host
    mediator's ``iteration``. The host's own iteration_tracker (bumped per forward by
    the iter hooks installed in ``_wire_host_channel``) advances so the N-hook fires
    on the right generation step. Idempotent per requester (per step).
    """
    iso = mediator._iso
    if requester in iso.registered:
        return
    iso.registered.add(requester)

    # The iteration suffix is always the rightmost ".i<digits>".
    parts = requester.rsplit(".i", 1)        # ["model.transformer.h.6.output", "2"]
    base = parts[0]
    iteration = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
    path, _, kind = base.rpartition(".")     # ("model.transformer.h.6", ".", "output")
    if kind not in ("output", "input"):
        return  # externally-provided eproperties (e.g. .result) need no module hook

    envoy = path_to_envoy(mediator).get(path)
    if envoy is None:
        return  # unknown path → let normal missed-provider handling surface it

    from .hooks import input_hook, output_hook

    if kind == "output":
        output_hook(mediator, envoy._module, base, iteration=iteration)
    else:
        input_hook(mediator, envoy._module, base, iteration=iteration)


# --------------------------------------------------------------------------- #
# Worker side                                                                  #
# --------------------------------------------------------------------------- #
class _WorkerBatcher:
    """Minimal stand-in for the Batcher that ``requires_*`` reads on the worker."""

    current_provider = None
    current_value = None


class _WorkerInterleaver:
    """Worker-side interleaver stub: enough for eproperty + requires_* to run.

    The real Batcher/narrow/swap all live on the host; the worker only builds
    requester strings and emits events.
    """

    def __init__(self, default_all=None):
        self.interleaving = True
        self.batcher = _WorkerBatcher()
        self.current = None
        # Open-ended `tracer.iter[:]` reads this to know how many generation steps
        # to run; set by generate(max_new_tokens=N) on the host and shipped over.
        self.default_all = default_all

    def iterate_requester(self, requester: str) -> str:
        med = self.current
        iteration = (
            med.iteration if med.iteration is not None else med.iteration_tracker[requester]
        )
        return f"{requester}.i{iteration}"


class _WorkerPersistent:
    """persistent_objects map for the worker: synthesize a dummy module for every
    ``Module:<path>`` (no weights), the worker interleaver for ``Interleaver``, and
    pass through everything else (Tokenizer/Processor)."""

    def __init__(self, interleaver, extras: dict):
        self._extras = dict(extras)
        self._extras["Interleaver"] = interleaver
        self._dummies: dict = {}

    def __contains__(self, key) -> bool:
        return str(key).startswith("Module:") or key in self._extras

    def __getitem__(self, key):
        skey = str(key)
        if skey.startswith("Module:"):
            d = self._dummies.get(skey)
            if d is None:
                d = nn.Module()
                d.__path__ = skey[len("Module:") :]
                self._dummies[skey] = d
            return d
        return self._extras[key]


def _transmissible_exc(e):
    """Degrade an exception to a form that pickles across the EXCEPTION event.

    Plain user exceptions (ValueError, ...) pickle and the host wraps them. *Dynamic*
    nnsight exceptions (NNsightException) don't ("Can't pickle nnsight.NNsightException")
    — degrade to a plain RuntimeError preserving type name + message.
    """
    from multiprocessing.reduction import ForkingPickler

    try:
        ForkingPickler.dumps(e)
        return e
    except Exception:  # noqa: BLE001
        return RuntimeError(f"{type(e).__name__}: {e}")


class WorkerMediator(Mediator):
    """The worker-process half of an isolated mediator.

    A job's mediator is deserialized as a plain :class:`Mediator` and then *adopted*
    (``__class__`` swap) into this subclass, which overrides exactly the methods whose
    in-process behavior relies on shared memory with the host:

    - ``end``: the worker's frame + ``Globals.saves`` live here, so the exit filter
      runs locally and the saved dict rides the END event (worker→host saves
      transmission).
    - ``exception``: dynamic nnsight exceptions don't pickle; degrade before shipping.
    - ``request``: when the trace contains ``with tensor.backward()``, tag each
      delivered activation clone (``requires_grad_`` + id→requester provenance) so the
      worker builds its half of the autograd graph and can seed the host backward.

    ``apply_meta`` / ``push_locals`` are the worker ends of the live host↔worker state
    piggyback (`iter[:]` bound + cross_invoker variable store) and are bound to the
    channel's ``on_meta`` / ``push_provider`` per job.
    """

    @classmethod
    def adopt(cls, mediator, channel, interleaver, opts: dict, device) -> "WorkerMediator":
        """Turn a freshly-deserialized mediator into this job's worker mediator."""
        mediator.__class__ = cls
        mediator.channel = channel
        mediator.interleaver = interleaver
        mediator.idx = 0
        # cross_invoker matches the host gate; var sharing rides the host store since
        # worker frames aren't shared across processes.
        mediator.cross_invoker = opts.get("cross_invoker", False)
        mediator._isolated_worker = True  # so Barrier sends the target count (host counts)
        mediator._device = device
        # `with tensor.backward()`: decided once in _build_job (host-side) and shipped
        # with the job; gates the delivered-clone tagging in ``request``
        # (BackwardsTracer reads the provenance).
        mediator._bwd_active = opts["backward_active"]
        mediator._bwd_prov = {}    # id(delivered clone) -> requester string
        mediator._bwd_tagged = []  # delivered clones made to require grad
        interleaver.current = mediator
        return mediator

    def request(self, requester: str):
        value = super().request(requester)
        if self._bwd_active:
            self._tag_delivered(value, requester)
        return value

    def _tag_delivered(self, value, requester: str) -> None:
        """Tag each delivered activation tensor with its requester provenance and make
        it require grad, so worker-side ops on it build the worker's half of the graph."""

        def _tag(t):
            if t.is_floating_point() and t.is_leaf:
                t.requires_grad_(True)
                self._bwd_prov[id(t)] = requester
                self._bwd_tagged.append(t)
            return t

        apply(value, _tag, torch.Tensor)

    def end(self):
        # Worker→host saves transmission: bundle .save()'d (and .carry()'d) values into the
        # END event. The intervention's compiled body calls ``end()`` on success; push()
        # populates the SerializedFrame's f_locals, which we filter by Globals.saves/shared.
        from .interleaver import Events, _ISO_CACHE_TAG
        from .tracing.globals import Globals
        from .tracing.tracer import Cache

        self.push()
        flocals = self.info.frame.f_locals
        saved = {k: v for k, v in flocals.items() if id(v) in Globals.saves}
        # Carried (.carry()) values: cross-trace handoffs within a session. Ship them too so
        # the host can write them to the session frame for the next trace, but tag which are
        # SAVED (saved_names) so the host surfaces only those to the user frame. With no
        # .carry() in play this is exactly the prior payload (saved only) — no regression for
        # the single-trace path.
        carried = {
            k: v for k, v in flocals.items()
            if id(v) in Globals.shared and id(v) not in Globals.saves
        }
        values = {**saved, **carried}
        # A tracer.cache() placeholder CacheDict is a live object, not a value — ship its
        # token as a plain-dict marker so the value codec accepts it; the host swaps in its
        # own forward-filled cache by token (top-level saves only, as the host handler is).
        for k, v in list(values.items()):
            if isinstance(v, Cache.CacheDict):
                values[k] = {_ISO_CACHE_TAG: getattr(v, "_iso_cache_token", None)}
        self.channel.put_event((Events.END, (values, list(saved.keys()))))

    def exception(self, exception: Exception):
        super().exception(_transmissible_exc(exception))

    def apply_meta(self, m: dict) -> None:
        # Live host state piggybacked on each response: the iter[:] bound and the
        # cross_invoker var store (pulled into the frame so push()/pull() see it).
        self.interleaver.default_all = m.get(
            "default_all", self.interleaver.default_all
        )
        store = m.get("xinvoke_store")
        if store:
            # Store tensors travel CPU-serialized (see push_locals); move them
            # back to the worker's device before the user code uses them.
            restored = apply(store, lambda t: t.to(self._device), torch.Tensor)
            self.info.frame.f_locals.update(restored)

    def push_locals(self) -> Optional[dict]:
        # cross_invoker: ship this worker's *data* locals to the host store.
        # push() (called by send() before put_event) has already written them into
        # the SerializedFrame's f_locals. We ship only transmittable data (tensors
        # + basic types/containers) — framework objects (Barrier/Envoy, which hold
        # the model) are skipped; the worker already has them via its own closure.
        # Tensors are moved to CPU: a worker tensor cloned from the CUDA-IPC bounce
        # buffer cannot be re-shared over IPC by the host.
        if not self.cross_invoker:
            return None
        out = {}
        for k, v in self.info.frame.f_locals.items():
            if str(k).startswith("__nnsight"):
                continue
            if not _transmittable(v):
                import warnings

                warnings.warn(
                    f"cross_invoker: variable {k!r} ({type(v).__name__}) is not "
                    f"transmittable across the isolation boundary and was not "
                    f"shared between invokes."
                )
                continue
            out[k] = apply(v, lambda t: t.detach().cpu(), torch.Tensor)
        return out


# The job currently running in this worker process (one at a time). Read by
# ``worker_backward_context`` so BackwardsTracer can find the ambient mediator.
_WORKER_CURRENT: Optional[WorkerMediator] = None


def worker_backward_context() -> Optional[WorkerMediator]:
    """Return this worker's mediator if its trace contains a backward block, else None.

    Used by ``BackwardsTracer.execute`` to detect that ``.backward()`` is running in an
    isolated worker (so it must drive the host's real backward instead of differentiating
    its detached clones locally); the mediator carries the delivered-clone provenance
    (``_bwd_prov`` / ``_bwd_tagged``)."""
    med = _WORKER_CURRENT
    if med is not None and med._bwd_active:
        return med
    return None


def _run_one_job(channel, payload, extras, opts, device) -> None:
    """Deserialize one mediator against fresh dummies, adopt it as this job's
    :class:`WorkerMediator`, run its intervention, and ship saves at END. Any failure
    (including a bad payload) is reported as an EXCEPTION event so the host never
    waits on a worker that won't speak."""
    from .interleaver import Events
    from .tracing.globals import Globals

    global _WORKER_CURRENT
    try:
        Globals.saves.clear()  # per-job reset (the only worker-side global state)
        Globals.shared.clear()

        interleaver = _WorkerInterleaver(default_all=opts.get("default_all"))
        mediator = serialization.loads(payload, _WorkerPersistent(interleaver, extras))
        WorkerMediator.adopt(mediator, channel, interleaver, opts, device)
        _WORKER_CURRENT = mediator
        channel.on_meta = mediator.apply_meta
        channel.push_provider = mediator.push_locals

        mediator.intervention(mediator, mediator.info, *mediator.args)
    except BaseException as e:  # noqa: BLE001 — contain the footgun; report it
        try:
            channel.put_event((Events.EXCEPTION, _transmissible_exc(e)))
        except Exception:  # noqa: BLE001
            pass
    finally:
        _WORKER_CURRENT = None


def _pool_worker_main(conn, buf, worker_iso_opts: IsoOptions):
    """Generic worker: warm CUDA + imports ONCE, optionally lock down, then loop
    serving ``("job", payload, extras, opts)`` messages until told to ``"stop"``.

    The CUDA context, warmed kernels, bounce buffer, and channel persist across jobs —
    that is what the warm pool amortizes. Per job, a fresh mediator is deserialized
    against fresh dummy modules (no cross-job state but ``Globals.saves``, cleared)."""
    from .tracing.globals import _ensure_mounted

    device = worker_iso_opts.device

    # Warm CUDA before any lockdown so kernels/contexts are loaded.
    if torch.cuda.is_available():
        _ = (torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)).sum()
        torch.cuda.synchronize()

    # Warm imports user ops might trigger, before seccomp closes new file opens.
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    import cloudpickle  # noqa: F401

    cloudpickle.loads(cloudpickle.dumps(lambda _t: _t))

    _ensure_mounted()  # install Object.save so `.save()` resolves in the worker

    # User-configurable warm-time pre-imports: load modules interventions may need
    # BEFORE seccomp freezes new file opens (import == open()). Under lockdown a module
    # not loaded here is unimportable in any job, so pre-warming the deployment's
    # allowed-module set brings user-facing import capability to parity with an
    # in-process whitelist. A failed pre-import is non-fatal (warn + skip).
    if worker_iso_opts.preimport:
        import importlib
        import warnings

        for _mod in worker_iso_opts.preimport:
            try:
                importlib.import_module(_mod)
            except Exception as _e:  # noqa: BLE001
                warnings.warn(f"isolated worker pre-import of {_mod!r} failed: {_e!r}")

    if worker_iso_opts.gpu_mem_fraction and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(worker_iso_opts.gpu_mem_fraction)

    channel = CudaIpcWorkerChannel(conn, buf)  # persistent; rebinds handlers per job

    # Footgun containment: after CUDA is warm and base imports are done (both may open
    # files), seccomp-block new fs/net/exec syscalls. Under the pool this locks the
    # import set for ALL jobs (jobs whose user code triggers a NEW import will fail) —
    # lockdown defaults off; document the trade-off. CUDA + the control Pipe + the IPC
    # buffer use already-open fds.
    if worker_iso_opts.lockdown:
        from ._sandbox import lock_down

        lock_down()

    # One-time ready ack: the spawner consumes this before the channel reads protocol.
    conn.send("ready")

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg == "stop":
            break
        if not (isinstance(msg, tuple) and msg and msg[0] == "job"):
            continue  # ignore stray control messages
        _, payload, extras, opts = msg
        _run_one_job(channel, payload, extras, opts, device)
        # Job done; the worker is idle and recyclable. Loop for the next job/stop.

    # Skip interpreter atexit handlers: under seccomp lockdown, tempfile's atexit
    # rmtree hits blocked openat/unlink and recurses. The worker is disposable and
    # the host owns the bounce buffer, so a hard exit is correct here.
    os._exit(0)
