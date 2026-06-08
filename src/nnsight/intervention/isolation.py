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
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from . import serialization
from .transport import CudaIpcHostChannel, CudaIpcWorkerChannel
from ..util import apply

# Types that cross_invoker may ship between workers (data, not framework objects).
_XINVOKE_SCALARS = (int, float, complex, bool, str, bytes, type(None))


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
_STATE: Dict[str, Any] = {
    "on": False,
    "arena_bytes": 64 << 20,
    "gpu_mem_fraction": 0.3,
    "device": "cuda",
    "timeout": 60.0,    # per-step wall-clock cap on user code (hang containment)
    "lockdown": False,  # functional-first; seccomp lockdown enabled separately
    "pool_size": 0,     # 0 => cold one-shot worker per trace; >0 => warm pool cap
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
):
    """Run interventions inside ``with model.trace(...)`` in an isolated GPU worker.

    Footguns in user intervention code (infinite loops, OOM allocations, device-side
    asserts, host-object pokes) are contained to the worker; the model server keeps
    serving.

    Args:
        timeout: per-step wall-clock cap on user code; a worker that produces no
            event within ``timeout`` is presumed hung and killed (the host survives).
        pool_size: if > 0, draw workers from a process-global warm pool capped at
            ``pool_size`` (auto-grown lazily, persists across traces, falls back to a
            cold one-shot worker past the cap). 0 (default) spawns a cold worker per
            trace — the original behavior. The pool's base options (device,
            arena_bytes, gpu_mem_fraction, lockdown) are fixed when it is first
            warmed; use :func:`warm_worker_pool` to pre-warm at startup.
    """
    prev = dict(_STATE)
    _STATE.update(
        on=True,
        arena_bytes=arena_bytes,
        gpu_mem_fraction=gpu_mem_fraction,
        device=device,
        timeout=timeout,
        lockdown=lockdown,
        pool_size=pool_size,
    )
    try:
        yield
    finally:
        _STATE.update(prev)


def _base_opts() -> Dict[str, Any]:
    """The per-worker (warm-time) options, distinct from per-job (per-trace) ones."""
    return {
        "device": _STATE["device"],
        "arena_bytes": _STATE["arena_bytes"],
        "gpu_mem_fraction": _STATE["gpu_mem_fraction"],
        "lockdown": _STATE["lockdown"],
        "timeout": _STATE["timeout"],
    }


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

    def __init__(self, proc, buf, conn, channel, poolable: bool):
        self.proc = proc
        self.buf = buf
        self.conn = conn
        self.channel = channel
        self.poolable = poolable        # False => one-shot cold worker (never recycled)
        self.registered: set = set()    # requesters whose host-side hook is registered
        self.path2envoy: Optional[dict] = None
        self.clean = False              # True once this job ended via a consumed END

    def send_job(self, payload, extras, opts) -> None:
        self.conn.send(("job", payload, extras, opts))

    def reset_for_release(self) -> None:
        """Clear per-job host state so the worker can serve the next trace."""
        self.registered = set()
        self.path2envoy = None
        self.clean = False
        self.channel.reset()

    def close(self) -> None:
        """Stop the worker, escalating SIGTERM->SIGKILL for a wedged CUDA/C call."""
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


class _WorkerPool:
    """Process-global pool of warm generic workers, persisting across traces.

    Thread-safe (mediator starts are sequential today, but a server may run several
    traces); ``acquire`` hands out an idle worker or lazily grows the pool up to the
    cap, with a cold one-shot worker as the past-cap fallback so correctness never
    blocks on the budget.
    """

    def __init__(self):
        self._idle: deque = deque()
        self._all: set = set()
        self._base_opts: Optional[dict] = None
        self._lock = threading.Lock()

    def _remember_base_opts(self, base_opts: dict) -> dict:
        # The pool's base options are fixed the first time it is warmed/grown; later
        # traces with different base options reuse the existing warm workers.
        if self._base_opts is None:
            self._base_opts = dict(base_opts)
        return self._base_opts

    def warm(self, n: int, base_opts: dict) -> None:
        base = self._remember_base_opts(base_opts)
        # Spawn outside the lock (each ~4 s); register under it.
        need = max(0, n - len(self._all))
        fresh = [_spawn_worker(base, poolable=True) for _ in range(need)]
        with self._lock:
            for w in fresh:
                self._all.add(w)
                self._idle.append(w)

    def acquire(self, base_opts: dict, cap: int) -> _PooledWorker:
        base = self._remember_base_opts(base_opts)
        with self._lock:
            if self._idle:
                return self._idle.popleft()
            grow = len(self._all) < cap
        if grow:
            w = _spawn_worker(base, poolable=True)  # spawn outside the lock
            with self._lock:
                self._all.add(w)
            return w
        # At cap with none idle: a cold one-shot worker so the trace never blocks.
        return _spawn_worker(base, poolable=False)

    def put_idle(self, w: _PooledWorker) -> None:
        with self._lock:
            if w in self._all:
                self._idle.append(w)

    def forget(self, w: _PooledWorker) -> None:
        with self._lock:
            self._all.discard(w)
            try:
                self._idle.remove(w)
            except ValueError:
                pass

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._all)
            self._all.clear()
            self._idle.clear()
            self._base_opts = None
        for w in workers:
            w.close()


_POOL = _WorkerPool()


def warm_worker_pool(
    size: int,
    device: str = "cuda",
    arena_bytes: int = 64 << 20,
    gpu_mem_fraction: float = 0.3,
    lockdown: bool = False,
    timeout: float = 60.0,
) -> None:
    """Pre-warm ``size`` generic workers (blocks until each is ready).

    Call once at server startup so the first request pays no spawn cost. The base
    options here fix the pool's per-worker configuration. Each worker costs ~0.55 GiB
    GPU per GPU it touches — size the pool as a GPU-memory budget.
    """
    _POOL.warm(
        size,
        {
            "device": device,
            "arena_bytes": arena_bytes,
            "gpu_mem_fraction": gpu_mem_fraction,
            "lockdown": lockdown,
            "timeout": timeout,
        },
    )


def shutdown_worker_pool() -> None:
    """Stop and free all pooled workers (e.g. at server shutdown)."""
    _POOL.shutdown()


def _spawn_worker(base_opts: dict, poolable: bool) -> _PooledWorker:
    """Spawn a generic worker, wait for its one-time ``ready`` ack, wire the channel."""
    ctx = mp.get_context("spawn")  # CUDA requires spawn, not fork
    buf = torch.empty(
        base_opts["arena_bytes"], dtype=torch.uint8, device=base_opts["device"]
    )
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_pool_worker_main, args=(child_conn, buf, base_opts), daemon=True
    )
    proc.start()
    # The worker warms CUDA + imports (~4 s) then sends exactly one "ready"; consume
    # it before the channel starts reading protocol frames on the same pipe.
    startup = base_opts.get("startup_timeout", 180.0)
    if not parent_conn.poll(startup):
        proc.terminate()
        raise TimeoutError(
            f"isolated worker failed to warm up within {startup}s"
        )
    msg = parent_conn.recv()
    if msg != "ready":
        proc.terminate()
        raise RuntimeError(f"unexpected isolated-worker handshake: {msg!r}")
    chan = CudaIpcHostChannel(parent_conn, buf, timeout=base_opts["timeout"])
    return _PooledWorker(proc, buf, parent_conn, chan, poolable=poolable)


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

    from .. import CONFIG

    worker_opts = {
        # default_all (= generate's max_new_tokens) bounds an open-ended iter[:] on
        # the worker; it is set AFTER spawn so the live value is also piggybacked on
        # each response (meta), but seed the job with the value known now.
        "default_all": mediator.interleaver.default_all,
        # cross_invoker matches the in-process gate (Mediator.start): multiple invokes
        # + config enabled. The worker can't share a frame, so it pushes/pulls through
        # the host store (see _run_one_job + meta below).
        "cross_invoker": (
            len(mediator.interleaver.mediators) > 1 and CONFIG.APP.CROSS_INVOKER
        ),
    }
    return payload, extras, worker_opts


def _wire_host_channel(mediator, iso: _PooledWorker) -> None:
    """Point the (possibly recycled) worker's host channel at THIS mediator."""
    chan = iso.channel
    chan.reset()                         # fresh single-slot buffer + startup-timeout
    chan._timeout = _STATE["timeout"]    # per-trace user-code cap
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
    pool_size = _STATE.get("pool_size", 0)
    if pool_size > 0:
        iso = _POOL.acquire(_base_opts(), pool_size)
    else:
        iso = _spawn_worker(_base_opts(), poolable=False)
    iso.send_job(payload, extras, worker_opts)
    _wire_host_channel(mediator, iso)


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

    if iso.path2envoy is None:
        model = mediator.interleaver.tracer.model
        iso.path2envoy = {e.path: e for e in model.modules()}
    envoy = iso.path2envoy.get(path)
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


def _run_one_job(channel, payload, extras, opts, device) -> None:
    """Deserialize one mediator against fresh dummies, run its intervention, and ship
    saves at END. Any failure (including a bad payload) is reported as an EXCEPTION
    event so the host never waits on a worker that won't speak."""
    from .interleaver import Events
    from .tracing.globals import Globals

    try:
        Globals.saves.clear()  # per-job reset (the only worker-side global state)

        interleaver = _WorkerInterleaver(default_all=opts.get("default_all"))
        mediator = serialization.loads(payload, _WorkerPersistent(interleaver, extras))
        mediator.channel = channel
        mediator.interleaver = interleaver
        mediator.idx = 0
        # cross_invoker matches the host gate; var sharing rides the host store since
        # worker frames aren't shared across processes.
        mediator.cross_invoker = opts.get("cross_invoker", False)
        mediator._isolated_worker = True  # so Barrier sends the target count (host counts)
        interleaver.current = mediator

        def _apply_meta(m):
            # Live host state piggybacked on each response: the iter[:] bound and the
            # cross_invoker var store (pulled into the frame so push()/pull() see it).
            interleaver.default_all = m.get("default_all", interleaver.default_all)
            store = m.get("xinvoke_store")
            if store:
                # Store tensors travel CPU-serialized (see _push_locals); move them
                # back to the worker's device before the user code uses them.
                restored = apply(store, lambda t: t.to(device), torch.Tensor)
                mediator.info.frame.f_locals.update(restored)

        channel.on_meta = _apply_meta

        def _push_locals():
            # cross_invoker: ship this worker's *data* locals to the host store.
            # push() (called by send() before put_event) has already written them into
            # the SerializedFrame's f_locals. We ship only transmittable data (tensors
            # + basic types/containers) — framework objects (Barrier/Envoy, which hold
            # the model) are skipped; the worker already has them via its own closure.
            # Tensors are moved to CPU: a worker tensor cloned from the CUDA-IPC bounce
            # buffer cannot be re-shared over IPC by the host.
            if not mediator.cross_invoker:
                return None
            out = {}
            for k, v in mediator.info.frame.f_locals.items():
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

        channel.push_provider = _push_locals

        # Worker→host saves transmission: bundle .save()'d values into the END event.
        # The intervention's compiled body calls ``mediator.end()`` on success; push()
        # populates the SerializedFrame's f_locals, which we filter by Globals.saves.
        def _end():
            mediator.push()
            flocals = mediator.info.frame.f_locals
            saved = {k: v for k, v in flocals.items() if id(v) in Globals.saves}
            mediator.channel.put_event((Events.END, saved))

        mediator.end = _end

        _orig_exception = mediator.exception
        mediator.exception = lambda e: _orig_exception(_transmissible_exc(e))

        mediator.intervention(mediator, mediator.info, *mediator.args)
    except BaseException as e:  # noqa: BLE001 — contain the footgun; report it
        try:
            channel.put_event((Events.EXCEPTION, _transmissible_exc(e)))
        except Exception:  # noqa: BLE001
            pass


def _pool_worker_main(conn, buf, base_opts):
    """Generic worker: warm CUDA + imports ONCE, optionally lock down, then loop
    serving ``("job", payload, extras, opts)`` messages until told to ``"stop"``.

    The CUDA context, warmed kernels, bounce buffer, and channel persist across jobs —
    that is what the warm pool amortizes. Per job, a fresh mediator is deserialized
    against fresh dummy modules (no cross-job state but ``Globals.saves``, cleared)."""
    from .tracing.globals import _ensure_mounted

    device = base_opts.get("device", "cuda")

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

    if base_opts.get("gpu_mem_fraction") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(base_opts["gpu_mem_fraction"])

    channel = CudaIpcWorkerChannel(conn, buf)  # persistent; rebinds handlers per job

    # Footgun containment: after CUDA is warm and base imports are done (both may open
    # files), seccomp-block new fs/net/exec syscalls. Under the pool this locks the
    # import set for ALL jobs (jobs whose user code triggers a NEW import will fail) —
    # lockdown defaults off; document the trade-off. CUDA + the control Pipe + the IPC
    # buffer use already-open fds.
    if base_opts.get("lockdown"):
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
