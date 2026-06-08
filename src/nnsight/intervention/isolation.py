"""Transparent isolated execution of mediators in a spawned GPU worker process.

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

Public surface: :func:`isolate_mediators` (context manager) + :func:`isolation_state`.
``Mediator.start`` calls :func:`spawn_isolated_worker` when isolation is on; the host
``handle`` loop calls :func:`ensure_isolated_provider` (host-side hook registration).
"""
from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Dict, Optional

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
):
    """Run interventions inside ``with model.trace(...)`` in an isolated GPU worker.

    Footguns in user intervention code (infinite loops, OOM allocations, device-side
    asserts, host-object pokes) are contained to the worker; the model server keeps
    serving.

    Args:
        timeout: per-step wall-clock cap on user code; a worker that produces no
            event within ``timeout`` is presumed hung and killed (the host survives).
    """
    prev = dict(_STATE)
    _STATE.update(
        on=True,
        arena_bytes=arena_bytes,
        gpu_mem_fraction=gpu_mem_fraction,
        device=device,
        timeout=timeout,
        lockdown=lockdown,
    )
    try:
        yield
    finally:
        _STATE.update(prev)


# --------------------------------------------------------------------------- #
# Host side                                                                    #
# --------------------------------------------------------------------------- #
class _IsoHandle:
    """Per-mediator host-side handle to its worker process + bounce buffer."""

    def __init__(self, proc, buf, conn):
        self.proc = proc
        self.buf = buf
        self.conn = conn
        self.registered: set = set()      # Host-side hook registration: requesters whose hook is registered
        self.path2envoy: Optional[dict] = None

    def close(self):
        try:
            self.conn.send("stop")
        except Exception:  # noqa: BLE001
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()  # SIGTERM
            self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()  # SIGKILL — for a worker wedged in a non-interruptible
            self.proc.join(timeout=5)  # CUDA/C call that ignored SIGTERM


def spawn_isolated_worker(mediator) -> None:
    """Serialize ``mediator``, spawn its GPU worker, and wire the host channel.

    Sets ``mediator.channel`` (host end), ``mediator.worker`` (the process, so
    ``alive`` is True), and ``mediator._iso`` (the handle used by host-side hook registration + cancel).
    """
    opts = _STATE
    model = mediator.interleaver.tracer.model
    # Module:* and Interleaver are synthesized on the worker; ship the rest
    # (Tokenizer/Processor) so the deserialized graph resolves them. Only
    # remoteable models (LanguageModel/VLM/...) carry those extras; a plain
    # NNsight(module) has none.
    from ..modeling.mixins.remoteable import RemoteableMixin

    extras = {}
    if isinstance(model, RemoteableMixin):
        real_map = model._remoteable_persistent_objects()
        extras = {
            k: v
            for k, v in real_map.items()
            if not k.startswith("Module:") and k != "Interleaver"
        }

    # The tracer attaches source during ITS __getstate__; per-mediator
    # serialization must do the same first (else source is unavailable).
    mediator.intervention.__source__ = "".join(mediator.info.source)
    payload = serialization.dumps(mediator)

    # Per-spawn options: base config + the host interleaver's default_all
    # (= generate's max_new_tokens), which an open-ended `tracer.iter[:]` needs
    # to bound its step loop on the worker side.
    from .. import CONFIG

    worker_opts = dict(opts)
    worker_opts["default_all"] = mediator.interleaver.default_all
    # cross_invoker matches the in-process gate (Mediator.start): multiple invokes
    # + config enabled. The worker can't share a frame, so it pushes/pulls through
    # the host store (see _worker_main + on_push/meta below).
    worker_opts["cross_invoker"] = (
        len(mediator.interleaver.mediators) > 1 and CONFIG.APP.CROSS_INVOKER
    )

    ctx = mp.get_context("spawn")  # CUDA requires spawn, not fork
    buf = torch.empty(opts["arena_bytes"], dtype=torch.uint8, device=opts["device"])
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_worker_main,
        args=(payload, extras, child_conn, buf, worker_opts),
        daemon=True,
    )
    proc.start()

    chan = CudaIpcHostChannel(parent_conn, buf, timeout=opts["timeout"])
    # default_all is set by generate() AFTER the worker spawns (LanguageModel
    # ._execute), so a spawn-time snapshot is stale. Piggyback the LIVE value +
    # the cross_invoker var store on each response; the worker reads default_all
    # before bounding its `iter[:]` loop and pulls the store before each access.
    chan.meta_provider = lambda: {
        "default_all": mediator.interleaver.default_all,
        "xinvoke_store": mediator.interleaver._xinvoke_store,
    }
    # Merge a worker's pushed cross_invoker locals into the shared host store.
    chan.on_push = mediator.interleaver._xinvoke_store.update
    mediator.channel = chan
    mediator.worker = proc
    mediator._iso = _IsoHandle(proc, buf, parent_conn)

    # Host-side iteration tracking. The worker's `tracer.iter[...]` loop runs on
    # DUMMY modules, so its iteration_tracker never advances; the HOST must bump
    # ITS tracker per forward pass so the host-side hook registration's per-step hooks fire on the right
    # generation step (multi-token). Harmless for single-forward traces (one bump).
    # Registered on mediator.hooks => torn down by cancel/remove_hooks.
    from .tracing.iterator import register_iter_hooks

    register_iter_hooks(mediator, model)


def ensure_isolated_provider(mediator, requester: str) -> None:
    """Host-side hook registration: register the one-shot hook for ``requester`` on the *real* module.

    Parses ``"<path>.<output|input>.i<N>"``, resolves the envoy on the host, and
    registers the existing ``output_hook``/``input_hook`` with the host mediator for
    the **specific step N** parsed from the requester — the worker's iteration counter
    lives in another process, so the step must come from the wire, not from the host
    mediator's ``iteration``. The host's own iteration_tracker (bumped per forward by
    the iter hooks installed in ``spawn_isolated_worker``) advances so the N-hook fires
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


def _worker_main(payload, extras, conn, buf, opts):
    """Spawn target: deserialize the mediator against dummies, run its intervention,
    ship saves at END, then stay alive until the host releases the shared buffer."""
    from .interleaver import Events
    from .tracing.globals import Globals, _ensure_mounted

    device = opts.get("device", "cuda")

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

    interleaver = _WorkerInterleaver(default_all=opts.get("default_all"))
    mediator = serialization.loads(payload, _WorkerPersistent(interleaver, extras))
    mediator.channel = CudaIpcWorkerChannel(conn, buf)
    mediator.interleaver = interleaver
    mediator.idx = 0
    # cross_invoker matches the host gate; var sharing rides the host store (below)
    # since worker frames aren't shared across processes.
    mediator.cross_invoker = opts.get("cross_invoker", False)
    mediator._isolated_worker = True  # so Barrier sends the target count (host counts)
    interleaver.current = mediator
    Globals.saves.clear()

    def _apply_meta(m):
        # Live host state piggybacked on each response: the iter[:] bound and the
        # cross_invoker var store (pulled into the frame so push()/pull() see it).
        interleaver.default_all = m.get("default_all", interleaver.default_all)
        store = m.get("xinvoke_store")
        if store:
            # Store tensors travel CPU-serialized (see _push_locals); move them back
            # to the worker's device before the user code uses them.
            restored = apply(store, lambda t: t.to(device), torch.Tensor)
            mediator.info.frame.f_locals.update(restored)

    mediator.channel.on_meta = _apply_meta

    def _push_locals():
        # cross_invoker: ship this worker's *data* locals to the host store. push()
        # (called by send() before put_event) has already written them into the
        # SerializedFrame's f_locals. We ship only transmittable data (tensors +
        # basic types/containers) — framework objects (Barrier/Envoy, which hold the
        # model) are skipped; the worker already has them via its own closure. Tensors
        # are moved to CPU: a worker tensor cloned from the CUDA-IPC bounce buffer
        # cannot be re-shared over IPC by the host ("received from another process").
        if not mediator.cross_invoker:
            return None
        out = {}
        for k, v in mediator.info.frame.f_locals.items():
            if str(k).startswith("__nnsight"):
                continue
            if not _transmittable(v):
                # A referenced cross-invoke var that can't cross (framework object or
                # a container with one) is skipped — warn so it's not silently lost.
                import warnings

                warnings.warn(
                    f"cross_invoker: variable {k!r} ({type(v).__name__}) is not "
                    f"transmittable across the isolation boundary and was not shared "
                    f"between invokes."
                )
                continue
            out[k] = apply(v, lambda t: t.detach().cpu(), torch.Tensor)
        return out

    mediator.channel.push_provider = _push_locals

    # Worker→host saves transmission: bundle .save()'d values into the END event. The intervention's
    # compiled body calls ``mediator.end()`` on success; push() populates the
    # SerializedFrame's f_locals, which we filter by Globals.saves.
    def _end():
        mediator.push()
        # info.frame is always a SerializedFrame here (deserialized), which always
        # has f_locals; push() just populated it. Direct access (no getattr-default).
        flocals = mediator.info.frame.f_locals
        saved = {k: v for k, v in flocals.items() if id(v) in Globals.saves}
        mediator.channel.put_event((Events.END, saved))

    mediator.end = _end

    # Plain user exceptions (e.g. ValueError) pickle across the EXCEPTION event and
    # the host wraps them. But *dynamic* nnsight exceptions (NNsightException) don't
    # pickle ("Can't pickle nnsight.NNsightException") — degrade them to a plain
    # RuntimeError preserving type name + message so the host still reports cleanly.
    from multiprocessing.reduction import ForkingPickler

    def _transmissible_exc(e):
        try:
            ForkingPickler.dumps(e)
            return e
        except Exception:  # noqa: BLE001
            return RuntimeError(f"{type(e).__name__}: {e}")

    _orig_exception = mediator.exception
    mediator.exception = lambda e: _orig_exception(_transmissible_exc(e))

    if opts.get("gpu_mem_fraction") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(opts["gpu_mem_fraction"])

    # Footgun containment: after CUDA is warm and the intervention is deserialized
    # (both may open files), seccomp-block new fs/net/exec syscalls. User code runs
    # next; CUDA + the control Pipe + the IPC buffer use already-open fds.
    if opts.get("lockdown"):
        from ._sandbox import lock_down

        lock_down()

    try:
        mediator.intervention(mediator, mediator.info, *mediator.args)
    except BaseException as e:  # noqa: BLE001 — contain the footgun; report it
        try:
            mediator.channel.put_event((Events.EXCEPTION, _transmissible_exc(e)))
        except Exception:  # noqa: BLE001
            pass

    # Stay alive (keeps the shared GPU buffer mapped) until the host has consumed
    # END/saves and releases us.
    try:
        while True:
            if conn.recv() == "stop":
                break
    except (EOFError, OSError):
        pass

    # Skip interpreter atexit handlers: under seccomp lockdown, tempfile's atexit
    # rmtree hits blocked openat/unlink and recurses. The worker is disposable and
    # the host owns the bounce buffer, so a hard exit is correct here.
    os._exit(0)
