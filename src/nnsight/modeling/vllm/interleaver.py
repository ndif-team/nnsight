"""An interleaver whose hooks survive CUDA-graph replay, at declared taps.

A replayed CUDA graph runs no Python, so the forward hooks
[`Interleaver.instrument`][nnsight.intervention.interleaver.Interleaver.instrument]
installs never fire under ``enforce_eager=False``. vLLM's *breakable* graphs
(``VLLM_USE_BREAKABLE_CUDAGRAPH=1``) leave a seam: a callable handed to the
recording's ``add_eager`` while the graph is being captured is run at that point
of every replay, against the recording's own tensors. A tap is
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle]
registered as one of those callables — the same handoff a hook makes, replayed.

Two things follow from replay that an eager hook never meets. The value at a
location lives at a fixed address the graph rewrites every step, so a kept
reference — ``.save()``, a list appended under ``tracer.iter`` — aliases memory
the next step overwrites: clone what you keep, or set `CLONE_READS` (see
`VLLMInterleaver.handle`). And the callable's return is
discarded, so an edit has to land *in place*; in-place edits already do, and a
replacement swap is copied back into the live tensor, shape-checked.

Taps are the only locations a replayed step reaches. Every other module's
handoff lives in its forward, which a replayed graph never runs; a block parked
on one is told so when its request ends (see ``Requests.finish_dangling``).
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import torch

from ...intervention.interleaver import Interleaver
from ...util import apply

#: Env var: serve each read location from one private copy per visit.
CLONE_READS = "NNSIGHT_VLLM_CLONE_READS"

# The falsy spellings `NNSIGHT_DISABLE_CPP_BACKTRACE` accepts, so the two agree.
_FALSY = {"", "0", "false", "no", "off"}


def clone_reads_enabled() -> bool:
    """Whether `CLONE_READS` is set to anything but a falsy spelling.

    Read from the environment rather than `CONFIG` because the interleaver that
    makes the copy lives in the engine's *worker* process, built when the engine
    loads the model. A `CONFIG.APP` field set in the client process would never
    reach it; an env var set before `VLLM(...)` is inherited when the worker is
    spawned.
    """
    value = os.environ.get(CLONE_READS)
    return value is not None and value.strip().lower() not in _FALSY


def _capture() -> Any:
    """vLLM's breakable graph capture while it is recording, else ``None``."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    capture = BreakableCUDAGraphCapture.current()
    return capture if capture is not None and capture._capturing else None


def _weak(value: Any) -> Any:
    """Weak-ref the CUDA tensors in a value, as vLLM's own eager breaks do.

    A strong reference from the recorded callable pins the recording's pool slot,
    and replay would then serve the warmup's tensors instead of the step's.
    """
    try:
        from vllm.utils.torch_utils import weak_ref_tensor
    except ImportError:
        from vllm.utils import weak_ref_tensor
    return apply(value, lambda t: weak_ref_tensor(t) if t.is_cuda else t, torch.Tensor)


def _copy_into(live: Any, edited: Any) -> None:
    """Land ``edited`` in ``live``'s memory, tensor by tensor — the graph reads from there.

    What arrives here has ``live``'s shape by construction: a swap goes through
    the batcher's widen, which raises inside the worker's handoff (and so errors
    that request alone) if the replacement does not fit its rows.
    """
    if edited is live:
        return
    if isinstance(live, torch.Tensor):
        live.copy_(edited)
    elif isinstance(live, (tuple, list)):
        for live_item, edited_item in zip(live, edited):
            _copy_into(live_item, edited_item)
    elif isinstance(live, dict):
        for key, live_item in live.items():
            _copy_into(live_item, edited[key])


class VLLMInterleaver(Interleaver):
    """An [`Interleaver`][nnsight.intervention.interleaver.Interleaver] that records its handoff into CUDA graphs at ``taps``.

    Attributes:
        taps: Full locations (``"model.model.layers.3.output"``) a replayed step
            serves. Empty on an ``enforce_eager`` engine, where this class behaves
            exactly as its base.
        clone_reads: Whether `CLONE_READS` was set when the worker built this
            interleaver, which it does once, when the engine loads the model.
    """

    def __init__(self, taps: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.taps = frozenset(taps)
        self.clone_reads = clone_reads_enabled()

    def instrument(self, envoy: Any) -> None:
        """Route the module through this interleaver, and open its forward if a tap reaches inside.

        A tap on an operation — ``{path}.source.{op}.output`` — is served by the
        module's source-instrumented forward, which hands each operation to
        `handle` as it runs. That forward is installed lazily, on the first
        ``.source`` access, and the graph recording runs before any trace could
        ask for it; so install it here, as the tree is built, for exactly the
        modules the taps name — and check the op exists while a typo is still a
        load error rather than a request that parks forever.
        """
        super().instrument(envoy)
        prefix = f"{envoy.path}.source."
        ops = {tap[len(prefix):].rpartition(".")[0] for tap in self.taps if tap.startswith(prefix)}
        if not ops:
            return
        source = envoy.source  # installs the instrumented forward; raises SourceNotAvailable
        missing = sorted(op for op in ops if op not in source.names)
        if missing:
            raise ValueError(
                f"Tap on {envoy.path}.source: its forward has no operation "
                f"{missing[0]!r}. Its operations are {list(source.names)}."
            )

    def handle(self, provider: str, value: Any) -> Any:
        # While vLLM is recording a graph, a tap's handoff is not served now but
        # registered to be replayed at this point of every later step. The runner
        # enters the interleaver around that recording so the handoff reaches here.
        if provider in self.taps:
            capture = _capture()
            if capture is not None:
                live = _weak(value)
                capture.add_eager(lambda: self.replay(provider, live))
                return value
        elif self.taps and provider.rsplit(".", 1)[-1] in ("input", "output", "skip"):
            # A module location that is not a tap. A replayed graph never reaches
            # it — but a prompt too long for any graph runs eagerly through here,
            # and would be served on that step and not the next. Skipped either
            # way, so what a tapped engine serves does not depend on prompt length.
            return value
        # Under `CLONE_READS` a visit that a worker or cache reads is served from
        # one copy, so a kept read holds what was computed rather than what vLLM's
        # in-place kernels later write over the live buffer. One copy per visit,
        # not per read, and in-place edits to it are copied back. A fragment is
        # left alone: vLLM's collectives are out-of-place, so the gathered tensor
        # the workers read is already private.
        if (
            self.clone_reads
            and (provider in self.observers or (provider in self.parked and self._ready(provider)))
            and not (self.fragments is not None and self.fragments.fragmented(provider))
        ):
            private = apply(value, lambda t: t.clone(), torch.Tensor)
            edited = super().handle(provider, private)
            if edited is not private:
                return edited  # a swap, built from the copy and so carrying its edits
            _copy_into(value, private)
            return value
        return super().handle(provider, value)

    def replay(self, location: str, live: Any) -> None:
        """Serve ``location`` from a replayed graph: `handle`, then land any edit in place.

        A graph is recorded for a batch size, so its tensors carry that many rows
        and this step fills the first ``batcher.total`` of them; the rest is
        padding no worker owns. Trimmed to a view before the handoff, so a worker
        is narrowed out of the rows that exist — and an edit copied back into the
        view lands in the graph's memory all the same.
        """
        if not self.interleaving:
            return
        total = self.batcher.total
        live = apply(live, lambda t: t[:total] if t.dim() and t.shape[0] > total else t, torch.Tensor)
        _copy_into(live, self.handle(location, live))
