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
the next step overwrites: clone what you keep. And the callable's return is
discarded, so an edit has to land *in place*; in-place edits already do, and a
replacement swap is copied back into the live tensor, shape-checked.

Taps are the only locations a replayed step reaches. Every other module's
handoff lives in its forward, which a replayed graph never runs; a block parked
on one is told so when its request ends (see ``Requests.finish_dangling``).
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from ...intervention.interleaver import Interleaver
from ...util import apply


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
    """

    def __init__(self, taps: Iterable[str] = (), **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.taps = frozenset(taps)

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
