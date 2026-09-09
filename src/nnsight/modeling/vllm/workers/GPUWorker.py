"""The one place nnsight gets into a vLLM worker process.

vLLM's worker builds its model runner in ``init_device``; once it has, the
runner's class is swapped for nnsight's subclass, which adds behaviour and no
constructor state. `_load` names this class as vLLM's ``worker_cls``, a supported
engine argument, so no part of vLLM's own startup is patched — and a runner
nnsight does not instrument is refused here, in the worker, rather than coming up
silently uninstrumented.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from ..model_runners.GPUModelRunner import NNsightGPUModelRunner


class NNsightGPUWorker(Worker):
    """A vLLM GPU worker whose model runner interleaves interventions."""

    def init_device(self) -> None:
        super().init_device()
        runner = self.model_runner
        if type(runner) is not GPUModelRunner:
            raise NotImplementedError(
                f"nnsight instruments vLLM's GPUModelRunner, but this worker built "
                f"{type(runner).__module__}.{type(runner).__name__} (the V2 runner, "
                "or a runner from another platform). Unset VLLM_USE_V2_MODEL_RUNNER "
                "to trace, or drop nnsight and use vLLM directly for that run."
            )
        runner.__class__ = NNsightGPUModelRunner

    def collect_nnsight(
        self,
        request_ids: list[str],
        finished_request_ids: Optional[list[str]] = None,
        outputs: Optional[Any] = None,
    ) -> Optional[bytes]:
        """Return this worker's saved values, as ``collective_rpc`` reaches it here.

        ``outputs`` arrives pickled (see ``NNsightLLMEngine.step``): the RPC is
        msgpack-encoded on the way in, and bytes are what it carries natively.
        """
        if isinstance(outputs, bytes):
            outputs = pickle.loads(outputs)
        return self.model_runner.collect_nnsight(
            request_ids, finished_request_ids, outputs
        )

    def nnsight_request_count(self) -> int:
        """How many requests this worker's runner still tracks, via ``collective_rpc``."""
        return self.model_runner.nnsight_request_count()

    def nnsight_register(
        self, registration_id: str, payload: bytes, name: str | None = None
    ) -> None:
        """Install a block this worker runs for every request (``collective_rpc``).

        ``name`` is what requests may address it by (``edits=[...]``).
        """
        return self.model_runner.nnsight_register(registration_id, payload, name=name)

    def nnsight_clear_registered(self, registration_id: str) -> None:
        """Remove a registration from this worker (``collective_rpc``)."""
        return self.model_runner.nnsight_clear_registered(registration_id)
