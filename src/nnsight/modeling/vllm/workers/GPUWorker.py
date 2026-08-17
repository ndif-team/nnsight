"""The one place nnsight gets into a vLLM worker process.

vLLM builds its model runner by looking the class up on the module at construction
time, so rebinding that name before ``Worker.__init__`` resolves it is what puts an
nnsight runner in the worker at all. Everything else in this package follows from
the runner installed here; `_load` names this
class as vLLM's ``worker_cls``, which is a supported engine argument, so no part of
vLLM's own startup is patched.
"""

from __future__ import annotations

from typing import Any, Optional

from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_worker import Worker

from ..model_runners.GPUModelRunner import NNsightGPUModelRunner


class NNsightGPUWorker(Worker):
    """A vLLM GPU worker whose model runner interleaves interventions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Rebind before super(), which resolves the runner class off this module.
        gpu_model_runner.GPUModelRunner = NNsightGPUModelRunner
        super().__init__(*args, **kwargs)

    def collect_nnsight(
        self,
        request_ids: list[str],
        finished_request_ids: Optional[list[str]] = None,
        outputs: Optional[dict] = None,
    ) -> Optional[bytes]:
        """Return this worker's saved values, as ``collective_rpc`` reaches it here."""
        return self.model_runner.collect_nnsight(
            request_ids, finished_request_ids, outputs
        )

    def nnsight_request_count(self) -> int:
        """How many requests this worker's runner still tracks, via ``collective_rpc``."""
        return self.model_runner.nnsight_request_count()

    def nnsight_register(self, registration_id: str, payload: bytes) -> None:
        """Install a block this worker runs for every request (``collective_rpc``)."""
        return self.model_runner.nnsight_register(registration_id, payload)

    def nnsight_clear_registered(self, registration_id: str) -> None:
        """Remove a registration from this worker (``collective_rpc``)."""
        return self.model_runner.nnsight_clear_registered(registration_id)
