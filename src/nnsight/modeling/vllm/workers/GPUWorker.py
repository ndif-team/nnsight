"""The one place nnsight gets into a vLLM worker process.

vLLM's worker builds its model runner in ``init_device``; once it has, the
runner's class is swapped for nnsight's subclass, which adds behaviour and no
constructor state. `_load` names this class as vLLM's ``worker_cls``, a supported
engine argument, so no part of vLLM's own startup is patched — and a runner
nnsight does not instrument is refused here, in the worker, rather than coming up
silently uninstrumented.

vLLM has two GPU model runners and chooses between them itself, per engine (its
``use_v2_model_runner``, which ``VLLM_USE_V2_MODEL_RUNNER`` overrides). nnsight
has an adapter for each and swaps in the one matching the runner vLLM built.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

from vllm import envs
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from ..model_runners.GPUModelRunner import NNsightGPUModelRunner
from ..model_runners.GPUModelRunnerV2 import NNsightGPUModelRunnerV2

# vLLM's runner class -> the nnsight subclass swapped in for it.
ADAPTERS = {
    GPUModelRunner: NNsightGPUModelRunner,
    GPUModelRunnerV2: NNsightGPUModelRunnerV2,
}


class NNsightGPUWorker(Worker):
    """A vLLM GPU worker whose model runner interleaves interventions."""

    def init_device(self) -> None:
        # vLLM's compiled-graph Ray executor runs the forward on a thread of its
        # own, and a block suspended there cannot be resumed from the thread a
        # collect arrives on, so `tracer.result` and a parked block's error are lost.
        if (
            self.vllm_config.parallel_config.distributed_executor_backend == "ray"
            and not envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND
        ):
            raise NotImplementedError(
                "nnsight does not support vLLM's compiled-graph Ray executor "
                "(VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0): it runs the forward on a "
                "different thread from the calls that collect a request's values. "
                "Unset VLLM_USE_RAY_V2_EXECUTOR_BACKEND to use vLLM's default Ray executor."
            )
        super().init_device()
        runner = self.model_runner
        adapter = ADAPTERS.get(type(runner))
        if adapter is None:
            raise NotImplementedError(
                f"nnsight instruments vLLM's two GPU model runners, but this worker "
                f"built {type(runner).__module__}.{type(runner).__name__} (a runner "
                "from another platform). Drop nnsight and use vLLM directly for "
                "that run."
            )
        runner.__class__ = adapter

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
