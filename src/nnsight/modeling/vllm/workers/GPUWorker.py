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

#: The full meta-device model a pipeline-parallel worker builds before its real
#: distributed groups exist, for the runner to take in ``load_model``. One
#: worker per process, so one slot.
PP_META_MODEL: Any = None


class NNsightGPUWorker(Worker):
    """A vLLM GPU worker whose model runner interleaves interventions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Under PP this rank's module tree holds PPMissingLayer stubs for other
        # stages' layers, and stubs have no children. A full meta-device copy of
        # the architecture, built before the real distributed groups exist,
        # provides both the modules the architecture builds only on some ranks
        # and the children to graft onto each stub's envoy (see the runner's
        # load_model).
        global PP_META_MODEL
        if self.parallel_config.pipeline_parallel_size > 1:
            PP_META_MODEL = self._create_pp_meta_model()

    def _create_pp_meta_model(self) -> Any:
        """Build the full vLLM model on the meta device with PP=1, TP=1.

        Bootstraps a temporary single-rank distributed env (no real groups
        exist yet), constructs the model without weights, then tears the env
        down so ``init_device`` can set up the real groups.
        """
        import copy
        import socket

        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, backend="gloo")
        initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        # The engine's own config, narrowed to one rank and the meta device:
        # every loading option the user gave (trust_remote_code, revision,
        # hf_overrides, quantization, ...) applies to the meta build exactly
        # as it applied to the real one.
        vllm_config = copy.deepcopy(self.vllm_config)
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.world_size = 1
        vllm_config.load_config.device = "meta"

        loader = DummyModelLoader(vllm_config.load_config)
        loader.load_weights = lambda *a, **kw: None
        model = loader.load_model(vllm_config, vllm_config.model_config)

        # The rope cache keyed under the bootstrap env must not leak into the
        # real one.
        _ROPE_DICT.clear()

        destroy_model_parallel()
        destroy_distributed_environment()

        return model

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
