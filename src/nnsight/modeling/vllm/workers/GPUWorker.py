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
        # Under PP this rank's module tree holds PPMissingLayer stubs for other
        # stages' layers — stubs with no children, so sub-stub paths
        # (``model.layers.5.attn`` on a rank that doesn't own layer 5) would
        # neither resolve at request deserialization nor be reachable in a
        # block. A full meta-device copy of the architecture, built BEFORE the
        # real distributed groups exist, provides the children to graft onto
        # each stub's envoy (see the runner's ``_graft_pp_missing_envoys``).
        if self.parallel_config.pipeline_parallel_size > 1:
            self._pp_meta_model = self._create_pp_meta_model()
        else:
            self._pp_meta_model = None

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

    def load_model(self) -> None:
        # Hand the meta tree to the runner before it builds the envoy tree, so
        # the graft can run inside its load_model.
        if self._pp_meta_model is not None:
            self.model_runner._pp_meta_model = self._pp_meta_model
            self._pp_meta_model = None  # transferred; don't hold two refs
        super().load_model()

    def collect_nnsight(
        self, request_ids: list[str], finished_request_ids: Optional[list[str]] = None
    ) -> Optional[bytes]:
        """Return this worker's saved values, as ``collective_rpc`` reaches it here."""
        return self.model_runner.collect_nnsight(request_ids, finished_request_ids)

    def nnsight_request_count(self) -> int:
        """How many requests this worker's runner still tracks, via ``collective_rpc``."""
        return self.model_runner.nnsight_request_count()
