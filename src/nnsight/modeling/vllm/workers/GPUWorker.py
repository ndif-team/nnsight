from vllm.v1.worker import gpu_worker
from ..model_runners.GPUModelRunner import NNsightGPUModelRunner
from vllm.v1.worker import gpu_model_runner


class NNsightGPUWorker(gpu_worker.Worker):
    """Custom vLLM GPU worker that uses :class:`NNsightGPUModelRunner`.

    Monkey-patches the default ``GPUModelRunner`` class before
    initialization so vLLM creates NNsight-aware model runners
    that can execute intervention code during model forward passes.
    """

    def __init__(self, *args, **kwargs):

        gpu_model_runner.GPUModelRunner = NNsightGPUModelRunner

        super().__init__(*args, **kwargs)

        # Create a full vLLM model on meta device BEFORE distributed
        # init (no PP/TP groups exist yet). This gives us the complete
        # module tree for grafting onto PPMissing envoys later.
        # The _load_meta pattern bootstraps a temporary PP=1 TP=1
        # distributed env, creates the model, then tears it down.
        if self.parallel_config.pipeline_parallel_size > 1:
            self._pp_meta_model = self._create_pp_meta_model()
        else:
            self._pp_meta_model = None

    def _create_pp_meta_model(self):
        """Create a full vLLM model on meta device with PP=1, TP=1.

        Called before distributed init so no PP groups conflict.
        Bootstraps a temporary single-rank distributed env (same
        pattern as VLLM.__init__), creates the model, then tears
        everything down so init_device can set up the real groups.
        """
        import socket
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.engine.arg_utils import EngineArgs
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        # Bootstrap temporary single-rank distributed env
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        engine_args = EngineArgs(
            model=self.model_config.model,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        vllm_config = engine_args.create_engine_config()
        vllm_config.load_config.device = "meta"

        loader = DummyModelLoader(vllm_config.load_config)
        loader.load_weights = lambda *a, **kw: None
        model = loader.load_model(vllm_config, vllm_config.model_config)

        _ROPE_DICT.clear()

        # Tear down so init_device can set up the real groups.
        destroy_model_parallel()
        destroy_distributed_environment()

        return model

    def init_device(self):
        # NNsightRayExecutor sets distributed_executor_backend to a class
        # instead of the string "ray". vLLM's init_device skips
        # local_world_size checks for "ray" backends, so normalize the
        # value before calling super().
        backend = self.parallel_config.distributed_executor_backend
        if backend is not None and not isinstance(backend, str):
            from vllm.v1.executor.ray_executor import RayDistributedExecutor

            if issubclass(backend, RayDistributedExecutor):
                self.parallel_config.distributed_executor_backend = "ray"
        super().init_device()

    def load_model(self):
        # Pass meta model to model runner before loading so it can
        # graft PPMissing envoys during load_model().
        if self._pp_meta_model is not None:
            self.model_runner._pp_meta_model = self._pp_meta_model
            self._pp_meta_model = None  # transferred, don't hold two refs
        super().load_model()

    def collect_nnsight(self, req_ids: list[str], finished_req_ids: list[str] | None = None):
        return self.model_runner.collect_nnsight(req_ids, finished_req_ids)

    def test_pp_buffer_put(self, entries):
        return self.model_runner.test_pp_buffer_put(entries)

    def test_pp_pull(self, source_rank, key, shape, dtype_str, offset):
        return self.model_runner.test_pp_pull(source_rank, key, shape, dtype_str, offset)

    def test_pp_buffer_clear(self):
        return self.model_runner.test_pp_buffer_clear()

    def test_pp_profile_pull(self, num_pulls, shape, dtype_str, direction):
        return self.model_runner.test_pp_profile_pull(num_pulls, shape, dtype_str, direction)
