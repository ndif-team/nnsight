from vllm.v1.worker import gpu_worker


class NNsightGPUWorker(gpu_worker.Worker):
    """Install the NNsight adapter for vLLM's selected GPU model runner.

    vLLM selects the runner during ``init_device``. Import and replace only
    that implementation while it constructs the runner, then restore the
    upstream class so other workers in this process remain unaffected.
    """

    def init_device(self):
        # The upstream worker resolves this flag from its configuration (or
        # the environment on older releases). Use the same decision rather
        # than inferring the runner from the installed version.
        if self.use_v2_model_runner:
            from ..model_runners.GPUModelRunnerV2 import NNsightGPUModelRunnerV2
            from vllm.v1.worker.gpu import model_runner

            runner_cls = NNsightGPUModelRunnerV2
            runner_cls.validate_config(self.vllm_config)
        else:
            from ..model_runners.GPUModelRunner import NNsightGPUModelRunner
            from vllm.v1.worker import gpu_model_runner as model_runner

            runner_cls = NNsightGPUModelRunner

        # NNsightRayExecutor sets distributed_executor_backend to a class
        # instead of the string "ray". vLLM's init_device skips
        # local_world_size checks for "ray" backends, so normalize the
        # value before calling super().
        backend = self.parallel_config.distributed_executor_backend
        if backend is not None and not isinstance(backend, str):
            from vllm.v1.executor.ray_executor import RayDistributedExecutor

            if issubclass(backend, RayDistributedExecutor):
                self.parallel_config.distributed_executor_backend = "ray"

        upstream_runner_cls = model_runner.GPUModelRunner
        model_runner.GPUModelRunner = runner_cls
        try:
            super().init_device()
        finally:
            model_runner.GPUModelRunner = upstream_runner_cls
            self.parallel_config.distributed_executor_backend = backend

        if not isinstance(self.model_runner, runner_cls):
            raise RuntimeError(
                "vLLM did not initialize the selected NNsight model runner "
                f"({runner_cls.__name__}); created "
                f"{type(self.model_runner).__name__} instead. "
                "This runner configuration is unsupported by NNsight."
            )

    def collect_nnsight(self, req_ids: list[str], finished_req_ids: list[str] | None = None):
        return self.model_runner.collect_nnsight(req_ids, finished_req_ids)
