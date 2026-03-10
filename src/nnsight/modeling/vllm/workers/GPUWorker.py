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

    def collect_nnsight(self, req_ids: list[str], finished_req_ids: list[str] | None = None):
        return self.model_runner.collect_nnsight(req_ids, finished_req_ids)
