from vllm.executor.ray_gpu_executor import RayGPUExecutor

class NNsightRayGPUExecutor(RayGPUExecutor):

    def _get_worker_module_and_class(self):
        return ("nnsight.modeling.vllm.workers.GPUWorker", "NNsightGPUWorker", None)