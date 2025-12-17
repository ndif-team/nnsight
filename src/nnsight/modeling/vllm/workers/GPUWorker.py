from vllm.v1.worker import gpu_worker
from ..model_runners.GPUModelRunner import NNsightGPUModelRunner
from vllm.outputs import RequestOutput

class NNsightGPUWorker(gpu_worker.Worker):

    def __init__(self, *args, **kwargs):
        
        gpu_worker.GPUModelRunner = NNsightGPUModelRunner
        
        super().__init__(*args, **kwargs)

    def finish_nnsight(self, finished_requests: list[RequestOutput]):
        return self.model_runner.finish_nnsight(finished_requests)