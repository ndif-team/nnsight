from vllm.v1.worker import gpu_worker
from ..model_runners.GPUModelRunner import NNsightGPUModelRunner


class NNsightGPUWorker(gpu_worker.Worker):

    def __init__(self, *args, **kwargs):
        
        gpu_worker.GPUModelRunner = NNsightGPUModelRunner
        
        super().__init__(*args, **kwargs)