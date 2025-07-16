from vllm.v1.worker import gpu_worker, gpu_model_runner
from .gpu_input_batch import NNsightInputBatch
from ..model_runners.GPUModelRunner import NNsightGPUModelRunner


class NNsightGPUWorker(gpu_worker.Worker):

    def __init__(self, *args, **kwargs):
        
        gpu_worker.GPUModelRunner = NNsightGPUModelRunner
        # gpu_model_runner.InputBatch = NNsightInputBatch
        
        super().__init__(*args, **kwargs)