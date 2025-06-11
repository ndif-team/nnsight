from vllm.worker.worker import Worker

from ..model_runners.GPUModelRunner import NNsightGPUModelRunner


class NNsightGPUWorker(Worker):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, model_runner_cls=NNsightGPUModelRunner, **kwargs)