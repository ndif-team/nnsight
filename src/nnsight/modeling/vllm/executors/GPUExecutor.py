from vllm.executor.gpu_executor import GPUExecutor, create_worker
from typing import Optional


class NNsightGPUExecutor(GPUExecutor):

    def _create_worker(self,
                       local_rank: int=0,
                       rank: int=0,
                       distributed_init_method: Optional[str]=None):

        kwargs = self._get_worker_kwargs(
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method
        )

        kwargs["vllm_config"].parallel_config.worker_cls = "nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"

        return create_worker(**kwargs)