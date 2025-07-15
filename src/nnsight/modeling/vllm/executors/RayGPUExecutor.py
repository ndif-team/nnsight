from typing import TYPE_CHECKING

from vllm.executor.ray_gpu_executor import RayGPUExecutor

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

class NNsightRayGPUExecutor(RayGPUExecutor):

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):

        self.vllm_config.parallel_config.worker_cls = "nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker"

        super()._init_workers_ray(placement_group, **ray_remote_kwargs)
