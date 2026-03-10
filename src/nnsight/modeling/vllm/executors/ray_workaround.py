"""Custom Ray executor for NNsight that works around a vLLM + Ray actor crash
and supports connecting to remote Ray clusters.

vLLM v0.15.1 + Ray 2.53.0 have a compatibility issue where Ray actor processes
crash during module-level imports of heavy vllm submodules (particularly
``vllm.multimodal``) during the actor construction phase. The crash occurs at
the C level (in grpcio's ``cygrpc`` extension) with no Python traceback.

The root cause: when Ray spawns an actor process and imports the module
containing ``RayWorkerWrapper``, the transitive module-level imports
(``worker_base.py`` → ``vllm.multimodal``, etc.) conflict with Ray's internal
gRPC event engine, causing the process to die before the actor is fully
constructed.

The fix: ``LazyRayWorkerWrapper`` is a thin wrapper class with no heavy
module-level imports. It defers all vllm imports to ``__init__`` time, which
runs after the actor process is fully constructed and Ray's gRPC connection is
stable. All methods are explicitly defined to satisfy Ray's remote method
resolution (``__getattr__`` delegation does not work with Ray actor handles).

``NNsightRayExecutor`` also replaces vLLM's ``initialize_ray_cluster`` with
logic that works when the driver is not on a cluster node (e.g. connecting to
a remote Ray cluster via ``RAY_ADDRESS``).  Upstream vLLM assumes the driver
runs on a node with GPUs and uses ``ray._private.state`` APIs that are
incompatible with remote drivers.
"""

from vllm.v1.executor.ray_executor import RayDistributedExecutor


class LazyRayWorkerWrapper:
    """Drop-in replacement for ``vllm.v1.executor.ray_utils.RayWorkerWrapper``.

    Defers heavy vllm imports to ``__init__`` (actor method execution time)
    rather than module import time (actor construction time).
    """

    def __init__(self, *args, **kwargs):
        from vllm.v1.executor.ray_utils import RayWorkerWrapper

        self._w = RayWorkerWrapper(*args, **kwargs)

    # --- WorkerWrapperBase methods ---

    def update_environment_variables(self, envs_list):
        return self._w.update_environment_variables(envs_list)

    def init_worker(self, all_kwargs):
        return self._w.init_worker(all_kwargs)

    def adjust_rank(self, rank_mapping):
        return self._w.adjust_rank(rank_mapping)

    def execute_method(self, method, *args, **kwargs):
        return self._w.execute_method(method, *args, **kwargs)

    def shutdown(self):
        return self._w.shutdown()

    # --- RayWorkerWrapper methods ---

    def get_node_ip(self):
        return self._w.get_node_ip()

    def get_node_and_gpu_ids(self):
        return self._w.get_node_and_gpu_ids()

    def setup_device_if_necessary(self):
        return self._w.setup_device_if_necessary()

    def execute_model_ray(self, execute_model_input):
        return self._w.execute_model_ray(execute_model_input)


class NNsightRayExecutor(RayDistributedExecutor):
    """Ray executor that uses ``LazyRayWorkerWrapper`` to avoid actor crashes
    and supports connecting to pre-existing Ray clusters.

    Pass this class as ``distributed_executor_backend`` instead of ``"ray"``::

        LLM("gpt2", distributed_executor_backend=NNsightRayExecutor)

    If ``RAY_ADDRESS`` is set, connects to that cluster.  Otherwise tries to
    find a local cluster and, failing that, starts a new one.
    """

    def _init_executor(self) -> None:
        import os
        import ray
        import vllm.v1.executor.ray_utils as ray_utils
        import vllm.v1.executor.ray_executor as ray_exec
        from vllm.v1.executor.ray_utils import _wait_until_pg_ready
        from vllm.platforms import current_platform

        # --- Swap in lazy wrapper ---
        ray_utils.RayWorkerWrapper = LazyRayWorkerWrapper
        ray_exec.RayWorkerWrapper = LazyRayWorkerWrapper

        self.forward_dag = None

        # --- Initialize Ray ---
        # vLLM's compiled DAGs require a full Ray runtime (not Ray Client).
        # If RAY_ADDRESS is set and there's no local Ray, join the remote
        # cluster as a driver-only node (0 GPUs, 0 CPUs) so we get full
        # runtime access.  Otherwise try a local cluster or start one.
        if not ray.is_initialized():
            import subprocess

            ray_address = os.environ.get("RAY_ADDRESS")
            if ray_address and ray_address.startswith("ray://"):
                raise ValueError(
                    "RAY_ADDRESS must be a GCS address (host:port), not a "
                    f"Ray Client address (ray://...). Got: {ray_address}"
                )

            try:
                ray.init(address="auto")
            except (ConnectionError, ValueError, RuntimeError):
                if ray_address:
                    subprocess.run(
                        [
                            "ray", "start",
                            f"--address={ray_address}",
                            "--num-gpus=0",
                            "--num-cpus=0",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    ray.init(address="auto")
                else:
                    ray.init()

        # Disable Ray usage stats collection.
        if os.environ.get("RAY_USAGE_STATS_ENABLED", "0") != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # --- Create placement group ---
        # Unlike upstream vLLM's initialize_ray_cluster, we do NOT assume the
        # driver is on a cluster node with GPUs.  This allows the driver to be
        # a remote process (e.g. connected via RAY_ADDRESS=host:6379).
        parallel_config = self.parallel_config
        device_str = current_platform.ray_device_key
        if not device_str:
            raise ValueError(
                f"Platform {current_platform.device_name} does not support Ray."
            )

        if parallel_config.placement_group:
            current_placement_group = parallel_config.placement_group
        else:
            current_placement_group = ray.util.get_current_placement_group()

        if not current_placement_group:
            placement_group_specs = [
                {device_str: 1.0}
                for _ in range(parallel_config.world_size)
            ]
            current_placement_group = ray.util.placement_group(
                placement_group_specs, strategy="PACK"
            )
            _wait_until_pg_ready(current_placement_group)

        parallel_config.placement_group = current_placement_group

        # --- Tell vLLM the driver is on the head node ---
        # When the driver process is remote (e.g. connected via RAY_ADDRESS),
        # vLLM's get_ip() returns the client's IP which isn't a cluster node.
        # _init_workers_ray validates that #IPs == #nodes and uses driver_ip
        # for worker sorting and distributed init.  Setting VLLM_HOST_IP to
        # the head node's IP makes the driver appear co-located with it.
        if not os.environ.get("VLLM_HOST_IP"):
            nodes = ray.nodes()
            for n in nodes:
                if n.get("Alive") and n.get("Resources", {}).get(
                    "node:__internal_head__"
                ):
                    os.environ["VLLM_HOST_IP"] = n["NodeManagerAddress"]
                    break

        # --- Create workers ---
        self._init_workers_ray(current_placement_group)

        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.uses_sampler = (
            self.vllm_config.model_config.runner_type != "pooling"
            and (
                self.vllm_config.ec_transfer_config is None
                or not self.vllm_config.ec_transfer_config.is_ec_producer
            )
        )
        self.scheduler_output = None
