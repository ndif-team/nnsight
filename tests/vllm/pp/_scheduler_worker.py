"""One PP=2 engine in a fresh process: which scheduler its engine core built,
and the per-step values of a block that sends the last stage's logits back to
the first stage. Prints one JSON line. Run by test_engine_scheduler.py.

Scenarios:
  default    no scheduling arguments: vLLM turns async scheduling on
  async_off  async_scheduling=False
  ray_auto   built inside a Ray actor in a placement group, with no backend
             named: vLLM picks Ray itself and turns async scheduling off
"""

import json
import os
import sys

STEPS = 4


def build_and_run(extra):
    # The engine core in this process, so its scheduler can be looked at.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import nnsight
    from nnsight.modeling.vllm import VLLM

    model = VLLM("Qwen/Qwen2.5-0.5B", pipeline_parallel_size=2, gpu_memory_utilization=0.12, dispatch=True, **extra)
    engine = model.vllm_entrypoint.llm_engine
    scheduler = type(engine.engine_core.engine_core.scheduler).__name__
    with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, max_tokens=STEPS, ignore_eos=True) as tracer:
        sums = nnsight.save([])
        total = 0.0
        for _ in tracer.iter[:STEPS]:
            # a first-stage layer, written from the last stage's logits of the step before
            model.model.layers[2].output[0][-1] += SCALE * total
            total = float(model.logits.float().sum())
            sums.append(total)
    return {
        "scheduler": scheduler,
        "backend": str(engine.vllm_config.parallel_config.distributed_executor_backend),
        "async_scheduling": bool(engine.vllm_config.scheduler_config.async_scheduling),
        "sums": list(sums),
    }


def ray_auto():
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray.init(num_gpus=2, include_dashboard=False, log_to_driver=False)
    group = placement_group([{"CPU": 1}, {"GPU": 1}, {"GPU": 1}])
    ray.get(group.ready())

    @ray.remote(num_cpus=1)
    class Builder:
        def run(self):
            return build_and_run({})

    builder = Builder.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(group, placement_group_capture_child_tasks=True),
        # The builder holds no GPU of its own; it sees the node's GPUs as
        # this process does, as vLLM's own placement-group setups expect.
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
    ).remote()
    return ray.get(builder.run.remote())


if __name__ == "__main__":
    scenario = sys.argv[1]
    # Scale of the first-stage write; 0 turns it off, for the run that shows it does something.
    SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    # Reads are the model's own buffers, so the in-place write reaches the model.
    # Set before Ray starts, so its workers get the same value.
    os.environ["NNSIGHT_VLLM_CLONE_READS"] = "0"
    if scenario == "ray_auto":
        result = ray_auto()
    else:
        result = build_and_run({"async_scheduling": False} if scenario == "async_off" else {})
    print("RESULT " + json.dumps(result), flush=True)
