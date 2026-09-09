"""Streaming traces on an async vLLM engine driven over Ray.

The two hard paths combined: the async output-handler loop (``mode="async"``) and
vLLM's Ray distributed executor (``distributed_executor_backend="ray"``), sharded
across two ranks. Saves ride home over Ray's RPC on the finished streamed output.

Skipped unless Ray is installed and the machine has >=2 GPUs.
"""

import asyncio

import pytest
import torch

pytest.importorskip("vllm")
ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Async Ray executor tests need >=2 GPUs"
)


@pytest.fixture(scope="module")
def async_ray_loop():
    # AsyncLLM binds its background tasks to the loop that created it, so this model
    # needs its own loop, distinct from the plain async model's.
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def vllm_qwen_async_ray(async_ray_loop):
    import os

    import psutil

    from nnsight.modeling.vllm import VLLM

    before = {p.pid for p in psutil.Process(os.getpid()).children(recursive=True)}
    model = VLLM(
        "Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.2,
        dispatch=True,
        mode="async",
    )
    model._child_pids = {
        p.pid for p in psutil.Process(os.getpid()).children(recursive=True)
    } - before
    yield model
    _shutdown_async_engine(model, async_ray_loop)


def _shutdown_async_engine(model, loop):
    """Wind the async engine down on its own loop, then reap only its own workers."""
    import psutil

    async def _graceful():
        try:
            model.vllm_entrypoint.shutdown()
        except Exception:
            pass
        await asyncio.sleep(1.0)

    try:
        loop.run_until_complete(_graceful())
    except Exception:
        pass

    survivors = [
        psutil.Process(pid)
        for pid in getattr(model, "_child_pids", ())
        if psutil.pid_exists(pid)
    ]
    for proc in survivors:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(survivors, timeout=5)


class TestAsyncRayExecutor:
    def test_streams_multiple_outputs(
        self, vllm_qwen_async_ray, async_ray_loop, ET_prompt
    ):
        async def run():
            with vllm_qwen_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                vllm_qwen_async_ray.logits.save()

            count = 0
            async for _ in tracer.backend:
                count += 1
            assert count > 1

        async_ray_loop.run_until_complete(run())

    def test_saves_on_finished_output(
        self, vllm_qwen_async_ray, async_ray_loop, ET_prompt
    ):
        async def run():
            with vllm_qwen_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=5
            ) as tracer:
                logits = vllm_qwen_async_ray.logits.save()

            last = None
            async for output in tracer.backend:
                last = output

            assert last.finished
            assert "logits" in last.saves
            token = vllm_qwen_async_ray.tokenizer.decode(
                last.saves["logits"].argmax(dim=-1)
            )
            assert token == " Paris"

        async_ray_loop.run_until_complete(run())

    def test_intervention_streams(
        self, vllm_qwen_async_ray, async_ray_loop, ET_prompt
    ):
        async def run():
            with vllm_qwen_async_ray.trace(
                ET_prompt, temperature=0.0, max_tokens=1
            ) as tracer:
                vllm_qwen_async_ray.model.layers[-2].mlp.output = torch.zeros_like(
                    vllm_qwen_async_ray.model.layers[-2].mlp.output
                )
                logits = vllm_qwen_async_ray.logits.save()

            last = None
            async for output in tracer.backend:
                last = output

            # Zeroing a late MLP over the async Ray path moves the prediction off
            # the clean answer.
            token = vllm_qwen_async_ray.tokenizer.decode(
                last.saves["logits"].argmax(dim=-1)
            )
            assert token != " Paris"

        async_ray_loop.run_until_complete(run())
