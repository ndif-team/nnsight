"""Interventions through vLLM's Ray distributed executor.

vLLM can drive its workers over Ray instead of the default multiprocessing
executor. The port carries saves home over Ray's RPC — a different transport, on a
different thread than the one the workers ran on — so these repeat the core reads,
edits, sharding, and generation against ``distributed_executor_backend="ray"`` to
confirm nothing depends on the multiprocessing path.

Skipped unless Ray is installed and the machine has >=2 GPUs.
"""

import pytest
import torch

pytest.importorskip("vllm")

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Ray executor tests need >=2 GPUs"
)


@pytest.fixture(scope="module")
def vllm_qwen_ray():
    """A Qwen sharded across two ranks, driven over Ray."""
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.2,
        dispatch=True,
    )


class TestRayExecutor:
    @torch.no_grad()
    def test_basic_logit(self, vllm_qwen_ray, ET_prompt):
        with vllm_qwen_ray.trace(ET_prompt, temperature=0.0, top_p=1):
            logits = vllm_qwen_ray.logits.save()

        assert vllm_qwen_ray.tokenizer.decode(logits.argmax(dim=-1)) == " Paris"

    @torch.no_grad()
    def test_generation(self, vllm_qwen_ray, MSG_prompt):
        with vllm_qwen_ray.trace(
            MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
        ) as tracer:
            logits = list().save()
            for _ in tracer.iter[:3]:
                logits.append(vllm_qwen_ray.logits)

        assert len(logits) == 3

    @torch.no_grad()
    def test_sharded_edit_lands(self, vllm_qwen_ray, ET_prompt):
        with vllm_qwen_ray.trace(ET_prompt, temperature=0.0, top_p=1):
            clean = vllm_qwen_ray.logits.save()
        with vllm_qwen_ray.trace(ET_prompt, temperature=0.0, top_p=1):
            out = vllm_qwen_ray.model.layers[5].self_attn.qkv_proj.output
            vllm_qwen_ray.model.layers[5].self_attn.qkv_proj.output = (
                torch.zeros_like(out[0]),
                *out[1:],
            )
            edited = vllm_qwen_ray.logits.save()

        # A column-parallel edit reaches the forward over the Ray transport too.
        assert not torch.allclose(edited.float(), clean.float())

    @torch.no_grad()
    def test_batched_requests(self, vllm_qwen_ray, ET_prompt, MSG_prompt):
        with vllm_qwen_ray.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt):
                et = vllm_qwen_ray.logits.save()
            with tracer.invoke(MSG_prompt):
                msg = vllm_qwen_ray.logits.save()

        assert et is not msg
        assert vllm_qwen_ray.tokenizer.decode(et.argmax(dim=-1)) == " Paris"
