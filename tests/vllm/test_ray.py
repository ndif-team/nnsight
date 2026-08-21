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


@pytest.fixture(scope="module")
def vllm_qwen_ray_uncached():
    """The same, with prefix caching off — what an installed edit needs."""
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        enable_prefix_caching=False,
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


class TestRayEdits:
    """`model.edit()` over Ray — install by RPC, values home by another.

    Worth its own coverage: the install is a ``collective_rpc`` to Ray actors
    rather than to multiprocessing workers, and the collect that carries a
    registered value home runs on a thread that never ran the block — the
    thread-local bookkeeping `collect_nnsight` does is a no-op there, so the
    values have to survive without it.
    """

    @torch.no_grad()
    def test_untraced_requests_carry_their_own_values(self, vllm_qwen_ray_uncached):
        model = vllm_qwen_ray_uncached
        prompts = ["The Eiffel Tower is in", "Hello world", "A"]

        with model.edit() as (tracer, edit):
            hidden = model.model.layers[10].output[0].save()
        try:
            outputs = model.generate(prompts, max_tokens=3, temperature=0.0,
                                     ignore_eos=True)

            for output in outputs:
                assert "hidden" in output.saves
                assert output.saves["hidden"].shape[0] == len(
                    output.prompt_token_ids
                )
            assert len({id(o.saves["hidden"]) for o in outputs}) == len(outputs)
        finally:
            edit.clear()

    @torch.no_grad()
    def test_a_sharded_read_is_gathered(self, vllm_qwen_ray_uncached, ET_prompt):
        # A column-parallel output read from an installed block comes back whole,
        # measured against the same read from a trace — one rank's shard would be
        # the right dtype and half the width.
        model = vllm_qwen_ray_uncached

        with model.trace(ET_prompt, temperature=0.0, top_p=1):
            traced = model.model.layers[5].self_attn.qkv_proj.output[0].save()

        with model.edit() as (tracer, edit):
            qkv = model.model.layers[5].self_attn.qkv_proj.output[0].save()
        try:
            output = model.generate([ET_prompt], max_tokens=1, temperature=0.0,
                                    ignore_eos=True)[0]

            assert output.saves["qkv"].shape == traced.shape
            # Every rank gathers and reports the same whole value; the earliest
            # rank's is the one kept, so it lands where a traced value would
            # rather than on whichever rank answered last.
            assert output.saves["qkv"].device == traced.device
            assert torch.allclose(output.saves["qkv"].float(), traced.float())
        finally:
            edit.clear()

    @torch.no_grad()
    def test_clear_stops_it(self, vllm_qwen_ray_uncached, ET_prompt):
        model = vllm_qwen_ray_uncached

        with model.edit() as (tracer, edit):
            hidden = model.model.layers[10].output[0].save()
        model.clear_edits()

        output = model.generate([ET_prompt], max_tokens=2, temperature=0.0,
                                ignore_eos=True)[0]
        assert "hidden" not in getattr(output, "saves", {})
