"""Interventions on a request that carries a LoRA adapter.

vLLM applies an adapter by swapping punica kernels in around the base layers, so
a traced module's ``.output`` is the *adapted* value and an edit lands on top of
it. Nothing in nnsight knows about LoRA — ``lora_request`` rides through
``trace``/``invoke`` to ``SamplingParams``'s sibling argument — which is exactly
what these confirm: the same reads and writes work with an adapter attached, and
the adapter is really in the forward.

The adapter is written here rather than downloaded: PEFT's on-disk format is an
``adapter_config.json`` and a safetensors file of ``lora_A``/``lora_B`` matrices,
so a deterministic one can be built for whatever base model the box has cached.
That also means the test knows the adapter is non-trivial — a randomly
initialized ``lora_B`` is zero in real checkpoints, which would change nothing.
"""

import json

import pytest
import torch

pytest.importorskip("vllm")

BASE = "meta-llama/Llama-3.2-1B"
# Large enough that the adapted logits are unmistakably different, small enough
# that the model still produces ordinary text.
SCALE = 0.02
RANK = 8


@pytest.fixture(scope="module")
def lora_path(tmp_path_factory):
    """A deterministic rank-8 LoRA on every layer's q_proj and v_proj."""
    from safetensors.torch import save_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(BASE)
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // heads)
    sizes = {
        "q_proj": heads * head_dim,
        "v_proj": kv_heads * head_dim,
    }

    generator = torch.Generator().manual_seed(0)
    weights = {}
    for layer in range(config.num_hidden_layers):
        for name, out_features in sizes.items():
            stem = f"base_model.model.model.layers.{layer}.self_attn.{name}"
            weights[f"{stem}.lora_A.weight"] = torch.randn(
                RANK, config.hidden_size, generator=generator
            ) * SCALE
            # Not zeros: a freshly initialized adapter is a no-op, and a no-op
            # adapter would pass a test that never applied it.
            weights[f"{stem}.lora_B.weight"] = torch.randn(
                out_features, RANK, generator=generator
            ) * SCALE

    directory = tmp_path_factory.mktemp("lora")
    save_file(weights, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": BASE,
                "r": RANK,
                "lora_alpha": 2 * RANK,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": list(sizes),
            }
        )
    )
    return str(directory)


@pytest.fixture(scope="module")
def vllm_lora():
    if torch.cuda.device_count() < 1:
        pytest.skip("LoRA tests need a GPU")
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        BASE,
        enable_lora=True,
        max_lora_rank=RANK,
        gpu_memory_utilization=0.25,
        dispatch=True,
    )


def request(path, id=1):
    from vllm.lora.request import LoRARequest

    return LoRARequest("nnsight-test", id, path)


class TestLoRA:
    @torch.no_grad()
    def test_the_adapter_is_in_the_forward(self, vllm_lora, lora_path, ET_prompt):
        with vllm_lora.trace(ET_prompt, temperature=0.0, top_p=1):
            base = vllm_lora.logits.save()
        with vllm_lora.trace(
            ET_prompt, temperature=0.0, top_p=1, lora_request=request(lora_path)
        ):
            adapted = vllm_lora.logits.save()

        assert not torch.allclose(base.float(), adapted.float())

    @torch.no_grad()
    def test_a_read_sees_the_adapted_value(self, vllm_lora, lora_path, ET_prompt):
        # vLLM fuses q/k/v into one `qkv_proj`, and the adapter covers the q and
        # v slices of it, so its output must differ; a read that saw only the base
        # projection would come back identical.
        with vllm_lora.trace(ET_prompt, temperature=0.0, top_p=1):
            base = vllm_lora.model.layers[5].self_attn.qkv_proj.output[0].save()
        with vllm_lora.trace(
            ET_prompt, temperature=0.0, top_p=1, lora_request=request(lora_path)
        ):
            adapted = vllm_lora.model.layers[5].self_attn.qkv_proj.output[0].save()

        assert base.shape == adapted.shape
        assert not torch.allclose(base.float(), adapted.float())

    @torch.no_grad()
    def test_an_edit_lands_on_an_adapted_request(self, vllm_lora, lora_path, ET_prompt):
        with vllm_lora.trace(
            ET_prompt, temperature=0.0, top_p=1, lora_request=request(lora_path)
        ):
            clean = vllm_lora.logits.save()
        with vllm_lora.trace(
            ET_prompt, temperature=0.0, top_p=1, lora_request=request(lora_path)
        ):
            vllm_lora.model.layers[10].output[0][:] = 0
            edited = vllm_lora.logits.save()

        assert not torch.allclose(clean.float(), edited.float())

    @torch.no_grad()
    def test_adapted_and_base_requests_batch_together(
        self, vllm_lora, lora_path, ET_prompt
    ):
        # Continuous batching mixes them in one forward; each invoke's read must
        # be its own request's, not whichever adapter the batch happened to hold.
        with vllm_lora.trace(temperature=0.0, top_p=1) as tracer:
            with tracer.invoke(ET_prompt, lora_request=request(lora_path)):
                adapted = vllm_lora.logits.save()
            with tracer.invoke(ET_prompt):
                base = vllm_lora.logits.save()

        assert not torch.allclose(base.float(), adapted.float())

    @torch.no_grad()
    def test_generation_runs_with_an_adapter(self, vllm_lora, lora_path, MSG_prompt):
        with vllm_lora.trace(
            MSG_prompt,
            temperature=0.0,
            top_p=1,
            max_tokens=3,
            lora_request=request(lora_path),
        ) as tracer:
            result = tracer.result.save()

        assert len(result.outputs[0].token_ids) == 3
