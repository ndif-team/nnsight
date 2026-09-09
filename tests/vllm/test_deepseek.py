"""DeepSeek-V2-Lite under tensor and decode-context parallelism.

A second architecture for the sharded path: multi-head latent attention
(``q_proj`` / ``kv_b_proj`` column-parallel, ``o_proj`` row-parallel, a
replicated ``kv_a_proj_with_mqa``) and a fused mixture of experts with shared
experts. Every parallel run is compared to the same trace on one rank.

With ``decode_context_parallel_size > 1`` and ``VLLM_DCP_Q_REPLICATE=1`` vLLM
builds ``q_proj`` as a ``DCPGroupColumnParallelLinear``: sharded across DCP
*groups* and replicated within one, so the plain all-gather carries every shard
``group_size`` times and the fragments layer drops the replicas.

Needs the checkpoint cached; the DCP class needs four GPUs.
"""

import os

import pytest
import torch

pytest.importorskip("vllm")

REPO = "deepseek-ai/DeepSeek-V2-Lite"
GPU_COUNT = torch.cuda.device_count()
LAYER = 3  # a mixture-of-experts layer (first_k_dense_replace=1)
PROMPT = "The Eiffel Tower is located in the city of"


def _engine(**kwargs):
    from nnsight.modeling.vllm import VLLM

    return VLLM(REPO, dispatch=True, max_model_len=512, trust_remote_code=True, **kwargs)


@pytest.fixture(scope="module")
def dsv2_ref():
    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    # The three engines share the first visible card (vLLM always takes the
    # first N devices): 31 GB of weights here, half and a quarter of that for the
    # sharded ones, each with enough left for a 512-token KV cache.
    return _engine(tensor_parallel_size=1, gpu_memory_utilization=0.5)


@pytest.fixture(scope="module")
def dsv2_tp2():
    if GPU_COUNT < 2:
        pytest.skip("tensor-parallel tests need >=2 GPUs")
    return _engine(tensor_parallel_size=2, gpu_memory_utilization=0.3)


@pytest.fixture(scope="module")
def dsv2_dcp():
    if GPU_COUNT < 4:
        pytest.skip("the DCP-group layer needs >=4 GPUs (tp=4, dcp=2)")
    # Read by the worker processes at spawn; it is what makes vLLM build q_proj
    # as the DCP-group layer rather than a plain column-parallel one.
    os.environ["VLLM_DCP_Q_REPLICATE"] = "1"
    return _engine(tensor_parallel_size=4, decode_context_parallel_size=2, gpu_memory_utilization=0.15)


def _dims(model):
    cfg = model.config
    heads = cfg.num_attention_heads
    return {
        "q": heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
        "kv_b": heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
        "o_in": heads * cfg.v_head_dim,
        "hidden": cfg.hidden_size,
    }


def _greedy_tokens(model, n=8):
    with model.trace(PROMPT, temperature=0.0, top_p=1.0, max_tokens=n) as tracer:
        toks = list().save()
        for _ in tracer.iter[:n]:
            toks.append(model.logits.argmax(dim=-1))
    return [t.item() for t in toks]


def _min_row_cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).min().item()


def _reads(model):
    layer = model.model.layers[LAYER]
    # Cloned: vLLM's MLA rotates the q projection in place after the module
    # returns, and a saved alias of the live tensor would see that rotation.
    with model.trace(PROMPT, temperature=0.0, top_p=1):
        q = layer.self_attn.q_proj.output[0].clone().save()
        kv_b = layer.self_attn.kv_b_proj.output[0].clone().save()
        o_in = layer.self_attn.o_proj.input.clone().save()
        moe = layer.mlp.output.clone().save()
        hidden = layer.output[0].clone().save()
        logits = model.logits.clone().save()
    return {"q": q, "kv_b": kv_b, "o_in": o_in, "moe": moe, "hidden": hidden, "logits": logits}


def _check_whole(model, ref):
    dims = _dims(model)
    mine = _reads(model)
    assert mine["q"].shape[-1] == dims["q"]
    assert mine["kv_b"].shape[-1] == dims["kv_b"]
    assert mine["o_in"].shape[-1] == dims["o_in"]
    assert mine["moe"].shape[-1] == dims["hidden"]
    for name in ("q", "kv_b", "o_in", "moe", "hidden"):
        assert _min_row_cosine(mine[name], ref[name]) > 0.98, name
    assert torch.equal(mine["logits"].argmax(-1), ref["logits"].argmax(-1))


@torch.no_grad()
def test_reference_shapes(dsv2_ref):
    dims = _dims(dsv2_ref)
    got = _reads(dsv2_ref)
    assert got["q"].shape[-1] == dims["q"]
    assert got["o_in"].shape[-1] == dims["o_in"]


@torch.no_grad()
def test_every_sharded_value_is_whole_at_tp2(dsv2_ref, dsv2_tp2):
    _check_whole(dsv2_tp2, _reads(dsv2_ref))


@torch.no_grad()
def test_zeroing_the_row_parallel_input_lands_at_tp2(dsv2_ref, dsv2_tp2):
    def zeroed(model):
        layer = model.model.layers[LAYER]
        with model.trace(PROMPT, temperature=0.0, top_p=1):
            layer.self_attn.o_proj.input[:] = 0
            logits = model.logits.save()
        return logits

    plain = _reads(dsv2_tp2)["logits"]
    edited, edited_ref = zeroed(dsv2_tp2), zeroed(dsv2_ref)
    assert not torch.allclose(edited.float(), plain.float())
    assert _min_row_cosine(edited, edited_ref) > 0.98


@torch.no_grad()
def test_generation_is_identical_at_tp2(dsv2_ref, dsv2_tp2):
    assert _greedy_tokens(dsv2_tp2) == _greedy_tokens(dsv2_ref)


@torch.no_grad()
def test_no_worker_is_left_on_any_rank(dsv2_tp2):
    _reads(dsv2_tp2)
    counts = dsv2_tp2.vllm_entrypoint.llm_engine.collective_rpc("nnsight_request_count")
    assert counts == [0] * len(counts)


class TestDecodeContextParallel:
    @torch.no_grad()
    def test_q_proj_is_the_dcp_group_layer(self, dsv2_dcp):
        # The point of the fixture: without the DCP-group class this exercises
        # nothing the plain TP tests do not. Asked of the workers' real model —
        # the client holds a one-rank meta tree, where q_proj is the plain layer.
        def worker_q_proj_class(worker):
            model = worker.model_runner.nnsight_model
            return type(model.model.layers[LAYER].self_attn.q_proj._module).__name__

        names = dsv2_dcp.vllm_entrypoint.llm_engine.collective_rpc(worker_q_proj_class)
        assert names == ["DCPGroupColumnParallelLinear"] * 4, names

    @torch.no_grad()
    def test_every_sharded_value_is_whole(self, dsv2_ref, dsv2_dcp):
        _check_whole(dsv2_dcp, _reads(dsv2_ref))

    @torch.no_grad()
    def test_generation_is_identical(self, dsv2_ref, dsv2_dcp):
        assert _greedy_tokens(dsv2_dcp) == _greedy_tokens(dsv2_ref)
