"""Architectures whose module trees differ across pipeline ranks.

vLLM builds some modules only on one rank (``logits_processor`` behind
``is_last_rank`` in 19 model files). The meta tree the worker builds names
them all, and a request serialized against the client's tree resolves on every
rank because the missing ones are stubbed. GraniteForCausalLM is one such
architecture and is not a Llama subclass. The checkpoint is large, so the test
runs only when it is already in the local HuggingFace cache.

GPT-2 shares one activation module across every block (vLLM's activation
registry hands out one instance per name), so the tree wraps it once and every
other block's ``act`` is an alias; each rank's persistent-id table must resolve
every alias path, whichever block the rank holds first.
"""

import os

import nnsight
import pytest
import torch

from _support import free_gpus

pytestmark = pytest.mark.gpu

GRANITE = "ibm-granite/granite-3.3-2b-instruct"


def _cached(repo_id: str) -> bool:
    from huggingface_hub.constants import HF_HUB_CACHE

    return os.path.isdir(os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--")))


@pytest.fixture(scope="module")
def granite_pp2():
    if not _cached(GRANITE):
        pytest.skip(f"{GRANITE} is not in the local HuggingFace cache")
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = free_gpus(min_free_mib=24000)
        if len(gpus) < 2:
            pytest.skip(f"needs 2 GPUs with 24 GB free, found {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:2])
    from nnsight.modeling.vllm import VLLM

    return VLLM(GRANITE, pipeline_parallel_size=2, gpu_memory_utilization=0.25, dispatch=True)


def test_rank_gated_architecture_traces_under_pp(granite_pp2):
    model = granite_pp2
    with model.trace("The capital of France is", temperature=0.0, max_tokens=1):
        logits = model.logits.save()
    assert isinstance(logits, torch.Tensor) and logits.shape[-1] > 1000
    assert torch.isfinite(logits.float()).all()


@pytest.fixture(scope="module")
def gpt2_pp2():
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = free_gpus()
        if len(gpus) < 2:
            pytest.skip(f"PP=2 needs 2 free GPUs, found {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:2])
    from nnsight.modeling.vllm import VLLM

    return VLLM("gpt2", pipeline_parallel_size=2, gpu_memory_utilization=0.15, dispatch=True)


def test_shared_activation_module_resolves_on_every_rank(gpt2_pp2):
    model = gpt2_pp2
    prompt = "The Eiffel Tower is located in the city of"
    with model.trace(prompt, temperature=0.0, max_tokens=1):
        early = model.transformer.h[1].output.save()
        late = model.transformer.h[10].output.save()
        logits = model.logits.save()
    assert early.shape == late.shape and early.shape[-1] == 768
    assert model.tokenizer.decode(logits[-1].argmax(-1)).strip() == "Paris"
    # The tied head's weight reads through its alias on both ranks.
    with model.trace(prompt, temperature=0.0, max_tokens=1):
        total = model.lm_head.weight.float().abs().sum().save()
    assert torch.isfinite(total) and total > 0


SMOLLM = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="module")
def smollm_pp2():
    """Llama-shaped, embeddings tied by weight: the head is its own module,
    held on the last stage only, so stage 0 sees it as a shell."""
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = free_gpus()
        if len(gpus) < 2:
            pytest.skip(f"PP=2 needs 2 free GPUs, found {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:2])
    from nnsight.modeling.vllm import VLLM

    return VLLM(SMOLLM, pipeline_parallel_size=2, gpu_memory_utilization=0.15, dispatch=True)


def test_head_weight_reads_across_stages_and_unembeds(smollm_pp2):
    model = smollm_pp2
    prompt = "The Eiffel Tower is located in the city of"
    with model.trace(prompt, temperature=0.0, max_tokens=1):
        head = model.lm_head.param("weight")
        tied = model.model.embed_tokens.param("weight")
        same = torch.equal(head, tied)
        # The norm returns (hidden, residual); indexing the element works for
        # the real tuple and for its cross-stage stand-in alike.
        hidden = model.model.norm.output[0]
        lens = (hidden[-1].float() @ head.float().T).argmax(-1).save()
        sampled = model.logits.save()
        flag = torch.tensor(same).save()
    assert bool(flag), "the tied head must equal the embedding table"
    assert int(lens) == int(sampled[-1].argmax(-1))


def test_head_weight_row_reads_every_decode_step(smollm_pp2):
    """The generation-steering pattern: one row of the head per step."""
    model = smollm_pp2
    with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, max_tokens=3) as tracer:
        rows = nnsight.save([])
        for _ in tracer.iter[:3]:
            rows.append(float(model.lm_head.param("weight")[42].float().sum()))
    assert len(rows) == 3 and rows[0] == rows[1] == rows[2]
