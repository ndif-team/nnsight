"""Architectures whose module trees differ across pipeline ranks.

vLLM builds some modules only on one rank (``logits_processor`` behind
``is_last_rank`` in 19 model files). The meta tree the worker builds names
them all, and a request serialized against the client's tree resolves on every
rank because the missing ones are stubbed. GraniteForCausalLM is one such
architecture and is not a Llama subclass. The checkpoint is large, so the test
runs only when it is already in the local HuggingFace cache.
"""

import os

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
