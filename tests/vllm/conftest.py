"""Fixtures shared by the vLLM integration tests.

The whole directory is skipped unless vLLM imports, so an environment without it
collects nothing here. Tensor-parallel fixtures parametrize over the parallel
sizes the machine has GPUs for; on a single-GPU box the parallel sizes drop out
and those tests skip rather than fail.

Two checkpoints are used: gpt2 for the single-rank tracing and request-tracking
tests, and Qwen2.5-0.5B for tensor parallelism, since it shards cleanly across
ranks where gpt2's fused layers do not.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# The request-cleanup tests read worker-side state by passing a function to
# vLLM's collective_rpc, which its default serializer rejects.
# Only for the tests that ship a *function* to the workers through
# collective_rpc (request-state probes): an ordinary trace, its saves and its
# result need no such flag — test_requests.py proves that in a clean subprocess.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import pytest
import torch

ET_PROMPT = "The Eiffel Tower is located in the city of"
MSG_PROMPT = "Madison Square Garden is located in the city of"

GPU_COUNT = torch.cuda.device_count()
# Qwen2.5-0.5B has 2 key/value heads, so it shards evenly only up to 2 ranks.
MODEL_MAX_TP = 2
# Parallel sizes the current machine can run and the model shards cleanly into.
TP_SIZES = [size for size in (2, 4, 8) if size <= min(GPU_COUNT, MODEL_MAX_TP)]


@pytest.fixture
def ET_prompt() -> str:
    return ET_PROMPT


@pytest.fixture
def MSG_prompt() -> str:
    return MSG_PROMPT


@pytest.fixture(scope="module")
def vllm_gpt2():
    """A dispatched single-rank gpt2 for tracing and request-tracking tests."""
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return VLLM(
        "gpt2", tensor_parallel_size=1, gpu_memory_utilization=0.1, dispatch=True
    )


@pytest.fixture(scope="module")
def vllm_gpt2_uncached():
    """A gpt2 with prefix caching off, which a registration needs.

    A prefix-cached token is served without a forward pass, so no hook fires for
    it and a registered block sees fewer rows than the prompt has — silently. A
    trace asks for its own request to be recomputed; a registration rides requests
    it did not create and cannot, so the cache has to be off at the engine.
    """
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return VLLM(
        "gpt2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        dispatch=True,
    )


@pytest.fixture(scope="module")
def vllm_qwen_ref():
    """A single-rank Qwen, the unsharded reference the parallel runs compare to."""
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        dispatch=True,
    )


def _tp_params():
    if TP_SIZES:
        return TP_SIZES
    return [
        pytest.param(
            2, marks=pytest.mark.skip(reason="tensor-parallel tests need >=2 GPUs")
        )
    ]


@pytest.fixture(scope="module", params=_tp_params())
def vllm_qwen_tp(request):
    """A Qwen sharded across ``request.param`` ranks."""
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=request.param,
        gpu_memory_utilization=0.2,
        dispatch=True,
    )
