"""Fixtures for the pipeline-parallel suite; see ``_support`` for the tiers."""

import os

import pytest
import torch

from _support import MODEL, free_gpus


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs GPUs and boots a real vLLM engine")


@pytest.fixture(scope="session")
def pp2_engine():
    """One PP=2 engine over Qwen2.5-0.5B, shared by the behavior tests.

    Honors an explicit ``CUDA_VISIBLE_DEVICES``; otherwise takes the two freest
    GPUs. The engine's processes are spawned, so the choice made here reaches
    them through the environment.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = free_gpus()
        if len(gpus) < 2:
            pytest.skip(f"PP=2 needs 2 free GPUs, found {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:2])
    from nnsight.modeling.vllm import VLLM

    return VLLM(MODEL, pipeline_parallel_size=2, gpu_memory_utilization=0.12, dispatch=True)
