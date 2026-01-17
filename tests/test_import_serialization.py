"""Test that imported code is properly serialized with remote='local'.

These tests verify that functions and classes from external modules
are correctly captured during serialization for remote execution.

On v0.5.16: These tests FAIL because cloudpickle only stores
            module references, not source code.

On source-serialization: These tests PASS because source code is
                         captured and reconstructed.
"""

import pytest
import torch

from nnsight import LanguageModel
from mymethods import normalize, RunningStats


@pytest.fixture(scope="module")
def gpt2():
    return LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)


def test_imported_function(gpt2):
    """Test that an imported function works with remote='local'."""
    with gpt2.trace("Hello", remote='local'):
        hidden = gpt2.transformer.h[0].output[0]
        normed = normalize(hidden)
        result = normed.save()

    # Verify normalization worked (unit norm along last dim)
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_imported_class(gpt2):
    """Test that an imported class works with remote='local'."""
    stats = RunningStats()

    with gpt2.trace("Hello", remote='local'):
        hidden = gpt2.transformer.h[0].output[0]
        stats.add(hidden)
        mean = stats.mean().save()

    assert mean is not None
    assert mean.shape[-1] == 768  # GPT-2 hidden size
