"""Basic test for remote='local' simulation."""

import pytest
import torch
from nnsight import LanguageModel


@pytest.fixture(scope="module")
def gpt2():
    return LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)


def test_basic_trace(gpt2):
    """Basic trace with remote='local' works without imported code."""
    with gpt2.trace("Hello", remote='local'):
        hidden = gpt2.transformer.h[0].output[0]
        result = hidden.mean(dim=-1).save()

    assert result.shape[0] == 1  # batch size
