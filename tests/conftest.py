"""
Pytest configuration and shared fixtures for nnsight tests.

This module contains:
- Command-line options (--device, --tp)
- Shared fixtures for prompts and models
- Test collection configuration
"""

import pytest
import toml
import torch
from collections import OrderedDict

# =============================================================================
# Command Line Options
# =============================================================================


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--device",
        action="store",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run tests on (default: cuda:0 if available, else cpu)",
    )
    parser.addoption(
        "--tp",
        action="store",
        type=int,
        default=1,
        help="Tensor parallel size for VLLM tests (default: 1)",
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests based on command-line options."""
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value], scope="module")


# =============================================================================
# Project Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def load_pyproject_toml():
    """Load and parse the pyproject.toml file."""
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)
        return data
    except toml.TomlDecodeError as e:
        pytest.fail(f"Failed to load pyproject.toml: {e}")


# =============================================================================
# Common Prompt Fixtures
# =============================================================================


@pytest.fixture
def ET_prompt():
    """Eiffel Tower prompt for testing."""
    return "The Eiffel Tower is located in the city of"


@pytest.fixture
def MSG_prompt():
    """Madison Square Garden prompt for testing."""
    return "Madison Square Garden is located in the city of"


# =============================================================================
# Tiny Model Fixtures (for base NNsight tests)
# =============================================================================

INPUT_SIZE = 5
HIDDEN_DIMS = 10
OUTPUT_SIZE = 2


@pytest.fixture(scope="module")
def tiny_model(device: str):
    """Create a simple two-layer model wrapped with NNsight."""
    from nnsight import NNsight

    net = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(INPUT_SIZE, HIDDEN_DIMS)),
                ("layer2", torch.nn.Linear(HIDDEN_DIMS, OUTPUT_SIZE)),
            ]
        )
    )
    return NNsight(net).to(device)


@pytest.fixture
def tiny_input():
    """Random input tensor for tiny model tests."""
    return torch.rand((1, INPUT_SIZE))


# =============================================================================
# Language Model Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def gpt2(device: str):
    """Load GPT-2 model with nnsight."""
    import nnsight

    return nnsight.LanguageModel(
        "openai-community/gpt2", device_map=device, dispatch=True
    )


# =============================================================================
# Test Collection Configuration
# =============================================================================

collect_ignore = ["examples/test_server.py", "examples/test_server_llama.py"]

