"""Shared fixtures for the offline nnsight test suite.

Model fixtures live here (not in the test modules) so that a test module
registered for serialize-by-value (e.g. by ``remote="local"``'s ``pull_env``)
doesn't have a module-global fixture name colliding with the local model
variable of the same name — which would pull the (decorated) fixture function
into the pickle graph.
"""


import pytest


@pytest.fixture(scope="module")
def gpt2():
    """A dispatched gpt2 for offline model tests."""
    from nnsight.modeling.transformers import TransformersModel

    return TransformersModel("openai-community/gpt2", dispatch=True)
