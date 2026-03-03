"""Tests for get_local_env() stdlib filtering."""

import sys

from nnsight.ndif import get_local_env


def test_no_stdlib_modules_marked_as_local():
    """Stdlib modules must never be classified as 'local'.

    Regression test for https://github.com/ndif-team/nnsight/issues/619
    where a platform-dependent path comparison caused stdlib modules like
    `collections` to be registered for pickle-by-value, breaking remote
    trace serialization.
    """
    env = get_local_env()
    packages = env.get("packages", {})

    local_stdlib = {
        name for name in packages
        if packages[name] == "local" and name in sys.stdlib_module_names
    }
    assert local_stdlib == set(), (
        f"Stdlib modules incorrectly marked as local: {sorted(local_stdlib)}"
    )
