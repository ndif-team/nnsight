"""Offline unit tests for the nnsight.ndif helpers.

Network-touching paths (status/is_model_running) are exercised only against an
unreachable host to check graceful failure; the live-server behavior is covered
by the remote suite.
"""


import pytest

import nnsight
from nnsight import ndif


def _entry(state="RUNNING", level="HOT"):
    return {
        "model_class": "TransformersModel",
        "repo_id": "openai-community/gpt2",
        "revision": "main",
        "level": level,
        "state": state,
    }


class TestRegister:
    def test_register_by_name(self):
        from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES

        ndif.register("some_local_pkg_xyz")
        assert "some_local_pkg_xyz" in _PICKLE_BY_VALUE_MODULES

    def test_register_by_module(self):
        import json

        from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES

        ndif.register(json)
        assert "json" in _PICKLE_BY_VALUE_MODULES


class TestNdifStatus:
    def test_status_up_when_any_running(self):
        s = ndif.NdifStatus({"gpt2": _entry("RUNNING")})
        assert s.status is ndif.NdifStatus.Status.UP

    def test_status_redeploying_when_deploying(self):
        s = ndif.NdifStatus({"gpt2": _entry("DEPLOYING")})
        assert s.status is ndif.NdifStatus.Status.REDEPLOYING

    def test_status_down_when_empty(self):
        assert ndif.NdifStatus({}).status is ndif.NdifStatus.Status.DOWN

    def test_dict_like_access(self):
        s = ndif.NdifStatus({"openai-community/gpt2": _entry()})
        assert "openai-community/gpt2" in s
        assert len(s) == 1
        assert list(s.keys()) == ["openai-community/gpt2"]
        assert s["openai-community/gpt2"]["state"] == "RUNNING"

    def test_str_renders_table(self):
        text = str(ndif.NdifStatus({"gpt2": _entry()}))
        assert "Up" in text
        assert "openai-community/gpt2" in text
        assert "RUNNING" in text and "HOT" in text


class TestEnvComparison:
    def test_get_local_env_shape(self):
        env = ndif.get_local_env()
        assert "python_version" in env
        assert isinstance(env["packages"], dict)
        assert "nnsight" in env["packages"]

    def test_build_table_flags_mismatches(self):
        local = {"packages": {"torch": "2.0.0", "foo": "1.0"}}
        remote = {"packages": {"torch": "2.1.0", "foo": "1.0", "bar": "3.0"}}
        table = ndif.build_table(local, remote)
        assert "CRITICAL" in table          # torch (critical) mismatches
        assert "bar" in table               # remote-only package shown
        assert "foo" in table               # matching package shown

    def test_comparison_object_is_inspectable(self):
        local = {"python_version": "3.12.0 (main)", "packages": {"torch": "2.0.0", "foo": "1.0"}}
        remote = {"python_version": "3.12.0 (main)", "packages": {"torch": "2.1.0", "foo": "1.0", "bar": "3.0"}}
        cmp = ndif.EnvComparison(local, remote)
        assert cmp.python_matches is True
        assert set(cmp.mismatches) == {"torch", "bar"}   # foo matches
        assert set(cmp.critical_mismatches) == {"torch"}  # torch is critical
        assert cmp.packages["foo"]["match"] is True

    def test_comparison_object_is_printable(self):
        local = {"python_version": "3.12.0", "packages": {"torch": "2.0.0"}}
        remote = {"python_version": "3.11.0", "packages": {"torch": "2.1.0"}}
        text = str(ndif.EnvComparison(local, remote))
        assert "Python Version:" in text
        assert "torch" in text and "CRITICAL" in text


class TestPullEnv:
    def test_registers_only_local_modules_and_caches(self, monkeypatch):
        from nnsight import ndif

        monkeypatch.setattr(ndif, "_PULLED_ENV", False)
        monkeypatch.setattr(
            ndif,
            "get_local_env",
            lambda: {"packages": {"mypkg_local": "local", "torch": "2.0.0"}},
        )
        registered = []
        monkeypatch.setattr(ndif, "register", lambda m: registered.append(m))

        ndif.pull_env()
        assert registered == ["mypkg_local"]  # only the "local" one, not torch
        assert ndif._PULLED_ENV is True

        # Cached: a second call is a no-op.
        registered.clear()
        ndif.pull_env()
        assert registered == []


class TestGracefulFailure:
    def test_status_unreachable_is_down(self, monkeypatch):
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", "http://localhost:1")
        s = nnsight.status()
        assert isinstance(s, ndif.NdifStatus)
        assert s.status is ndif.NdifStatus.Status.DOWN

    def test_is_model_running_unreachable_is_false(self, monkeypatch):
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", "http://localhost:1")
        assert nnsight.is_model_running("openai-community/gpt2") is False


class TestExports:
    def test_top_level_exports(self):
        for name in ("register", "status", "ndif_status", "is_model_running", "compare"):
            assert hasattr(nnsight, name)

    def test_ndif_status_deprecated(self, monkeypatch):
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", "http://localhost:1")
        with pytest.warns(nnsight.NNsightDeprecationWarning):
            nnsight.ndif_status()
