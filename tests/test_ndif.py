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


_TG_KEY = (
    'nnsight.modeling.transformers.TransformersModel:'
    '{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}'
)
_FX_KEY = (
    'nnsight.modeling.transformers.TransformersModel:'
    '{"repo_id": "openai-community/gpt2", "revision": null, "task": "feature-extraction"}'
)


def _replica(key, state="RUNNING", level="HOT", revision=None, task="text-generation"):
    # One /status entry, shaped like the controller's: per *replica*, carrying
    # the deployment's model_key and task.
    return {
        "deployment_level": level,
        "application_state": state,
        "model_key": key,
        "repo_id": "openai-community/gpt2",
        "revision": revision,
        "task": task,
    }


class TestStatusGrouping:
    """status() folds replicas of one model_key into one entry and keys by repo
    id, splitting the key only for genuinely distinct deployments."""

    def _serve(self, monkeypatch, deployments):
        monkeypatch.setattr(ndif, "_get", lambda path, **kwargs: {"deployments": deployments})

    def test_two_replicas_are_one_entry(self, monkeypatch):
        # Two replicas of one deployment must not look like a repo-id
        # collision: the entry keeps the plain repo-id key, and a spare
        # replica still DEPLOYING doesn't mask the RUNNING one.
        self._serve(monkeypatch, {
            "a:ModelActor:" + _TG_KEY: _replica(_TG_KEY, "RUNNING"),
            "b:ModelActor:" + _TG_KEY: _replica(_TG_KEY, "DEPLOYING"),
        })
        s = nnsight.status()
        assert "openai-community/gpt2" in s
        assert len(s) == 1
        assert s["openai-community/gpt2"]["state"] == "RUNNING"
        assert s.status is ndif.NdifStatus.Status.UP

    def test_two_tasks_key_per_task(self, monkeypatch):
        self._serve(monkeypatch, {
            "a:ModelActor:" + _TG_KEY: _replica(_TG_KEY, task="text-generation"),
            "b:ModelActor:" + _FX_KEY: _replica(_FX_KEY, task="feature-extraction"),
        })
        s = nnsight.status()
        assert "openai-community/gpt2 (text-generation)" in s
        assert "openai-community/gpt2 (feature-extraction)" in s
        assert len(s) == 2

    def test_same_task_two_revisions_key_by_revision(self, monkeypatch):
        key_b = _TG_KEY.replace('"revision": null', '"revision": "abc123"')
        self._serve(monkeypatch, {
            "a:ModelActor:" + _TG_KEY: _replica(_TG_KEY, revision=None),
            "b:ModelActor:" + key_b: _replica(key_b, revision="abc123"),
        })
        s = nnsight.status()
        assert "openai-community/gpt2 (text-generation, main)" in s
        assert "openai-community/gpt2 (text-generation, abc123)" in s


class TestIsModelRunningTask:
    def _serve(self, monkeypatch, deployments):
        monkeypatch.setattr(ndif, "_get", lambda path, **kwargs: {"deployments": deployments})
        import huggingface_hub

        class _Api:
            def model_info(self, repo_id):
                class _Info:
                    id = repo_id
                return _Info()

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    def test_task_narrows_to_one_deployment(self, monkeypatch):
        self._serve(monkeypatch, {
            "a:ModelActor:" + _TG_KEY: _replica(_TG_KEY, "RUNNING", task="text-generation"),
            "b:ModelActor:" + _FX_KEY: _replica(_FX_KEY, "DEPLOYING", task="feature-extraction"),
        })
        assert nnsight.is_model_running("openai-community/gpt2") is True
        assert nnsight.is_model_running("openai-community/gpt2", task="text-generation") is True
        assert nnsight.is_model_running("openai-community/gpt2", task="feature-extraction") is False
        assert nnsight.is_model_running("openai-community/gpt2", task="absent-task") is False


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
