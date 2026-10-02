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


class TestRemoteEnv:
    """`get_remote_env` caches one environment per host; no network here — the
    module's HTTP getter is replaced with a counting fake."""

    DEFAULT = "http://default.test"
    OTHER = "http://other.test:5001"

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", self.DEFAULT)
        monkeypatch.setattr(ndif, "_REMOTE_ENVS", {})

    @pytest.fixture
    def fetches(self, monkeypatch):
        """Hosts fetched, in order. Each host serves its own package set."""
        calls = []

        def fake_get(path, timeout=None, headers=None, host=None):
            assert path == "/env"
            host = ndif.resolve_host(host)
            calls.append(host)
            return {
                "python_version": "3.12.0 (main)",
                "packages": {"nnterp": host, "fetch": str(len(calls))},
            }

        monkeypatch.setattr(ndif, "_get", fake_get)
        return calls

    def test_one_fetch_per_host(self, fetches):
        first = ndif.get_remote_env()
        assert ndif.get_remote_env() is first
        assert fetches == [self.DEFAULT]

    def test_hosts_cache_independently(self, fetches):
        default = ndif.get_remote_env()
        other = ndif.get_remote_env(self.OTHER)
        assert default["packages"]["nnterp"] == self.DEFAULT
        assert other["packages"]["nnterp"] == self.OTHER
        # Asking again, in either order, hits neither server.
        assert ndif.get_remote_env(self.OTHER) is other
        assert ndif.get_remote_env() is default
        assert fetches == [self.DEFAULT, self.OTHER]

    def test_force_refresh_refetches_that_host_only(self, fetches):
        ndif.get_remote_env()
        ndif.get_remote_env(self.OTHER)
        refreshed = ndif.get_remote_env(self.OTHER, force_refresh=True)
        assert refreshed["packages"]["fetch"] == "3"
        assert ndif.get_remote_env(self.OTHER) is refreshed
        assert fetches == [self.DEFAULT, self.OTHER, self.OTHER]

    def test_force_refresh_without_a_host(self, fetches):
        ndif.get_remote_env()
        ndif.get_remote_env(force_refresh=True)
        ndif.get_remote_env(True)  # the positional spelling
        assert fetches == [self.DEFAULT] * 3

    def test_default_host_is_read_at_call_time(self, fetches, monkeypatch):
        ndif.get_remote_env()
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", self.OTHER)
        assert ndif.get_remote_env()["packages"]["nnterp"] == self.OTHER
        # The explicit spelling of a host and the configured default share a key.
        assert ndif.get_remote_env(self.DEFAULT)["packages"]["nnterp"] == self.DEFAULT
        assert fetches == [self.DEFAULT, self.OTHER]

    def test_trailing_slash_is_one_host(self, fetches, monkeypatch):
        ndif.get_remote_env(self.OTHER + "/")
        ndif.get_remote_env(self.OTHER)
        monkeypatch.setattr(nnsight.CONFIG.API, "HOST", self.OTHER + "/")
        ndif.get_remote_env()
        assert fetches == [self.OTHER]

    def test_backend_and_cache_spell_a_host_alike(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        backend = RemoteBackend("k", host=self.OTHER + "/")
        assert backend.host == ndif.resolve_host(self.OTHER) == self.OTHER
        assert backend.ws_host == "ws://other.test:5001"
        assert RemoteBackend("k").host == self.DEFAULT

    def test_resolve_host_rejects_a_bare_hostname(self):
        with pytest.raises(ValueError, match="Invalid host URL"):
            ndif.resolve_host("api.ndif.us")

    def test_set_remote_env_seeds_without_a_fetch(self, fetches):
        seeded = {"python_version": "3.11.0", "packages": {"nnterp": "9.9"}}
        ndif.set_remote_env(seeded, self.OTHER + "/")
        assert ndif.get_remote_env(self.OTHER) is seeded
        ndif.set_remote_env(seeded)
        assert ndif.get_remote_env() is seeded
        assert fetches == []

    def test_clear_remote_env(self, fetches):
        ndif.set_remote_env({"packages": {}}, self.OTHER)
        ndif.set_remote_env({"packages": {}})
        ndif.clear_remote_env(self.OTHER)
        ndif.get_remote_env(self.OTHER)
        assert fetches == [self.OTHER]          # the default host is still seeded
        ndif.clear_remote_env()
        ndif.get_remote_env()
        ndif.get_remote_env(self.OTHER)
        assert fetches == [self.OTHER, self.DEFAULT, self.OTHER]

    def test_failed_fetch_is_not_cached(self, monkeypatch):
        import httpx

        attempts = []

        def flaky_get(path, timeout=None, headers=None, host=None):
            attempts.append(host)
            if len(attempts) == 1:
                raise httpx.ConnectError("refused")
            return {"python_version": "3.12.0", "packages": {}}

        monkeypatch.setattr(ndif, "_get", flaky_get)
        with pytest.raises(RuntimeError, match="other.test:5001"):
            ndif.get_remote_env(self.OTHER)
        assert self.OTHER not in ndif._REMOTE_ENVS
        assert ndif.get_remote_env(self.OTHER)["packages"] == {}
        assert len(attempts) == 2

    def test_failed_refresh_keeps_the_cached_env(self, monkeypatch):
        import httpx

        seeded = {"python_version": "3.12.0", "packages": {"nnterp": "1.0"}}
        ndif.set_remote_env(seeded, self.OTHER)

        def down(path, timeout=None, headers=None, host=None):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(ndif, "_get", down)
        with pytest.raises(RuntimeError):
            ndif.get_remote_env(self.OTHER, force_refresh=True)
        assert ndif.get_remote_env(self.OTHER) is seeded

    def test_server_without_env_endpoint_names_the_host(self, monkeypatch):
        import httpx

        def not_found(path, timeout=None, headers=None, host=None):
            request = httpx.Request("GET", f"{host}{path}")
            raise httpx.HTTPStatusError(
                "404", request=request, response=httpx.Response(404, request=request)
            )

        monkeypatch.setattr(ndif, "_get", not_found)
        with pytest.raises(RuntimeError, match=r"http://other\.test:5001 does not serve /env"):
            ndif.get_remote_env(self.OTHER)
        assert ndif._REMOTE_ENVS == {}

    def test_the_request_goes_to_the_named_host(self, monkeypatch):
        import httpx

        urls = []

        def fake_httpx_get(url, **kwargs):
            urls.append(url)
            request = httpx.Request("GET", url)
            if "other" in url:
                return httpx.Response(404, request=request)
            return httpx.Response(200, request=request, json={"packages": {"a": "1"}})

        monkeypatch.setattr(httpx, "get", fake_httpx_get)
        assert ndif.get_remote_env()["packages"] == {"a": "1"}
        with pytest.raises(RuntimeError, match="does not serve /env"):
            ndif.get_remote_env(self.OTHER + "/")
        assert urls == [f"{self.DEFAULT}/env", f"{self.OTHER}/env"]

    def test_compare_uses_the_named_hosts_env(self, fetches):
        assert ndif.compare().remote["packages"]["nnterp"] == self.DEFAULT
        assert ndif.compare(self.OTHER).remote["packages"]["nnterp"] == self.OTHER
        assert nnsight.compare(host=self.OTHER).packages["nnterp"]["remote"] == self.OTHER
        assert fetches == [self.DEFAULT, self.OTHER]


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
