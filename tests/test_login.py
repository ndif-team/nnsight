import sys

import nnsight
from nnsight import login, whoami
from nnsight.ndif import main
from nnsight.schema.config import CONFIG

ndif = sys.modules["nnsight.ndif"]


class TestLogin:
    def _capture_keys(self, monkeypatch):
        """Intercept persisted keys without writing the real user config."""
        keys: list[str] = []
        monkeypatch.setattr(
            type(CONFIG), "set_default_api_key", lambda self, key: keys.append(key)
        )
        return keys

    def _stub_whoami(self, monkeypatch, email=None, raises=None):
        def fake(api_key=None):
            if raises is not None:
                raise raises
            return {"email": email, "tags": []}

        monkeypatch.setattr(ndif, "whoami", fake)

    def test_explicit_key_is_stripped_and_saved(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        self._stub_whoami(monkeypatch, email="a@b.c")
        login("  my-key  ")
        assert keys == ["my-key"]

    def test_prompts_when_no_key(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        self._stub_whoami(monkeypatch)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "prompted-key")
        login()
        assert keys == ["prompted-key"]

    def test_empty_input_is_a_noop(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        login("   ")
        assert keys == []

    def test_greets_with_email_when_verified(self, monkeypatch, capsys):
        self._capture_keys(monkeypatch)
        self._stub_whoami(monkeypatch, email="you@ndif.us")
        login("k")
        assert "you@ndif.us" in capsys.readouterr().out

    def test_saves_even_when_verification_fails(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        self._stub_whoami(monkeypatch, raises=RuntimeError("no /whoami here"))
        login("k")
        assert keys == ["k"]  # saved despite the verification error

    def test_cli_login_subcommand(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        self._stub_whoami(monkeypatch, email="a@b.c")
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "cli-key")
        monkeypatch.setattr(sys, "argv", ["nnsight", "login"])
        main()
        assert keys == ["cli-key"]

    def test_cli_no_command_prints_help(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["nnsight"])
        main()
        assert "login" in capsys.readouterr().out


class TestWhoami:
    def test_hits_endpoint_with_key_header(self, monkeypatch):
        calls = {}

        def fake_get(path, headers=None, **kwargs):
            calls["path"], calls["headers"] = path, headers
            return {"email": "x@y.z", "tags": ["dev"]}

        monkeypatch.setattr(ndif, "_get", fake_get)
        out = whoami("explicit-key")
        assert calls["path"] == "/whoami"
        assert calls["headers"] == {"ndif-api-key": "explicit-key"}
        assert out == {"email": "x@y.z", "tags": ["dev"]}
