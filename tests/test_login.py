import sys

import nnsight
from nnsight.login import main
from nnsight.schema.config import CONFIG

# The submodule (login.py), reached unambiguously despite the `login` function
# sharing its name in the nnsight namespace.
_login_module = sys.modules["nnsight.login"]


class TestLogin:
    def _capture_keys(self, monkeypatch):
        """Intercept persisted keys without writing the real user config."""
        keys: list[str] = []
        monkeypatch.setattr(
            type(CONFIG), "set_default_api_key", lambda self, key: keys.append(key)
        )
        return keys

    def test_explicit_key_is_stripped_and_saved(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        nnsight.login("  my-key  ")
        assert keys == ["my-key"]

    def test_prompts_when_no_key(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        monkeypatch.setattr(_login_module, "getpass", lambda prompt="": "prompted-key")
        nnsight.login()
        assert keys == ["prompted-key"]

    def test_empty_input_is_a_noop(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        nnsight.login("   ")
        assert keys == []

    def test_cli_login_subcommand(self, monkeypatch):
        keys = self._capture_keys(monkeypatch)
        monkeypatch.setattr(_login_module, "getpass", lambda prompt="": "cli-key")
        monkeypatch.setattr(sys, "argv", ["nnsight", "login"])
        main()
        assert keys == ["cli-key"]

    def test_cli_no_command_prints_help(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["nnsight"])
        main()  # no subcommand -> help, no error
        assert "login" in capsys.readouterr().out
