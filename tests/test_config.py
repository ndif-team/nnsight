"""Config loading precedence, Colab userdata fallback, and set_default_api_key."""

import sys
import types

import yaml

from nnsight.schema.config import Config


class TestEnvOverrides:
    def test_env_api_key_and_host(self, monkeypatch):
        monkeypatch.setenv("NDIF_API_KEY", "env-key")
        monkeypatch.setenv("NDIF_HOST", "http://env-host")
        config = Config.load()
        assert config.API.APIKEY == "env-key"
        assert config.API.HOST == "http://env-host"

    def test_env_debug(self, monkeypatch):
        monkeypatch.setenv("NNSIGHT_DEBUG", "1")
        assert Config.load().APP.DEBUG is True

    def test_user_file_layers_over_defaults(self, monkeypatch, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"API": {"HOST": "http://file-host"}}))
        monkeypatch.setenv("NNSIGHT_CONFIG", str(path))
        monkeypatch.delenv("NDIF_HOST", raising=False)
        assert Config.load().API.HOST == "http://file-host"

    def test_env_wins_over_user_file(self, monkeypatch, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"API": {"HOST": "http://file-host"}}))
        monkeypatch.setenv("NNSIGHT_CONFIG", str(path))
        monkeypatch.setenv("NDIF_HOST", "http://env-host")
        assert Config.load().API.HOST == "http://env-host"


class TestColabUserdata:
    def _install_fake_colab(self, monkeypatch, secrets):
        userdata = types.ModuleType("google.colab.userdata")
        userdata.get = lambda key: secrets[key]  # raises KeyError if missing
        colab = types.ModuleType("google.colab")
        colab.userdata = userdata
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)

    def test_userdata_fallback_when_no_env(self, monkeypatch):
        monkeypatch.delenv("NDIF_API_KEY", raising=False)
        self._install_fake_colab(monkeypatch, {"NDIF_API_KEY": "colab-key"})
        assert Config.load().API.APIKEY == "colab-key"

    def test_env_wins_over_userdata(self, monkeypatch):
        monkeypatch.setenv("NDIF_API_KEY", "env-key")
        self._install_fake_colab(monkeypatch, {"NDIF_API_KEY": "colab-key"})
        assert Config.load().API.APIKEY == "env-key"

    def test_missing_secret_swallowed(self, monkeypatch):
        monkeypatch.delenv("NDIF_API_KEY", raising=False)
        self._install_fake_colab(monkeypatch, {})  # userdata.get -> KeyError
        assert Config.load().API.APIKEY is None


class TestSetDefaultApiKey:
    def test_persists_to_user_file(self, monkeypatch, tmp_path):
        path = tmp_path / "nested" / "config.yaml"
        monkeypatch.setenv("NNSIGHT_CONFIG", str(path))
        monkeypatch.delenv("NDIF_API_KEY", raising=False)

        config = Config.load()
        config.set_default_api_key("saved-key")
        assert config.API.APIKEY == "saved-key"
        assert path.exists()

        # A fresh load picks the key back up from the persisted file.
        assert Config.load().API.APIKEY == "saved-key"
