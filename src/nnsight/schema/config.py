from __future__ import annotations

import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class ApiConfig(BaseModel):
    """Settings for talking to the NDIF service (``CONFIG.API``)."""

    #: Base URL of the NDIF API. Overridden by the ``NDIF_HOST`` env var.
    HOST: str = "https://api.ndif.us"
    #: NDIF API key for remote requests. Set via ``NDIF_API_KEY`` (or a Colab
    #: secret of that name), or persisted with :meth:`Config.set_default_api_key`.
    APIKEY: Optional[str] = None
    #: Whether to zstd-compress the request payload; also tells the server to
    #: compress the result blob it returns.
    COMPRESS: bool = True


class AppConfig(BaseModel):
    """Client-side behavior settings (``CONFIG.APP``)."""

    #: Whether to run remote jobs verbosely (payload/result sizes, per-status
    #: lines) and show full tracebacks. Also enabled by the ``NNSIGHT_DEBUG`` env
    #: var.
    DEBUG: bool = False
    #: Whether to show the live status display while a remote job runs.
    REMOTE_LOGGING: bool = True
    #: Whether to mount ``.save()`` on every object so ``value.save()`` works in a
    #: trace. When ``False`` (or if the C extension didn't build), use
    #: ``nnsight.save(value)`` instead. Mounting adds ``.save`` to all objects
    #: process-wide, so anything checking ``hasattr(x, "save")`` will see it.
    PYMOUNT: bool = True
    #: Whether to neutralize glibc ``backtrace()`` at import so a torch C++ error
    #: raised inside an interleaving greenlet propagates as a normal Python
    #: exception instead of segfaulting the process (see
    #: :mod:`nnsight._c.backtrace`). Only glibc/x86-64 Linux is affected. Turn off
    #: with ``NNSIGHT_DISABLE_CPP_BACKTRACE=0``.
    DISABLE_CPP_BACKTRACE: bool = True


def _read_yaml(path: Any) -> dict:
    # Works for both importlib.resources Traversables and pathlib Paths.
    try:
        text = path.read_text()
    except (FileNotFoundError, OSError):
        return {}
    return yaml.safe_load(text) or {}


def _user_config_path() -> Path:
    override = os.environ.get("NNSIGHT_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    return Path(base).expanduser() / "nnsight" / "config.yaml"


def _colab_userdata(key: str) -> Optional[str]:
    """Best-effort read of a Colab notebook secret; None outside Colab.

    Swallows every error (not-in-Colab, secret missing, access not granted) so a
    non-Colab environment falls through silently.
    """
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        return userdata.get(key) or None
    except Exception:
        return None


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


class Config(BaseModel):
    """The nnsight config, exposed as the module-level singleton :data:`CONFIG`.

    Loaded once at import with the precedence shipped defaults < user config file
    < environment. The user file lives at ``$XDG_CONFIG_HOME/nnsight/config.yaml``
    (or ``NNSIGHT_CONFIG`` if set); :meth:`save` writes the current values back to
    it. Environment overrides: ``NDIF_HOST``, ``NDIF_API_KEY``, ``NNSIGHT_DEBUG``.
    """

    API: ApiConfig = ApiConfig()
    APP: AppConfig = AppConfig()

    @classmethod
    def load(cls) -> Config:
        # shipped defaults < user file < environment
        data = _read_yaml(files("nnsight") / "config.yaml")

        user_path = _user_config_path()
        if user_path is not None and user_path.exists():
            _merge(data, _read_yaml(user_path))

        config = cls(**data)
        config._from_env()
        config._from_cli()
        return config

    def _from_env(self) -> None:
        # NDIF_API_KEY wins; otherwise fall back to a Colab secret of the same
        # name (leaving any key from the config files in place if neither is set).
        api_key = os.environ.get("NDIF_API_KEY")
        if api_key:
            self.API.APIKEY = api_key
        elif self.API.APIKEY is None:
            self.API.APIKEY = _colab_userdata("NDIF_API_KEY")

        host = os.environ.get("NDIF_HOST")
        if host:
            self.API.HOST = host

        if os.environ.get("NNSIGHT_DEBUG"):
            self.APP.DEBUG = True

        # Default-on guard; set NNSIGHT_DISABLE_CPP_BACKTRACE to a falsy value to
        # keep torch's C++ backtrace capture (and the greenlet segfault risk).
        cpp_backtrace = os.environ.get("NNSIGHT_DISABLE_CPP_BACKTRACE")
        if cpp_backtrace is not None:
            self.APP.DISABLE_CPP_BACKTRACE = cpp_backtrace.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
                "",
            )

    def _from_cli(self) -> None:
        # `-v`/`--verbose` on the launching command turns on debug mode (verbose
        # remote logging and full, unfiltered tracebacks).
        if "-v" in sys.argv or "--verbose" in sys.argv:
            self.APP.DEBUG = True

    def save(self) -> None:
        """Persist the current config to the user config file (created if needed)."""
        path = _user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump()))

    def set_default_api_key(self, api_key: str) -> None:
        """Set the NDIF API key and persist it to the user config file."""
        self.API.APIKEY = api_key
        self.save()


CONFIG = Config.load()
