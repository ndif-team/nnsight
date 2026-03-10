import os
import warnings
from typing import Optional
import sys
import yaml
from pydantic import BaseModel


class ApiConfigModel(BaseModel):
    HOST: str = "https://api.ndif.us"
    COMPRESS: bool = True
    APIKEY: Optional[str] = None


class AppConfigModel(BaseModel):
    """
    REMOTE_LOGGING: Whether to enable remote logging updates for remote NDIF.
    PYMOUNT: Whether to enable pymount. This allows calling .save() on values in a trace.
        If False, use nnsight.save() instead. Pymounting has some performance cost.
    DEBUG: Whether to enable debug mode. Errors within a trace will include inner nnsight stack traces.
    CACHE_DIR: The directory to cache the model.
    CROSS_INVOKER: Whether to enable cross-invoker. This allows you to refernce variable directly from one invoker to another.
        This has some performance cost.
    """

    REMOTE_LOGGING: bool = True
    PYMOUNT: bool = True
    DEBUG: bool = True
    CACHE_DIR: str = "~/.cache/nnsight/"
    CROSS_INVOKER: bool = True
    TRACE_CACHING: bool = False

    def __setattr__(self, name, value):
        if name == "TRACE_CACHING" and value is True:
            warnings.warn(
                "TRACE_CACHING is deprecated. Trace caching (source, AST, and code object caching) "
                "is now always enabled. Setting TRACE_CACHING has no effect and will be "
                "removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__setattr__(name, value)


class ConfigModel(BaseModel):
    API: ApiConfigModel = ApiConfigModel()
    APP: AppConfigModel = AppConfigModel()

    @classmethod
    def load(cls, path: str) -> "ConfigModel":
        """Load config from YAML file, then apply environment overrides."""
        with open(os.path.join(path, "config.yaml"), "r") as file:
            config = cls(**yaml.safe_load(file))

        config.from_env()
        config.from_cli()

        return config

    def from_env(self) -> None:
        """Override config values from environment variables or Colab userdata."""
        # Check environment variable first
        env_key = os.environ.get("NDIF_API_KEY", None)
        if env_key:
            self.API.APIKEY = env_key
        else:
            # Try Colab userdata
            try:
                from google.colab import userdata

                key = userdata.get("NDIF_API_KEY")
                if key:
                    self.API.APIKEY = key
            except (ImportError, ModuleNotFoundError, Exception):
                pass

        host = os.environ.get("NDIF_HOST", None)
        if host:
            self.API.HOST = host

    def from_cli(self) -> None:

        args = sys.argv
        if "-d" in args or "--debug" in args:
            self.APP.DEBUG = True

    def set_default_api_key(self, apikey: str):

        self.API.APIKEY = apikey

        self.save()

    def set_default_app_debug(self, debug: bool):

        self.APP.DEBUG = debug

        self.save()

    def save(self):

        from .. import PATH

        with open(os.path.join(PATH, "config.yaml"), "w") as file:

            yaml.dump(self.model_dump(), file)
