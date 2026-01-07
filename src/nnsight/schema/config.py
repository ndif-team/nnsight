import os
from typing import Optional

import yaml
from pydantic import BaseModel


class ApiConfigModel(BaseModel):
    HOST: str = "api.ndif.us"
    SSL: bool = True
    ZLIB: bool = True
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



class ConfigModel(BaseModel):
    API: ApiConfigModel = ApiConfigModel()
    APP: AppConfigModel = AppConfigModel()

    @classmethod
    def load(cls, path: str) -> "ConfigModel":
        """Load config from YAML file, then apply environment overrides."""
        with open(os.path.join(path, "config.yaml"), "r") as file:
            config = cls(**yaml.safe_load(file))

        config.from_env()
        
        return config

    def from_env(self) -> None:
        """Override config values from environment variables or Colab userdata."""
        if self.API.APIKEY is None:
            # Check environment variable first
            env_key = os.environ.get("NDIF_API_KEY")
            if env_key:
                self.API.APIKEY = env_key
            else:
                # Try Colab userdata
                try:
                    from google.colab import userdata

                    self.API.APIKEY = userdata.get("NDIF_API_KEY")
                except (ImportError, ModuleNotFoundError, Exception):
                    pass

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
