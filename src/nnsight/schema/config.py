import os
from typing import Optional

import yaml
from pydantic import BaseModel

from ..log import remote_logger


class ApiConfigModel(BaseModel):
    HOST: str = "ndif.dev"
    SSL: bool = True
    ZLIB: bool = True
    APIKEY: Optional[str] = None
    JOB_ID: Optional[str] = None


class AppConfigModel(BaseModel):
    REMOTE_LOGGING: bool = True
    DEBUG: bool = True
    CACHE_DIR:str = '~/.cache/nnsight/'


    def __setattr__(self, name, value):
        if name == "REMOTE_LOGGING":
            self.on_remote_logging_change(value)
        super().__setattr__(name, value)

    def on_remote_logging_change(self, value: bool):
        if value != self.REMOTE_LOGGING:
            remote_logger.disabled = (not value)
        self.__dict__["REMOTE_LOGGING"] = value


class ConfigModel(BaseModel):
    API: ApiConfigModel = ApiConfigModel()
    APP: AppConfigModel = AppConfigModel()

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