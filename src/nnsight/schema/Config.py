import os
from typing import Optional

import yaml
from pydantic import BaseModel

class ApiConfigModel(BaseModel):
    HOST: str = "ndif.dev"
    SSL: bool = True
    APIKEY: Optional[str] = None
    JOB_ID:Optional[str] = None


class AppConfigModel(BaseModel):
    LOGGING: bool = False
    REMOTE_LOGGING: bool = True


class ConfigModel(BaseModel):
    API: ApiConfigModel = ApiConfigModel()
    APP: AppConfigModel = AppConfigModel()

    def set_default_api_key(self, apikey: str):

        self.API.APIKEY = apikey

        self.save()

    def save(self):
        
        from .. import PATH

        with open(os.path.join(PATH, "config.yaml"), "w") as file:

            yaml.dump(self.model_dump(), file)
