import os
from typing import Optional

import yaml
from pydantic import BaseModel

class ApiConfigModel(BaseModel):
    HOST: str
    APIKEY: Optional[str]


class AppConfigModel(BaseModel):
    LOGGING: bool


class ConfigModel(BaseModel):
    API: ApiConfigModel
    APP: AppConfigModel

    def set_default_api_key(self, apikey: str):

        self.API.APIKEY = apikey

        self.save()

    def save(self):
        
        from .. import PATH

        with open(os.path.join(PATH, "config.yaml"), "w") as file:

            yaml.dump(self.model_dump(), file)
