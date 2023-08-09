from pydantic import BaseModel


class AppConfigModel(BaseModel):
    MODEL_CACHE_PATH: str


class ApiConfigModel(BaseModel):
    HOST: str


class ConfigModel(BaseModel):
    API: ApiConfigModel
    APP: AppConfigModel
