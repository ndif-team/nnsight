from pydantic import BaseModel


class ApiConfigModel(BaseModel):
    HOST: str


class AppConfigModel(BaseModel):
    LOGGING: bool


class ConfigModel(BaseModel):
    API: ApiConfigModel
    APP: AppConfigModel
