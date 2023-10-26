from pydantic import BaseModel


class ApiConfigModel(BaseModel):
    HOST: str


class ConfigModel(BaseModel):
    API: ApiConfigModel
