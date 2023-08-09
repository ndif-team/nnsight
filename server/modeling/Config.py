from typing import Dict, List

from pydantic import BaseModel


class ModelConfigModel(BaseModel):
    repo_id: str

    device_map: Dict = "auto"
    max_memory: Dict[int, str] = None
    checkpoint_path: str = None


class ConfigModel(BaseModel):
    RESPONSE_PATH: str
    PORT: int
    LOG_PATH: str
    MODEL_CONFIGURATIONS: List[ModelConfigModel]
