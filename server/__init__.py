import os
import shutil
from glob import glob

import yaml
from engine import CONFIG as ENGINE_CONFIG
from huggingface_hub import snapshot_download

from .modeling import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

os.makedirs(CONFIG.LOG_PATH, exist_ok=True)

for model_configuration in CONFIG.MODEL_CONFIGURATIONS:
    model_configuration.checkpoint_path = snapshot_download(
        model_configuration.repo_id,
        force_download=False,
        allow_patterns=["*.bin", "*.json", "*.model"],
        cache_dir=CONFIG.APP.MODEL_CACHE_PATH,
    )


json_file_paths = glob(
    os.path.join(ENGINE_CONFIG.APP.MODEL_CACHE_PATH, "**/*.json"), recursive=True
)

for json_file_path in json_file_paths:
    real_path = os.path.realpath(json_file_path)
    abs_path = os.path.abspath(json_file_path)

    if real_path != abs_path:
        os.remove(abs_path)
        shutil.copy(real_path, abs_path, follow_symlinks=False)
        os.remove(real_path)
