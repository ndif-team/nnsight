import os
import shutil
from glob import glob

from engine import CONFIG as ENGINE_CONFIG
from huggingface_hub import snapshot_download

from . import CONFIG

# For each model configuration, download the needed files if they don't exist.
# Set checkpoint path of each model.
for model_configuration in CONFIG.MODEL_CONFIGURATIONS:
    model_configuration.checkpoint_path = snapshot_download(
        model_configuration.repo_id,
        force_download=False,
        allow_patterns=["*.bin", "*.json", "*.model"],
        cache_dir=ENGINE_CONFIG.APP.MODEL_CACHE_PATH,
    )

# Get paths of all json files in model cache
json_file_paths = glob(
    os.path.join(ENGINE_CONFIG.APP.MODEL_CACHE_PATH, "**/*.json"), recursive=True
)

# Remove all softlinks and replace them with the actual file.
for json_file_path in json_file_paths:
    real_path = os.path.realpath(json_file_path)
    abs_path = os.path.abspath(json_file_path)

    if real_path != abs_path:
        os.remove(abs_path)
        shutil.copy(real_path, abs_path, follow_symlinks=False)
        os.remove(real_path)
