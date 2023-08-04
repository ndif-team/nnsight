import os
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Dict

from engine import CONFIG
from huggingface_hub import snapshot_download


@dataclass
class InferenceConfiguration:
    repo_id: str
    device_map: Dict = "auto"
    max_memory: Dict[int, str] = None

    checkpoint_path: str = None


inference_configurations = [
    InferenceConfiguration("gpt2"),
    InferenceConfiguration(
        "decapoda-research/llama-65b-hf",
        # {
        #     0: "0GiB",
        #     1: "0GiB",
        #     2: "0GiB",
        #     3: "0GiB",
        #     4: "86GiB",
        #     5: "86GiB",
        #     6: "86GiB",
        #     7: "86GiB",
        # },
    ),
]

for inference_configuration in inference_configurations:
    inference_configuration.checkpoint_path = snapshot_download(
        inference_configuration.repo_id,
        force_download=False,
        allow_patterns=["*.bin", "*.json", "*.model"],
        cache_dir=CONFIG["APP"]["MODEL_CACHE_PATH"],
    )


json_file_paths = glob(os.path.join(CONFIG["APP"]["MODEL_CACHE_PATH"], "**/*.json"), recursive=True)

for json_file_path in json_file_paths:
    real_path = os.path.realpath(json_file_path)
    abs_path = os.path.abspath(json_file_path)

    if real_path != abs_path:
        
        os.remove(abs_path)
        shutil.copy(real_path, abs_path, follow_symlinks=False)
        os.remove(real_path)
