from huggingface_hub import snapshot_download

from . import CONFIG

# For each model configuration, download the needed files if they don't exist.
# Set checkpoint path of each model.
for model_configuration in CONFIG.MODEL_CONFIGURATIONS:
    model_configuration.checkpoint_path = snapshot_download(
        model_configuration.repo_id,
        force_download=False,
        allow_patterns=["*.bin", "*.json", "*.model"],
    )
