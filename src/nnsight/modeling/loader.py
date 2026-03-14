"""Streaming model weight loading via run:ai SafetensorsStreamer."""

import os
from pathlib import Path

import torch


def resolve_shard_paths(repo_id: str, revision: str = "main") -> list[str]:
    """Resolve local .safetensors shard paths from HF cache.

    Raises ValueError if no .safetensors files found (no .bin fallback —
    SafetensorsStreamer only handles safetensors format).
    """
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(repo_id, revision=revision, local_files_only=True)
    paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not paths:
        raise ValueError(
            f"No .safetensors files found for {repo_id} (rev={revision}). "
            f"SafetensorsStreamer requires safetensors format."
        )
    return [str(p) for p in paths]


def stream_to_state_dict(
    shard_paths: list[str],
    concurrency: int = 16,
) -> dict[str, torch.Tensor]:
    """Stream safetensors shards to a CPU state dict via run:ai.

    Uses run:ai SafetensorsStreamer for concurrent disk I/O with O_DIRECT,
    bypassing the kernel page cache for consistent performance on
    networked filesystems (e.g., Lustre).

    The returned dict can be passed directly to
    ``from_pretrained(None, state_dict=...)``, which handles all weight
    renaming, conversion (MoE, VLM), dtype casting, device placement,
    and tied weight resolution.

    Args:
        shard_paths: Paths to .safetensors shard files.
        concurrency: Number of concurrent I/O threads for the streamer.

    Returns:
        Dict mapping checkpoint key names to CPU tensors.
    """
    # ImportError propagates to _load() which decides whether to fall back
    from runai_model_streamer import SafetensorsStreamer

    os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)

    state_dict = {}
    with SafetensorsStreamer() as streamer:
        streamer.stream_files(shard_paths, device="cpu")
        for name, tensor in streamer.get_tensors():
            state_dict[name] = tensor

    return state_dict
