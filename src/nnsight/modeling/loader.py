"""Streaming model weight loading via run:ai SafetensorsStreamer."""

import os
from pathlib import Path
from typing import Optional

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


def _resolve_device_for_param(
    param_name: str, device_map: dict[str, str]
) -> str:
    """Walk up the module path to find the device assignment for a parameter."""
    parts = param_name.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in device_map:
            return device_map[prefix]
    return device_map.get("", "cpu")


def _detect_key_prefix(
    model_keys: set[str], safetensors_keys: set[str]
) -> str:
    """Detect the prefix that maps safetensors keys to model parameter names.

    HuggingFace safetensors files often strip the base model prefix
    (e.g., ``transformer.``) from parameter names. This function detects
    that prefix by comparing the two key sets.

    Returns:
        The prefix string (e.g., ``"transformer."``), or ``""`` if keys
        match directly.
    """
    # Fast path: keys match directly
    if safetensors_keys <= model_keys:
        return ""

    # Find a safetensors key that doesn't match any model key directly
    sample_key = next(k for k in safetensors_keys if k not in model_keys)

    # Try to find it in model_keys with a prefix
    for model_key in model_keys:
        if model_key.endswith("." + sample_key):
            candidate_prefix = model_key[: -(len(sample_key))]
            # Verify this prefix works for most keys
            hits = sum(
                1 for k in safetensors_keys if (candidate_prefix + k) in model_keys
            )
            if hits > len(safetensors_keys) * 0.5:
                return candidate_prefix

    return ""


def stream_weights_into_model(
    model: torch.nn.Module,
    shard_paths: list[str],
    device_map: dict[str, str],
    *,
    torch_dtype: Optional[torch.dtype] = None,
    concurrency: int = 16,
) -> None:
    """Stream safetensor weights into a meta-initialized model.

    Uses run:ai SafetensorsStreamer for concurrent disk I/O with pipelined
    pinned-memory GPU transfers.

    Args:
        model: Model initialized on meta device (from ``from_config()``
            inside ``init_empty_weights()``).
        shard_paths: Paths to .safetensors shard files.
        device_map: Module-name → device mapping (from ``infer_auto_device_map``).
        torch_dtype: Target dtype for weights. If None, keeps original dtype.
        concurrency: Number of concurrent I/O threads for the streamer.
    """
    # ImportError propagates to _load() which decides whether to fall back
    from runai_model_streamer import SafetensorsStreamer

    from accelerate.utils import set_module_tensor_to_device

    # Detect prefix mismatch between safetensors keys and model parameter names.
    # HF safetensors files often strip the base model prefix (e.g., "transformer.").
    import safetensors

    safetensors_keys = set()
    for path in shard_paths:
        with safetensors.safe_open(path, framework="pt") as f:
            safetensors_keys.update(f.keys())

    model_keys = set(
        n for n, _ in list(model.named_parameters()) + list(model.named_buffers())
    )
    prefix = _detect_key_prefix(model_keys, safetensors_keys)

    # Build param-name → device lookup using model parameter names
    param_device = {}
    for name in model_keys:
        param_device[name] = _resolve_device_for_param(name, device_map)

    os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(concurrency)

    with SafetensorsStreamer() as streamer:
        streamer.stream_files(shard_paths, device="cpu")
        for name, tensor in streamer.get_tensors():
            # Map safetensors key to model parameter name
            model_name = prefix + name if (prefix + name) in model_keys else name

            target = param_device.get(model_name, "cpu")
            if target not in ("cpu", "disk"):
                # Pipelined: cast → pin → async GPU copy
                if torch_dtype is not None:
                    tensor = tensor.to(dtype=torch_dtype)
                tensor = tensor.pin_memory()
                set_module_tensor_to_device(
                    model, model_name, target,
                    value=tensor.to(target, non_blocking=True),
                )
            else:
                if torch_dtype is not None:
                    tensor = tensor.to(dtype=torch_dtype)
                set_module_tensor_to_device(
                    model, model_name, target, value=tensor
                )
            del tensor

    if torch.cuda.is_available():
        torch.cuda.synchronize()
