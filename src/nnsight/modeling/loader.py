"""Streaming model weight loading via run:ai SafetensorsStreamer."""

import json
import os
import struct
import threading
from pathlib import Path

import torch

# -- Safetensors dtype string → torch.dtype -----------------------------------

_SAFETENSORS_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_FLOATING_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


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


# -- Lazy shard-by-shard loading -----------------------------------------------


def _parse_safetensors_keys(
    shard_paths: list[str],
) -> tuple[dict[str, tuple[str, list[int], str]], dict[str, int]]:
    """Parse safetensors JSON headers to build key→metadata mapping.

    Reads only the 8-byte length prefix + JSON header per file — no tensor
    data is touched.  ~0.2 ms per file vs ~130 ms for ``safe_open`` on
    networked FS.

    Returns:
        key_map: ``{key: (shard_path, shape, dtype_str)}``
        shard_key_counts: ``{shard_path: num_keys}``
    """
    key_map: dict[str, tuple[str, list[int], str]] = {}
    shard_key_counts: dict[str, int] = {}

    for path in shard_paths:
        with open(path, "rb") as f:
            (header_size,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_size))

        count = 0
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            key_map[key] = (path, meta["shape"], meta["dtype"])
            count += 1
        shard_key_counts[path] = count

    return key_map, shard_key_counts


class RunAIShardCache:
    """Incremental shard cache with per-tensor notification.

    Instead of loading an entire shard before serving any tensor,
    stores tensors one-by-one as Run:AI produces them and wakes
    waiting workers immediately.  This enables pipelining between
    disk I/O (loader thread) and GPU transfers (worker threads).

    Thread coordination handles up to ``GLOBAL_WORKERS`` concurrent callers
    from transformers' ``ThreadPoolExecutor``.
    """

    def __init__(self, concurrency: int, pin_memory: bool = False) -> None:
        self._concurrency = concurrency
        self._pin_memory = pin_memory
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Per-tensor storage (flat, not grouped by shard)
        self._tensors: dict[str, torch.Tensor] = {}

        # Shard-level tracking
        self._refcounts: dict[str, int] = {}
        self._shard_loading: set[str] = set()
        self._errors: dict[str, Exception] = {}

    def register(self, shard_path: str, num_keys: int) -> None:
        """Set the initial refcount for a shard (= number of keys it holds)."""
        self._refcounts[shard_path] = num_keys

    def get(self, shard_path: str, key: str) -> torch.Tensor:
        """Return the cloned tensor for *key*, streaming the shard if needed."""
        should_load = False

        with self._condition:
            while True:
                if shard_path in self._errors:
                    raise self._errors[shard_path]

                if key in self._tensors:
                    return self._pop_tensor(shard_path, key)

                if shard_path not in self._shard_loading:
                    self._shard_loading.add(shard_path)
                    should_load = True
                    break

                # Wait for notification (spurious wakeups handled by loop)
                self._condition.wait()

        if should_load:
            try:
                self._stream_shard_incremental(shard_path)
            except Exception as e:
                with self._condition:
                    self._errors[shard_path] = e
                    self._condition.notify_all()
                raise

            # Loader's own tensor is now available
            with self._condition:
                if shard_path in self._errors:
                    raise self._errors[shard_path]
                return self._pop_tensor(shard_path, key)

    # -- internals -------------------------------------------------------------

    def _pop_tensor(self, shard_path: str, key: str) -> torch.Tensor:
        """Pop *key* from the cache and clean up shard tracking if exhausted."""
        tensor = self._tensors.pop(key)
        self._refcounts[shard_path] -= 1
        if self._refcounts[shard_path] == 0:
            del self._refcounts[shard_path]
            self._shard_loading.discard(shard_path)
        return tensor

    def _stream_shard_incremental(self, shard_path: str) -> None:
        """Stream a shard via run:ai, storing and notifying per tensor."""
        from runai_model_streamer import SafetensorsStreamer

        os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(self._concurrency)

        with SafetensorsStreamer() as streamer:
            streamer.stream_files([shard_path], device="cpu")
            for name, tensor in streamer.get_tensors():
                cloned = tensor.clone()
                if self._pin_memory and torch.cuda.is_available():
                    cloned = cloned.pin_memory()
                with self._condition:
                    self._tensors[name] = cloned
                    self._condition.notify_all()


class LazyRunAITensor:
    """Drop-in replacement for ``SafetensorSlice``.

    Implements the minimal interface that transformers'
    ``convert_and_load_state_dict_in_model`` expects from state-dict values:
    ``__getitem__``, ``.dtype``, ``.shape``, ``.get_shape()``, and
    ``.is_floating_point()``.

    The actual tensor is streamed on the first (and only) ``__getitem__``
    call via the shared :class:`RunAIShardCache`.
    """

    def __init__(
        self,
        shard_path: str,
        key: str,
        cache: RunAIShardCache,
        shape: list[int],
        dtype_str: str,
    ) -> None:
        self._shard_path = shard_path
        self._key = key
        self._cache = cache
        self._shape_list = shape
        self._dtype = _SAFETENSORS_DTYPE[dtype_str]

    def __getitem__(self, idx):
        tensor = self._cache.get(self._shard_path, self._key)
        return tensor[idx]

    def to(self, *args, **kwargs):
        """Materialize the tensor and move it to the target device/dtype."""
        tensor = self._cache.get(self._shard_path, self._key)
        return tensor.to(*args, **kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._shape_list)

    def get_shape(self) -> list[int]:
        return list(self._shape_list)

    def is_floating_point(self) -> bool:
        return self._dtype in _FLOATING_DTYPES


def build_lazy_state_dict(
    shard_paths: list[str],
    concurrency: int = 16,
    pin_memory: bool = False,
) -> dict[str, LazyRunAITensor]:
    """Build a lazy state dict backed by run:ai incremental streaming.

    Values are :class:`LazyRunAITensor` instances that stream the
    underlying safetensors shard on first ``__getitem__`` access.  Tensors
    are streamed incrementally (one-by-one) and workers are notified
    immediately, enabling pipelined disk I/O and GPU transfers.  Shards
    are evicted from CPU memory once all their keys have been consumed,
    keeping peak RSS to ~2-3 shards instead of the full model.

    Args:
        shard_paths: Paths to ``.safetensors`` shard files.
        concurrency: Passed to ``RUNAI_STREAMER_CONCURRENCY``.
        pin_memory: Pin cloned tensors to page-locked memory for faster
            GPU transfers via ``cudaMemcpyAsync``.

    Returns:
        Dict mapping checkpoint key names to :class:`LazyRunAITensor`.
    """
    # Fail fast if run:ai is not installed.
    from runai_model_streamer import SafetensorsStreamer  # noqa: F401

    key_map, shard_key_counts = _parse_safetensors_keys(shard_paths)

    cache = RunAIShardCache(concurrency, pin_memory=pin_memory)
    for shard_path, count in shard_key_counts.items():
        cache.register(shard_path, count)

    return {
        key: LazyRunAITensor(shard_path, key, cache, shape, dtype_str)
        for key, (shard_path, shape, dtype_str) in key_map.items()
    }
