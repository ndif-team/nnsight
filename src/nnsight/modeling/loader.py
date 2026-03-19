"""Streaming model weight loading via run:ai SafetensorsStreamer."""

import json
import os
import struct
import time
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

    Stores tensors one-by-one as Run:AI produces them and wakes waiting
    workers immediately.  This enables pipelining between disk I/O
    (loader thread) and GPU transfers (worker threads).

    When ``device_map`` is provided and contains GPU targets, tensors are
    copied **directly from the Run:AI buffer to GPU** during streaming,
    bypassing the intermediate CPU clone.  HF's ``_materialize_copy().to()``
    then becomes a no-op (tensor is already on the target device/dtype).

    Thread coordination handles up to ``GLOBAL_WORKERS`` concurrent callers
    from transformers' ``ThreadPoolExecutor``.
    """

    def __init__(
        self,
        concurrency: int,
        device_map: dict | None = None,
        torch_dtype: torch.dtype | None = None,
        lazy: bool = False,
    ) -> None:
        self._concurrency = concurrency
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._gpu_direct = (
            device_map is not None and torch.cuda.is_available()
        )
        self._lazy = lazy
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Per-tensor storage (flat, not grouped by shard)
        self._tensors: dict[str, torch.Tensor] = {}

        # Shard-level tracking
        self._refcounts: dict[str, int] = {}
        self._shard_loading: set[str] = set()
        self._errors: dict[str, Exception] = {}

        # Lazy-mode iterator state (persistent across get() calls)
        self._shard_contexts: dict = {}     # shard_path → SafetensorsStreamer ctx
        self._shard_iterators: dict = {}    # shard_path → iterator
        self._shard_exhausted: set[str] = set()
        self._expandable_set = False

        # Profiling stats (thread-safe via _lock/_condition)
        self.stats_shard_wall_s = 0.0  # wall time of _stream_shard_incremental
        self.stats_io_wait_s = 0.0     # time waiting for streamer (uncovered I/O)
        self.stats_clone_s = 0.0       # time cloning tensors (CPU-target path)
        self.stats_gpu_copy_s = 0.0    # time in buffer→GPU .to() copies
        self.stats_notify_s = 0.0      # time holding lock for store+notify
        self.stats_consumer_wait_s = 0.0  # cumulative consumer wait on Condition
        self.stats_pop_count = 0       # tensors served

    def register(self, shard_path: str, num_keys: int) -> None:
        """Set the initial refcount for a shard (= number of keys it holds)."""
        self._refcounts[shard_path] = num_keys

    def get(self, shard_path: str, key: str) -> torch.Tensor:
        """Return the tensor for *key*, streaming the shard if needed."""
        if self._lazy:
            return self._get_lazy(shard_path, key)
        return self._get_eager(shard_path, key)

    def _get_eager(self, shard_path: str, key: str) -> torch.Tensor:
        """Eager: preload entire shard, then pop the requested key."""
        should_load = False
        t_wait_start = time.perf_counter()

        with self._condition:
            while True:
                if shard_path in self._errors:
                    raise self._errors[shard_path]

                if key in self._tensors:
                    self.stats_consumer_wait_s += time.perf_counter() - t_wait_start
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

    def _get_lazy(self, shard_path: str, key: str) -> torch.Tensor:
        """Lazy: advance shard iterator only until the requested key."""
        t_wait_start = time.perf_counter()

        while True:
            with self._condition:
                if shard_path in self._errors:
                    raise self._errors[shard_path]

                if key in self._tensors:
                    self.stats_consumer_wait_s += (
                        time.perf_counter() - t_wait_start
                    )
                    return self._pop_tensor(shard_path, key)

                if shard_path in self._shard_exhausted:
                    raise KeyError(
                        f"Key {key!r} not found in exhausted shard "
                        f"{shard_path!r}"
                    )

                if shard_path in self._shard_loading:
                    self._condition.wait()
                    continue

                self._shard_loading.add(shard_path)
                break

        try:
            self._stream_until(shard_path, key)
        except Exception as e:
            with self._condition:
                self._errors[shard_path] = e
                self._shard_loading.discard(shard_path)
                self._condition.notify_all()
            raise

        with self._condition:
            self._shard_loading.discard(shard_path)
            self._condition.notify_all()
            self.stats_consumer_wait_s += (
                time.perf_counter() - t_wait_start
            )
            if shard_path in self._errors:
                raise self._errors[shard_path]
            return self._pop_tensor(shard_path, key)

    # -- internals -------------------------------------------------------------

    def _pop_tensor(self, shard_path: str, key: str) -> torch.Tensor:
        """Pop *key* from the cache and clean up shard tracking if exhausted."""
        tensor = self._tensors.pop(key)
        self._refcounts[shard_path] -= 1
        self.stats_pop_count += 1
        if self._refcounts[shard_path] == 0:
            del self._refcounts[shard_path]
            self._shard_loading.discard(shard_path)
            if self._lazy:
                self._close_shard_streamer(shard_path)
                self._shard_exhausted.discard(shard_path)
        return tensor

    def _resolve_device(self, checkpoint_key: str) -> torch.device:
        """Map a checkpoint key to its target device via the device_map."""
        from transformers.integrations.accelerate import expand_device_map

        expanded = expand_device_map(self._device_map, [checkpoint_key])
        device = expanded[checkpoint_key]
        if isinstance(device, int):
            return torch.device("cuda", device)
        if device == "disk":
            return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, tensor: torch.Tensor) -> torch.dtype | None:
        """Return target dtype for a tensor, or None to keep original.

        Only floating-point tensors are cast; integer/bool buffers are
        left as-is.
        """
        if self._torch_dtype is None:
            return None
        if tensor.dtype in _FLOATING_DTYPES:
            return self._torch_dtype
        return None

    @staticmethod
    def _enable_expandable_segments() -> None:
        """Enable CUDA expandable segments to avoid fragmentation.

        Streaming hundreds of individually-sized tensors to GPU causes
        fragmentation in the default CUDA caching allocator.  Expandable
        segments let the allocator grow incrementally instead of reserving
        large fixed blocks, with no measurable performance penalty.
        """
        key = "PYTORCH_CUDA_ALLOC_CONF"
        conf = os.environ.get(key, "")
        if "expandable_segments" not in conf:
            entry = "expandable_segments:True"
            os.environ[key] = f"{conf},{entry}" if conf else entry

    def _stream_shard_incremental(self, shard_path: str) -> None:
        """Stream a shard via run:ai, storing and notifying per tensor."""
        from runai_model_streamer import SafetensorsStreamer

        os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(self._concurrency)

        # Avoid CUDA memory fragmentation from streaming many varied-size tensors
        if self._gpu_direct:
            self._enable_expandable_segments()

        t_shard_start = time.perf_counter()
        t_iter_start = t_shard_start
        with SafetensorsStreamer() as streamer:
            streamer.stream_files([shard_path], device="cpu")
            for name, tensor in streamer.get_tensors():
                t_yield = time.perf_counter()
                io_wait = t_yield - t_iter_start

                if self._gpu_direct:
                    target_device = self._resolve_device(name)
                    target_dtype = self._resolve_dtype(tensor)
                    if target_device.type == "cuda":
                        # Direct buffer → GPU (one copy, no intermediate clone)
                        result = tensor.to(
                            device=target_device,
                            dtype=target_dtype or tensor.dtype,
                        )
                        t_copied = time.perf_counter()
                        with self._condition:
                            self.stats_io_wait_s += io_wait
                            self.stats_gpu_copy_s += t_copied - t_yield
                            self._tensors[name] = result
                            self._condition.notify_all()
                    else:
                        # CPU/disk target — must clone since buffer is transient
                        cloned = tensor.clone()
                        t_cloned = time.perf_counter()
                        if target_dtype is not None:
                            cloned = cloned.to(dtype=target_dtype)
                        with self._condition:
                            self.stats_io_wait_s += io_wait
                            self.stats_clone_s += t_cloned - t_yield
                            self._tensors[name] = cloned
                            self._condition.notify_all()
                else:
                    # CPU-only path (no device_map or no CUDA)
                    cloned = tensor.clone()
                    t_cloned = time.perf_counter()
                    with self._condition:
                        self.stats_io_wait_s += io_wait
                        self.stats_clone_s += t_cloned - t_yield
                        self._tensors[name] = cloned
                        self._condition.notify_all()

                t_iter_start = time.perf_counter()
        shard_wall = time.perf_counter() - t_shard_start

        with self._condition:
            self.stats_shard_wall_s += shard_wall

    # -- lazy streaming helpers ------------------------------------------------

    def _ensure_shard_iterator(self, shard_path: str):
        """Open a streamer for *shard_path* if not already open."""
        if shard_path in self._shard_iterators:
            return self._shard_iterators[shard_path]

        from runai_model_streamer import SafetensorsStreamer

        os.environ["RUNAI_STREAMER_CONCURRENCY"] = str(self._concurrency)
        if self._gpu_direct and not self._expandable_set:
            self._enable_expandable_segments()
            self._expandable_set = True

        streamer = SafetensorsStreamer()
        streamer.__enter__()
        streamer.stream_files([shard_path], device="cpu")

        self._shard_contexts[shard_path] = streamer
        self._shard_iterators[shard_path] = iter(streamer.get_tensors())
        return self._shard_iterators[shard_path]

    def _close_shard_streamer(self, shard_path: str) -> None:
        """Close and discard the streamer context for *shard_path*."""
        ctx = self._shard_contexts.pop(shard_path, None)
        self._shard_iterators.pop(shard_path, None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    def _place_tensor(
        self, name: str, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Copy *tensor* to its target device/dtype.

        Returns ``(result, gpu_copy_seconds, clone_seconds)``.
        """
        gpu_copy_s = 0.0
        clone_s = 0.0

        if self._gpu_direct:
            target_device = self._resolve_device(name)
            target_dtype = self._resolve_dtype(tensor)
            if target_device.type == "cuda":
                t0 = time.perf_counter()
                result = tensor.to(
                    device=target_device,
                    dtype=target_dtype or tensor.dtype,
                )
                gpu_copy_s = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                result = tensor.clone()
                clone_s = time.perf_counter() - t0
                if target_dtype is not None:
                    result = result.to(dtype=target_dtype)
        else:
            t0 = time.perf_counter()
            result = tensor.clone()
            clone_s = time.perf_counter() - t0

        return result, gpu_copy_s, clone_s

    def _stream_until(self, shard_path: str, target_key: str) -> None:
        """Advance the shard iterator until *target_key* is cached."""
        iterator = self._ensure_shard_iterator(shard_path)

        t_stream_start = time.perf_counter()
        t_iter_start = t_stream_start
        for name, tensor in iterator:
            t_yield = time.perf_counter()
            io_wait = t_yield - t_iter_start

            result, gpu_copy_s, clone_s = self._place_tensor(name, tensor)

            with self._condition:
                self.stats_io_wait_s += io_wait
                self.stats_gpu_copy_s += gpu_copy_s
                self.stats_clone_s += clone_s
                self._tensors[name] = result
                self._condition.notify_all()
                if name == target_key:
                    self.stats_shard_wall_s += (
                        time.perf_counter() - t_stream_start
                    )
                    return

            t_iter_start = time.perf_counter()

        # Iterator exhausted
        self._close_shard_streamer(shard_path)
        with self._condition:
            self._shard_exhausted.add(shard_path)
            self.stats_shard_wall_s += (
                time.perf_counter() - t_stream_start
            )


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
    device_map: dict | None = None,
    torch_dtype: torch.dtype | None = None,
    lazy: bool = False,
) -> dict[str, LazyRunAITensor]:
    """Build a lazy state dict backed by run:ai streaming.

    Values are :class:`LazyRunAITensor` instances that stream the
    underlying safetensors shard on first ``__getitem__`` access.  Shards
    are evicted from memory once all their keys have been consumed,
    keeping peak memory to ~1 tensor instead of the full model.

    Args:
        shard_paths: Paths to ``.safetensors`` shard files.
        concurrency: Passed to ``RUNAI_STREAMER_CONCURRENCY``.
        device_map: Resolved device map (param name → device).  When
            provided with GPU targets, the streaming cache copies tensors
            directly from the Run:AI buffer to GPU, so that HF's
            ``_materialize_copy().to()`` is a no-op.
        torch_dtype: Target dtype for floating-point tensors.  Applied
            during GPU transfer so HF's dtype cast is also a no-op.

    Returns:
        Dict mapping checkpoint key names to :class:`LazyRunAITensor`.
    """
    # Fail fast if run:ai is not installed.
    from runai_model_streamer import SafetensorsStreamer  # noqa: F401

    key_map, shard_key_counts = _parse_safetensors_keys(shard_paths)

    cache = RunAIShardCache(
        concurrency, device_map=device_map, torch_dtype=torch_dtype,
        lazy=lazy,
    )
    for shard_path, count in shard_key_counts.items():
        cache.register(shard_path, count)

    return {
        key: LazyRunAITensor(shard_path, key, cache, shape, dtype_str)
        for key, (shard_path, shape, dtype_str) in key_map.items()
    }
