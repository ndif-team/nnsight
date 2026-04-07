"""PP Listener — cross-rank tensor pull via a dedicated gloo process group.

Uses tags to separate request and response traffic on the same group:
  - TAG_REQUEST (0): consumer sends pull requests, producer's listener recvs
  - TAG_RESPONSE (1): producer's listener sends data back, consumer recvs

This avoids concurrent recv on the same (group, tag) from different threads.

Buffers store narrowed (per-mediator) tensors on GPU; moved to CPU at pull time.
Dtype and shape are resolved locally from a shared metadata map built at model
load time — no metadata on the wire for modules with known shapes.  Modules
without metadata fall back to a legacy protocol that sends shape info.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

TAG_REQUEST = 0
TAG_RESPONSE = 1
_META_SLOTS = 32  # legacy metadata buffer size


class PPListener:
    """Cross-rank tensor pull service.

    Producer (background listener thread): recvs on TAG_REQUEST,
    sends on TAG_RESPONSE.

    Consumer (main/mediator thread): sends on TAG_REQUEST,
    recvs on TAG_RESPONSE.

    Request header is ``[source_rank, key_len, num_tokens]``.
    When ``num_tokens > 0``, the consumer has pre-computed the recv
    buffer size from shared metadata — producer sends flat data only.
    When ``num_tokens == 0``, legacy mode: producer sends shape
    metadata then flat data.

    No thread ever does concurrent recv on the same (group, tag).
    """

    def __init__(
        self,
        buffer: Dict[str, Any],
        condition: threading.Condition,
        pull_group: Optional[dist.ProcessGroup],
        local_rank: int,
        device: torch.device,
        meta_map: Optional[Dict[str, dict]] = None,
    ):
        self._buffer = buffer
        self._condition = condition
        self._pull_group = pull_group
        self._local_rank = local_rank
        self._device = device
        self._meta_map = meta_map or {}
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Local buffer lookup (blocks until value available)
    # ------------------------------------------------------------------

    def local_lookup(
        self,
        provider_string: str,
        timeout: Optional[float] = 60.0,
    ) -> torch.Tensor:
        with self._condition:
            while provider_string not in self._buffer:
                if not self._condition.wait(timeout=timeout):
                    raise TimeoutError(
                        f"PPListener: timed out waiting for {provider_string!r}"
                    )
            return self._buffer[provider_string]

    # ------------------------------------------------------------------
    # Producer: background listener thread
    # ------------------------------------------------------------------

    def start(self):
        if self._pull_group is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="pp-listener"
        )
        self._thread.start()

    def _listen_loop(self):
        group = self._pull_group
        world_size = dist.get_world_size(group)
        other_ranks = [r for r in range(world_size) if r != self._local_rank]

        while True:
            try:
                # 1. Recv request on TAG_REQUEST: [source_rank, key_len, num_tokens]
                header = torch.zeros(3, dtype=torch.int64)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(header, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank = int(header[0].item())
                key_len = int(header[1].item())
                num_tokens = int(header[2].item())

                # 2. Recv the key string
                key_buf = torch.zeros(key_len, dtype=torch.uint8)
                dist.recv(key_buf, group=group, group_src=requesting_rank, tag=TAG_REQUEST)
                provider_string = key_buf.numpy().tobytes().decode("utf-8")

                # 3. Look up value in buffer (blocks until available)
                value = self.local_lookup(provider_string)

                # Normalize to list of tensors (handles both tensor and tuple)
                tensors = list(value) if isinstance(value, (tuple, list)) else [value]
                cpu_tensors = [t.detach().contiguous().cpu() for t in tensors]

                if num_tokens == 0:
                    # Legacy mode: send shape metadata then data.
                    shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
                    shape_meta[0] = len(cpu_tensors)
                    idx = 1
                    for t in cpu_tensors:
                        shape_meta[idx] = t.ndim
                        idx += 1
                        for s in t.shape:
                            shape_meta[idx] = s
                            idx += 1
                    dist.send(shape_meta, group=group, group_dst=requesting_rank, tag=TAG_RESPONSE)

                # 4. Send all tensor data concatenated as one flat buffer.
                flat = torch.cat([t.contiguous().view(-1) for t in cpu_tensors])
                dist.send(flat, group=group, group_dst=requesting_rank, tag=TAG_RESPONSE)

            except Exception:
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Consumer: pull tensor from remote rank
    # ------------------------------------------------------------------

    def pull_from_remote(
        self,
        source_rank: int,
        provider_string: str,
        num_tokens: int = 0,
    ):
        """Pull tensor(s) from a remote rank.

        Args:
            source_rank: PP rank that owns the module.
            provider_string: Key in the remote buffer.
            num_tokens: Number of tokens for this request (from scheduler,
                same on all PP ranks).  When > 0 and metadata is available,
                the recv buffer is pre-computed — no shape on the wire.
                When 0, falls back to legacy protocol with metadata.
        """
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group
        module_path = _provider_to_module_path(provider_string)
        meta = self._meta_map.get(module_path)

        # Decide protocol mode: optimized (pre-computed shapes) or legacy.
        use_precomputed = (
            num_tokens > 0
            and meta is not None
            and isinstance(meta, dict)
            and meta.get("static_suffixes")
        )
        header_num_tokens = num_tokens if use_precomputed else 0

        # 1. Send request on TAG_REQUEST: [my_rank, key_len, num_tokens]
        key_bytes = provider_string.encode("utf-8")
        header = torch.tensor(
            [self._local_rank, len(key_bytes), header_num_tokens],
            dtype=torch.int64,
        )
        dist.send(header, group=group, group_dst=source_rank, tag=TAG_REQUEST)

        # 2. Send the key
        key_tensor = torch.tensor(list(key_bytes), dtype=torch.uint8)
        dist.send(key_tensor, group=group, group_dst=source_rank, tag=TAG_REQUEST)

        if use_precomputed:
            return self._recv_precomputed(group, source_rank, meta, num_tokens)
        else:
            dtype = meta.get("dtype", torch.float32) if isinstance(meta, dict) else (meta if isinstance(meta, torch.dtype) else torch.float32)
            return self._recv_legacy(group, source_rank, dtype)

    def _recv_precomputed(self, group, source_rank, meta, num_tokens):
        """Recv flat data using pre-computed buffer size from metadata."""
        dtype = meta["dtype"]
        num_outputs = meta["num_outputs"]
        static_suffixes = meta["static_suffixes"]

        shapes = []
        total_numel = 0
        for suffix in static_suffixes:
            shape = (num_tokens, *suffix)
            numel = 1
            for s in shape:
                numel *= s
            shapes.append((shape, numel))
            total_numel += numel

        flat = torch.zeros(total_numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        results = []
        offset = 0
        for shape, numel in shapes:
            results.append(flat[offset:offset + numel].reshape(shape).to(self._device))
            offset += numel

        if num_outputs == 1:
            return results[0]
        return tuple(results)

    def _recv_legacy(self, group, source_rank, dtype):
        """Recv shape metadata then data (legacy fallback)."""
        shape_meta = torch.zeros(_META_SLOTS, dtype=torch.int64)
        dist.recv(shape_meta, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        num_elements = int(shape_meta[0].item())
        shapes = []
        idx = 1
        total_numel = 0
        for _ in range(num_elements):
            ndim = int(shape_meta[idx].item())
            idx += 1
            shape = [int(shape_meta[idx + j].item()) for j in range(ndim)]
            idx += ndim
            numel = 1
            for s in shape:
                numel *= s
            shapes.append((shape, numel))
            total_numel += numel

        flat = torch.zeros(total_numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        results = []
        offset = 0
        for shape, numel in shapes:
            results.append(flat[offset:offset + numel].reshape(shape).to(self._device))
            offset += numel

        if num_elements == 1:
            return results[0]
        return tuple(results)


def _provider_to_module_path(provider_string: str) -> str:
    """Strip '.output.iN' or '.input.iN' suffix to get the module path."""
    parts = provider_string.split(".")
    # Walk backwards to find and remove the access suffix
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("i") and parts[i][1:].isdigit():
            # Found iteration marker, the part before is "output" or "input"
            return ".".join(parts[: i - 1])
    return provider_string
