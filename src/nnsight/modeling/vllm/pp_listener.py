"""PP Listener — cross-rank tensor pull via a dedicated gloo process group.

Uses tags to separate request and response traffic on the same group:
  - TAG_REQUEST (0): consumer sends pull requests, producer's listener recvs
  - TAG_RESPONSE (1): producer's listener sends data back, consumer recvs

This avoids concurrent recv on the same (group, tag) from different threads.

Buffers store narrowed (per-mediator) tensors on GPU; moved to CPU at pull time.
Dtype is resolved locally from a shared metadata map built at model load time —
no dtype encoding on the wire.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

TAG_REQUEST = 0
TAG_RESPONSE = 1
_META_SLOTS = 32  # metadata buffer size: supports ~5 elements of 5D tensors


class PPListener:
    """Cross-rank tensor pull service.

    Producer (background listener thread): recvs on TAG_REQUEST,
    sends on TAG_RESPONSE.

    Consumer (main/mediator thread): sends on TAG_REQUEST,
    recvs on TAG_RESPONSE.

    No thread ever does concurrent recv on the same (group, tag).
    """

    def __init__(
        self,
        buffer: Dict[str, Any],
        condition: threading.Condition,
        pull_group: Optional[dist.ProcessGroup],
        local_rank: int,
        device: torch.device,
        dtype_map: Optional[Dict[str, torch.dtype]] = None,
    ):
        self._buffer = buffer
        self._condition = condition
        self._pull_group = pull_group
        self._local_rank = local_rank
        self._device = device
        self._dtype_map = dtype_map or {}
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
                # 1. Recv request on TAG_REQUEST: [source_rank, key_len]
                header = torch.zeros(2, dtype=torch.int64)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(header, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank = int(header[0].item())
                key_len = int(header[1].item())

                # 2. Recv the key string
                key_buf = torch.zeros(key_len, dtype=torch.uint8)
                dist.recv(key_buf, group=group, group_src=requesting_rank, tag=TAG_REQUEST)
                provider_string = key_buf.numpy().tobytes().decode("utf-8")

                # 3. Look up value in buffer (blocks until available)
                value = self.local_lookup(provider_string)

                # Normalize to list of tensors (handles both tensor and tuple)
                tensors = list(value) if isinstance(value, (tuple, list)) else [value]
                cpu_tensors = [t.detach().contiguous().cpu() for t in tensors]

                # 4. Send metadata: [num_elements, ndim0, *shape0, ndim1, *shape1, ...]
                # 32 slots supports up to ~5 elements of 5D tensors.
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

                # 5. Send all tensor data concatenated as one flat buffer
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
    ):
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group

        # 1. Send request on TAG_REQUEST: [my_rank, key_len]
        key_bytes = provider_string.encode("utf-8")
        header = torch.tensor(
            [self._local_rank, len(key_bytes)], dtype=torch.int64
        )
        dist.send(header, group=group, group_dst=source_rank, tag=TAG_REQUEST)

        # 2. Send the key
        key_tensor = torch.tensor(list(key_bytes), dtype=torch.uint8)
        dist.send(key_tensor, group=group, group_dst=source_rank, tag=TAG_REQUEST)

        # 3. Resolve dtype locally from the shared metadata map.
        module_path = _provider_to_module_path(provider_string)
        dtype = self._dtype_map.get(module_path, torch.float32)

        # 4. Recv metadata: [num_elements, ndim0, *shape0, ndim1, *shape1, ...]
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

        # 5. Recv all tensor data as one flat buffer
        flat = torch.zeros(total_numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        # 6. Split, reshape, and move to GPU
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
