"""PP Listener — cross-rank tensor pull via a dedicated gloo process group.

Uses tags to separate request and response traffic on the same group:
  - TAG_REQUEST (0): consumer sends pull requests, producer's listener recvs
  - TAG_RESPONSE (1): producer's listener sends data back, consumer recvs

This avoids concurrent recv on the same (group, tag) from different threads.

Buffers live on GPU; tensors are moved to CPU only at pull time.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

TAG_REQUEST = 0
TAG_RESPONSE = 1


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
    ):
        self._buffer = buffer
        self._condition = condition
        self._pull_group = pull_group
        self._local_rank = local_rank
        self._device = device
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

                # 3. Look up tensor in GPU buffer (blocks until available)
                tensor = self.local_lookup(provider_string)

                # 4. Move to CPU for gloo transfer
                cpu_tensor = tensor.detach().contiguous().cpu()

                # 5. Send metadata on TAG_RESPONSE: [ndim, dtype_code, *shape]
                meta = torch.tensor(
                    [cpu_tensor.ndim, _dtype_to_code(cpu_tensor.dtype)]
                    + list(cpu_tensor.shape),
                    dtype=torch.int64,
                )
                meta_header = torch.tensor([meta.numel()], dtype=torch.int64)
                dist.send(meta_header, group=group, group_dst=requesting_rank, tag=TAG_RESPONSE)
                dist.send(meta, group=group, group_dst=requesting_rank, tag=TAG_RESPONSE)

                # 6. Send flattened tensor data on TAG_RESPONSE
                dist.send(
                    cpu_tensor.contiguous().view(-1),
                    group=group,
                    group_dst=requesting_rank,
                    tag=TAG_RESPONSE,
                )

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
    ) -> torch.Tensor:
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

        # 3. Recv metadata on TAG_RESPONSE
        meta_header = torch.zeros(1, dtype=torch.int64)
        dist.recv(meta_header, group=group, group_src=source_rank, tag=TAG_RESPONSE)
        meta = torch.zeros(meta_header.item(), dtype=torch.int64)
        dist.recv(meta, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        meta_list = meta.tolist()
        ndim = int(meta_list[0])
        dtype = _code_to_dtype(int(meta_list[1]))
        shape = [int(x) for x in meta_list[2 : 2 + ndim]]

        # 4. Recv flattened tensor on TAG_RESPONSE
        numel = 1
        for s in shape:
            numel *= s
        flat = torch.zeros(numel, dtype=dtype)
        dist.recv(flat, group=group, group_src=source_rank, tag=TAG_RESPONSE)

        # 5. Reshape and move to GPU
        return flat.reshape(shape).to(self._device)


# ------------------------------------------------------------------
# Dtype encoding
# ------------------------------------------------------------------

_DTYPE_MAP = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.bfloat16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
}
_CODE_MAP = {v: k for k, v in _DTYPE_MAP.items()}


def _dtype_to_code(dtype: torch.dtype) -> int:
    return _DTYPE_MAP.get(dtype, 1)


def _code_to_dtype(code: int) -> torch.dtype:
    return _CODE_MAP.get(code, torch.float32)
