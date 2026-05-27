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

from .pp import resolve_meta as _resolve_meta

TAG_REQUEST = 0
TAG_RESPONSE = 1
_META_SLOTS = 32  # legacy metadata buffer size

# Separator between req_id and provider in the wire-encoded key.
# Format: ``"{req_id}|{provider}"`` (null-terminated).  req_id may be
# empty for requests that predate the composite-key discipline — the
# listener then falls back to the string-only provider key (legacy).
_KEY_SEP = "|"


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
        # Set by ``stop()`` to break the listen loop cleanly. Also
        # checked after every caught exception so a torn-down ``dist``
        # context (e.g. when the worker process is being shut down)
        # doesn't busy-loop the listener at 100% CPU.
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def clear_buffer(self, req_ids=None):
        """Drop buffer entries and wake pending lookups.

        Args:
            req_ids: Optional iterable of request ids.  When supplied,
                only keys whose 2-tuple tail equals one of these ids
                are removed — in-flight requests' entries stay intact.
                When ``None``, falls back to the legacy blanket clear
                (all keys removed).

        Scoped clear is the Bug B companion fix: with composite keys
        ``(provider, req_id)``, a blanket clear at ANY request's
        completion would also wipe concurrent in-flight requests'
        slices and send their still-pending cross-rank pulls into a
        30 s timeout.

        We notify_all so waiters whose key was removed stop spinning
        immediately; waiters on untouched keys see their key still
        present after the wake and continue normally.
        """
        with self._condition:
            if req_ids is None:
                self._buffer.clear()
            else:
                id_set = set(req_ids)
                to_remove = [
                    k for k in self._buffer
                    if isinstance(k, tuple) and len(k) == 2 and k[1] in id_set
                ]
                for k in to_remove:
                    del self._buffer[k]
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Local buffer lookup (blocks until value available)
    # ------------------------------------------------------------------

    def local_lookup(
        self,
        key,
        timeout: Optional[float] = 60.0,
    ) -> torch.Tensor:
        """Block until ``key`` is in the buffer, then return its value.

        ``key`` is whatever the producer wrote under — for Bug B the
        producer writes ``(provider, req_id)`` composite tuples, so
        ``key`` here is a 2-tuple.  String keys are still supported for
        legacy callers and unit tests.
        """
        with self._condition:
            while key not in self._buffer:
                if not self._condition.wait(timeout=timeout):
                    raise TimeoutError(
                        f"PPListener: timed out waiting for {key!r}"
                    )
            return self._buffer[key]

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

    def stop(self):
        """Signal the listener thread to exit on its next loop iteration."""
        self._stop_event.set()

    def _listen_loop(self):
        group = self._pull_group
        world_size = dist.get_world_size(group)
        other_ranks = [r for r in range(world_size) if r != self._local_rank]

        while not self._stop_event.is_set():
            try:
                # 1. Recv request on TAG_REQUEST: [source_rank, key_len, num_tokens]
                header = torch.zeros(3, dtype=torch.int64)
                src = other_ranks[0] if len(other_ranks) == 1 else None
                dist.recv(header, group=group, group_src=src, tag=TAG_REQUEST)

                requesting_rank = int(header[0].item())
                key_len = int(header[1].item())
                num_tokens = int(header[2].item())

                # 2. Recv the key string.  Wire encoding is
                #    ``"{req_id}|{provider}"`` for Bug B composite keys.
                #    A missing separator (legacy callers / tests) is
                #    treated as a plain provider string with req_id=None.
                key_buf = torch.zeros(key_len, dtype=torch.uint8)
                dist.recv(key_buf, group=group, group_src=requesting_rank, tag=TAG_REQUEST)
                encoded = key_buf.numpy().tobytes().decode("utf-8")

                if _KEY_SEP in encoded:
                    req_id_str, provider_string = encoded.split(_KEY_SEP, 1)
                    lookup_key = (provider_string, req_id_str or None)
                else:
                    provider_string = encoded
                    lookup_key = encoded

                # 3. Look up value in buffer (blocks until available)
                value = self.local_lookup(lookup_key)

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
                # If the dist context is gone (worker tearing down, peer
                # crashed) every ``dist.recv`` here would raise instantly
                # and the loop would burn CPU at 100% printing tracebacks
                # forever. Detect that case and exit cleanly. Genuine
                # transient errors still get logged and retried after a
                # short backoff so the listener stays responsive without
                # spinning.
                if not dist.is_initialized() or self._stop_event.is_set():
                    return
                import traceback
                traceback.print_exc()
                # Wait with timeout so ``stop()`` from the main thread
                # wakes us promptly; expire-then-retry on its own keeps
                # the listener alive through transient failures.
                if self._stop_event.wait(timeout=0.5):
                    return

    # ------------------------------------------------------------------
    # Consumer: pull tensor from remote rank
    # ------------------------------------------------------------------

    def pull_from_remote(
        self,
        source_rank: int,
        provider_string: str,
        num_tokens: int = 0,
        req_id: Optional[str] = None,
    ):
        """Pull tensor(s) from a remote rank.

        Args:
            source_rank: PP rank that owns the module.
            provider_string: Module-level provider (``"…output.iN"``).
            num_tokens: Number of tokens for this request (from scheduler,
                same on all PP ranks).  When > 0 and metadata is available,
                the recv buffer is pre-computed — no shape on the wire.
                When 0, falls back to legacy protocol with metadata.
            req_id: Bug B fix — vLLM request id.  When non-None, the wire
                key encodes ``"{req_id}|{provider}"`` and the producer
                looks up the composite ``(provider, req_id)`` tuple so
                concurrent mediators can't deliver each other's slices.
                ``None`` falls back to the legacy string-only key (used
                by unit tests and any non-vLLM caller).
        """
        if self._pull_group is None:
            raise RuntimeError("No pull_group configured for cross-rank pull")

        group = self._pull_group
        module_path = _provider_to_module_path(provider_string)
        meta = _resolve_meta(self._meta_map, module_path)

        # Decide protocol mode: optimized (pre-computed shapes) or legacy.
        use_precomputed = self._should_use_precomputed(meta, num_tokens)
        header_num_tokens = num_tokens if use_precomputed else 0

        # Wire-encode the key with req_id if available (Bug B composite
        # key); producer parses on receipt.
        if req_id is not None:
            wire_key = f"{req_id}{_KEY_SEP}{provider_string}"
        else:
            wire_key = provider_string

        # 1. Send request on TAG_REQUEST: [my_rank, key_len, num_tokens]
        key_bytes = wire_key.encode("utf-8")
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
            return self._recv_legacy(
                group, source_rank, dtype,
                module_path=module_path, meta=meta,
            )

    def _recv_precomputed(self, group, source_rank, meta, num_tokens):
        """Recv flat data using the learned module output shape — no shape
        on the wire.

        The producer's tensor is ``[num_tokens, *features]`` (vLLM's flat
        token-major layout). We keep the learned per-feature shape and
        substitute this request's ``num_tokens`` for the leading dim, since
        the token count is the only part that varies between requests.
        """
        dtype = meta["dtype"]
        num_outputs = meta["num_outputs"]
        module_shapes = meta["module_shapes"]

        shapes = []
        total_numel = 0
        for learned_shape in module_shapes:
            shape = (num_tokens, *learned_shape[1:])
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

    def _recv_legacy(self, group, source_rank, dtype,
                     module_path=None, meta=None):
        """Recv shape metadata then data (legacy fallback).

        Also learns the module's output shape from the wire so subsequent
        pulls of the same module use the precomputed path — see
        :meth:`_cache_module_shapes`.
        """
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

        self._cache_module_shapes(module_path, meta, shapes, dtype)

        if num_elements == 1:
            return results[0]
        return tuple(results)

    @staticmethod
    def _should_use_precomputed(meta, num_tokens) -> bool:
        """Whether a pull can skip shape-on-wire (precomputed recv buffer).

        Requires this request's token count (to size the buffer) and a
        metadata entry whose ``module_shapes`` have been learned. Shapes
        start empty at init and are filled by :meth:`_cache_module_shapes`
        on the first legacy pull, so the first pull of each module is
        legacy and subsequent pulls are precomputed.
        """
        return bool(
            num_tokens > 0
            and isinstance(meta, dict)
            and meta.get("module_shapes")
        )

    def _cache_module_shapes(self, module_path, meta, shapes, dtype):
        """Learn a module's output shape(s) from a real legacy response.

        Records the per-element output shape so the next pull of this
        module takes the precomputed path. The token (leading) dimension
        is overridden per request at recv time, so what matters here is
        the trailing feature shape — we keep the whole observed shape and
        let :meth:`_recv_precomputed` substitute the live token count.
        """
        if module_path is None:
            return

        module_shapes = [tuple(shape) for shape, _ in shapes]

        # Mutate the resolved entry in place when it exists (keeps the
        # allgather-provided dtype and is agnostic to the prefix-tolerant
        # key under which it was found); otherwise create a fresh entry.
        if isinstance(meta, dict):
            entry = meta
        else:
            entry = {"dtype": dtype}
            self._meta_map[module_path] = entry
        entry["module_shapes"] = module_shapes
        entry["num_outputs"] = len(shapes)


def _provider_to_module_path(provider_string: str) -> str:
    """Strip '.output.iN' or '.input.iN' suffix to get the module path."""
    parts = provider_string.split(".")
    # Walk backwards to find and remove the access suffix
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("i") and parts[i][1:].isdigit():
            # Found iteration marker, the part before is "output" or "input"
            return ".".join(parts[: i - 1])
    return provider_string
