"""Client backend for nnsight-vllm-serve.

Sends compiled traces to a local nnsight-serve instance and retrieves saves.

Design choices:
- Reuses RequestModel serialization (same format as NDIF remote backend).
- Synchronous HTTP by default: POSTs to /v1/nnsight/generate and blocks
  until done, then pushes saves into the caller's frame (same as local execution).
- Non-blocking mode (blocking=False): fires the HTTP request in a background
  thread and returns immediately. The tracer is returned from the ``with``
  block with a ``.collect(timeout=None)`` method. Calling ``.result()``
  blocks until the response arrives and returns a dict of saved values.
  Frame injection is NOT possible in non-blocking mode because the caller's
  frame has moved on by the time the response arrives.
- Response uses torch.save format (handles tensors correctly).
- The client does NOT need a GPU or dispatched model. Only the meta model
  is needed for envoy proxies.
"""

from __future__ import annotations

import io
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx
import torch

from ..tracing.util import wrap_exception
from ...schema.request import RequestModel
from .base import Backend

if TYPE_CHECKING:
    from ..tracing.tracer import Tracer

# Shared thread pool for non-blocking requests.
# Threads are cheap here — the work is IO-bound (HTTP round-trip).
_pool = ThreadPoolExecutor(max_workers=16)


class LocalServeBackend(Backend):
    """Backend that sends compiled traces to a local nnsight-serve instance.

    Args:
        model: The NNsight model wrapper (meta model for envoy proxies).
        host: URL of the nnsight-serve instance.
        blocking: If True (default), block until the server responds and
            push saves into the caller's frame (identical to local execution).
            If False, fire the request in a background thread. The tracer
            gets a ``.collect(timeout=None)`` method that blocks and returns
            the saves dict.

    Usage (blocking, default)::

        with model.trace("prompt", serve=url):
            out = model.logits.output.save()
        print(out)  # tensor, available immediately

    Usage (non-blocking)::

        with model.trace("prompt1", serve=url, blocking=False) as t1:
            out1 = model.logits.output.save()
        with model.trace("prompt2", serve=url, blocking=False) as t2:
            out2 = model.logits.output.save()

        # Both in-flight concurrently inside vLLM's engine.
        saves1 = t1.collect()  # blocks, returns {"out1": tensor}
        saves2 = t2.collect()  # blocks, returns {"out2": tensor}
        out1, out2 = saves1["out1"], saves2["out2"]
    """

    CONNECT_TIMEOUT: float = 10.0
    READ_TIMEOUT: float = 600.0  # 10 min for large models.

    def __init__(self, model: Any, host: str, blocking: bool = True):
        self.model = model
        self.host = host.rstrip("/")
        self.blocking = blocking

    def __call__(self, tracer: Optional["Tracer"] = None):
        if tracer is None:
            return

        # Compile trace → intervention function.
        interventions = Backend.__call__(self, tracer)

        # Serialize eagerly — the tracer state is ephemeral.
        try:
            compress = False
            data = RequestModel(
                interventions=interventions, tracer=tracer
            ).serialize(compress)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None

        if self.blocking:
            saves = self._send(data, compress)
            tracer.push(saves)
        else:
            future = _pool.submit(self._send, data, compress)
            tracer._serve_future = future

    def _send(self, data: bytes, compress: bool) -> Dict[str, Any]:
        """Send serialized request, return saves dict."""
        headers = {
            "Content-Type": "application/octet-stream",
            "nnsight-compress": str(compress),
        }

        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.host}/v1/nnsight/generate",
                content=data,
                headers=headers,
            )

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.reason_phrase)
            except Exception:
                detail = response.reason_phrase
            raise ConnectionError(
                f"nnsight-serve returned {response.status_code}: {detail}"
            )

        result = torch.load(
            io.BytesIO(response.content),
            map_location="cpu",
            weights_only=False,
        )
        return result.get("saves", {})
