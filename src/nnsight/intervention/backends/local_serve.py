"""Client backend for nnsight-vllm-serve.

Sends compiled traces to a local nnsight-serve instance and retrieves saves.

Design choices:
- Reuses RequestModel serialization (same format as NDIF remote backend).
- Synchronous HTTP: POSTs to /v1/nnsight/generate and blocks until done.
  No WebSocket or polling — the server processes the request and returns
  the result in the same HTTP response.
- Response uses torch.save format (handles tensors correctly, avoids
  arbitrary pickle deserialization by using weights_only where possible).
- The client does NOT need a GPU or dispatched model. Only the meta model
  is needed for envoy proxies (so user code like model.layers[5].output
  resolves correctly during tracing).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Optional

import httpx
import torch

from ..tracing.util import wrap_exception
from ...schema.request import RequestModel
from .base import Backend

if TYPE_CHECKING:
    from ..tracing.tracer import Tracer


class LocalServeBackend(Backend):
    """Backend that sends compiled traces to a local nnsight-serve instance."""

    CONNECT_TIMEOUT: float = 10.0
    READ_TIMEOUT: float = 600.0  # 10 min for large models.

    def __init__(self, model: Any, host: str):
        self.model = model
        self.host = host.rstrip("/")

    def __call__(self, tracer: Optional["Tracer"] = None):
        if tracer is not None:
            self._compile_and_send(tracer)

    def _compile_and_send(self, tracer: "Tracer"):
        """Compile traced code, serialize as RequestModel, POST to server."""
        # Compile trace → intervention function (same as RemoteBackend.request).
        interventions = Backend.__call__(self, tracer)

        try:
            compress = False
            data = RequestModel(
                interventions=interventions, tracer=tracer
            ).serialize(compress)

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

            # Deserialize response (torch.save format).
            result = torch.load(
                io.BytesIO(response.content),
                map_location="cpu",
                weights_only=False,
            )
            saves = result.get("saves", {})

            # Push saves to the tracer's original frame
            # (same as RemoteBackend.blocking_request line 875).
            tracer.push(saves)

        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
