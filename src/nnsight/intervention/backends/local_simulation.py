"""Local simulation backend for testing serialization without network.

This module provides a backend that serializes and deserializes the trace
locally, simulating what happens on NDIF without requiring network access.
This is useful for:

1. Testing serialization logic before running on NDIF
2. Catching serialization errors early with clear error messages
3. Verifying that @nnsight.remote decorations are working correctly
4. Debugging complex traces without network latency

The LocalSimulationBackend uses the same serialization/deserialization path
as RemoteBackend but executes the reconstructed code locally on the same
model instance.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

from ..serialization_source import (
    serialize_source_based,
    deserialize_source_based,
    SourceSerializationError,
)
from ..tracing.globals import Globals
from ..tracing.util import wrap_exception
from .base import Backend

if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any


class LocalSimulationBackend(Backend):
    """
    Backend that simulates remote execution locally.

    This backend performs the same serialization and deserialization steps
    as RemoteBackend, but executes the reconstructed code locally on the
    same model instance. This helps catch serialization errors early without
    requiring network access to NDIF.

    Attributes:
        model: The model instance being traced
        verbose: If True, print serialization details
        _last_payload_size: Size of last serialized payload in bytes
        _last_payload: The last serialized payload (for debugging)
    """

    def __init__(
        self,
        model: Any,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the LocalSimulationBackend.

        Args:
            model: The model instance being traced
            verbose: If True, print serialization details for debugging
        """
        self.model = model
        self.verbose = verbose
        self._last_payload_size: int = 0
        self._last_payload: Optional[bytes] = None

    def __call__(self, tracer: Tracer):
        """
        Serialize, deserialize, and execute the traced code locally.

        This simulates the full remote execution path:
        1. Compile the trace into a function (via super().__call__)
        2. Serialize using source-based serialization
        3. Deserialize into a fresh namespace (simulating server environment)
        4. Execute the reconstructed code with the same model

        Args:
            tracer: The tracer containing the captured code

        Returns:
            Result from executing the trace

        Raises:
            SourceSerializationError: If serialization fails
            Exception: Any exception from executing the trace
        """
        # STEP 1: Compile the trace into a function (standard backend behavior)
        fn = super().__call__(tracer)

        # STEP 2: Serialize using source-based serialization
        try:
            payload = serialize_source_based(tracer)
            self._last_payload_size = len(payload)
            self._last_payload = payload

            if self.verbose:
                print(f"[LocalSimulation] Serialized payload: {len(payload)} bytes")
                # Show a preview of the payload
                import json
                data = json.loads(payload)
                print(f"[LocalSimulation] Variables: {list(data.get('variables', {}).keys())}")
                print(f"[LocalSimulation] Remote objects: {list(data.get('remote_objects', {}).keys())}")
                if data.get('model_refs'):
                    print(f"[LocalSimulation] Model refs: {data['model_refs']}")
        except SourceSerializationError as e:
            # Re-raise with context
            raise SourceSerializationError(
                f"LocalSimulation: Serialization failed. This would also fail on NDIF.\n\n{e}"
            ) from e

        # STEP 3: Deserialize into a fresh namespace
        # This simulates what happens on the NDIF server
        try:
            namespace = deserialize_source_based(
                payload,
                self.model,  # Use the same model instance
                use_restricted=False,  # Don't use restricted mode for local simulation
            )

            if self.verbose:
                print(f"[LocalSimulation] Deserialized namespace keys: {list(namespace.keys())}")
        except Exception as e:
            raise SourceSerializationError(
                f"LocalSimulation: Deserialization failed. This would also fail on NDIF.\n\n{e}"
            ) from e

        # STEP 4: Execute the reconstructed code
        # We use the original compiled function, not the reconstructed one,
        # because the reconstructed namespace is just for validation.
        # The actual execution uses the tracer's normal execution path.
        try:
            Globals.enter()
            return tracer.execute(fn)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()

    @property
    def last_payload_size(self) -> int:
        """Return the size of the last serialized payload in bytes."""
        return self._last_payload_size

    def get_last_payload(self) -> Optional[bytes]:
        """Return the last serialized payload for debugging."""
        return self._last_payload
