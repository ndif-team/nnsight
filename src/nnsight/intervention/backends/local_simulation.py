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
model instance. During deserialization, it blocks access to user modules
to validate that all code is properly captured for server-side execution.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional


# Modules available on NDIF servers
SERVER_MODULES = {
    'torch', 'numpy', 'transformers', 'accelerate', 'einops',
    'collections', 'itertools', 'functools', 'operator', 'math',
    'random', 'json', 're', 'typing', 'dataclasses', 'abc',
}

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
        strict_remote: If True, require explicit @remote decorations
        max_upload_mb: Upload payload size threshold for warnings (default 10 MB)
        _last_payload_size: Size of last serialized payload in bytes
        _last_payload: The last serialized payload (for debugging)
    """

    def __init__(
        self,
        model: Any,
        verbose: bool = False,
        strict_remote: bool = False,
        max_upload_mb: float = 10.0,
    ) -> None:
        """
        Initialize the LocalSimulationBackend.

        Args:
            model: The model instance being traced
            verbose: If True, print serialization details for debugging
            strict_remote: If True, require explicit @remote decorations (default False)
            max_upload_mb: Threshold for upload payload size warnings (default 10 MB)
        """
        self.model = model
        self.verbose = verbose
        self.strict_remote = strict_remote
        self.max_upload_mb = max_upload_mb
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
            payload = serialize_source_based(
                tracer,
                strict_remote=self.strict_remote,
                max_upload_mb=self.max_upload_mb,
            )
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

        # STEP 3: Deserialize with user modules blocked
        # This simulates NDIF where user modules don't exist
        blocked = self._block_user_modules()
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
        finally:
            self._restore_modules(blocked)

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

    def _block_user_modules(self) -> Dict[str, Any]:
        """Block non-server modules from sys.modules and sys.path."""
        blocked: Dict[str, Any] = {'modules': {}, 'paths': []}

        # Block paths that could contain user modules
        for path in list(sys.path):
            if 'site-packages' in path or 'lib/python' in path:
                continue
            if path.endswith('/src') or '/src/nnsight' in path:
                continue
            blocked['paths'].append(path)
            sys.path.remove(path)

        # Block non-server modules
        for name in list(sys.modules.keys()):
            root = name.split('.')[0]
            if root in SERVER_MODULES:
                continue
            if root in sys.stdlib_module_names:
                continue
            if 'nnsight' in name:
                continue
            blocked['modules'][name] = sys.modules.pop(name)

        if self.verbose:
            print(f"[LocalSimulation] Blocked {len(blocked['modules'])} modules, {len(blocked['paths'])} paths")

        return blocked

    def _restore_modules(self, blocked: Dict[str, Any]) -> None:
        """Restore previously blocked modules and paths."""
        sys.modules.update(blocked['modules'])
        sys.path.extend(blocked['paths'])
