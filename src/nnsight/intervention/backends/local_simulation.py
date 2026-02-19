"""Local simulation backend for testing serialization.

This backend serializes and deserializes traces locally, simulating
remote execution on NDIF. It validates that all code is properly
captured for server-side execution by temporarily blocking access
to non-server modules during deserialization.
"""

import sys
from typing import Any

from ..tracing.globals import Globals
from ..tracing.util import wrap_exception
from .base import Backend
from .remote import pull_env


# Modules available on NDIF servers (same as remote.py)
SERVER_MODULES = {
    "torch",
    "numpy",
    "transformers",
    "accelerate",
    "einops",
    "collections",
    "itertools",
    "functools",
    "operator",
    "math",
    "random",
    "json",
    "re",
    "typing",
    "dataclasses",
    "abc",
}


class LocalSimulationBackend(Backend):
    """
    Backend that simulates remote execution by serializing/deserializing locally.

    Validates that serialization captures all dependencies by blocking access
    to user modules during deserialization - just like on NDIF servers.
    """

    def __init__(self, model: Any, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

    def __call__(self, tracer):
        from nnsight.schema.request import RequestModel

        # Step 1: Compile the trace (standard backend behavior)
        interventions = super().__call__(tracer)

        # Step 2: Serialize
        request = RequestModel(interventions=interventions, tracer=tracer)
        pull_env()
        payload = request.serialize(compress=False)

        if self.verbose:
            print(f"[LocalSimulation] Payload: {len(payload)} bytes")

        # Step 3: Deserialize with user modules blocked
        # This simulates NDIF where user modules don't exist
        persistent_objects = self.model._remoteable_persistent_objects()

        blocked = self._block_user_modules()
        try:
            restored = RequestModel.deserialize(
                payload, persistent_objects, compress=False
            )
        finally:
            self._restore_modules(blocked)

        # Step 4: Execute the restored function
        try:
            Globals.enter()
            return tracer.execute(restored.interventions)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()

    def _block_user_modules(self):
        """Block non-server modules from sys.modules and sys.path."""
        blocked = {"modules": {}, "paths": []}

        # Block paths that could contain user modules
        # Keep only: site-packages, stdlib paths, and src/ for nnsight
        for path in list(sys.path):
            # Keep package installation paths
            if "site-packages" in path:
                continue
            if "lib/python" in path:
                continue
            # Keep nnsight src path
            if path.endswith("/src") or "/src/nnsight" in path:
                continue
            # Block everything else (tests/, user directories, etc.)
            blocked["paths"].append(path)
            sys.path.remove(path)

        # Block non-server modules
        for name in list(sys.modules.keys()):
            root = name.split(".")[0]
            if root in SERVER_MODULES:
                continue
            if root in sys.stdlib_module_names:
                continue
            if "nnsight" in name:
                continue
            blocked["modules"][name] = sys.modules.pop(name)

        if self.verbose:
            print(
                f"[LocalSimulation] Blocked {len(blocked['modules'])} modules, {len(blocked['paths'])} paths"
            )

        return blocked

    def _restore_modules(self, blocked):
        """Restore previously blocked modules and paths."""
        sys.modules.update(blocked["modules"])
        sys.path.extend(blocked["paths"])
