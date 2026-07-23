"""Local-simulation backend: exercise the remote path without a server.

``model.trace(..., remote="local")`` runs the trace here instead of shipping it
to NDIF. It serializes the trace exactly as :class:`~nnsight.intervention.backends.remote.RemoteBackend`
would, then deserializes it *with local (non-installed) modules hidden* — mimicking
a server where the user's own source files don't exist — and runs the deserialized
block locally against the real model.

If the block references a local function/class that wasn't shipped by value, the
deserialize step raises (``ModuleNotFoundError``) just as it would on the server,
so a passing ``remote="local"`` run is strong evidence a real remote run will work.
Results land back in the caller's frame like an ordinary local trace.
"""

from __future__ import annotations

import linecache
import sys
from typing import Any

from ...tracing.backend import Backend

# Roots always present on an NDIF server, so they are never hidden during the
# simulated deserialize (mirrors the server's installed environment).
_SERVER_MODULES = {
    "torch",
    "numpy",
    "transformers",
    "accelerate",
    "diffusers",
    "einops",
    "peft",
    "nnsight",
}


class LocalSimulationBackend(Backend):
    """Serialize/deserialize a trace locally (user modules hidden), then run it."""

    def __init__(self, model: Any, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

    def __call__(self, tracer: Any) -> None:
        from ...ndif import pull_env
        from ...schema.request import RequestModel

        # Register local modules for by-value pickling (same as the remote path).
        pull_env()
        blob = RequestModel.serialize(tracer, compress=False)
        if self.verbose:
            print(f"[local] payload: {len(blob):,} bytes")

        persistent = self.model._remoteable_persistent_objects()

        # deserialize() registers the *block* source in linecache under the user's
        # filename. On a real server that's a fresh process; in-process it would
        # clobber the true file's entry and corrupt later source lookups (e.g.
        # serializing another local function), so snapshot and restore it.
        filename = tracer.info.frame.f_code.co_filename
        saved_line = linecache.cache.get(filename)

        # Deserialize with local modules hidden — if the block's code wasn't fully
        # shipped by value, this raises exactly as the server would.
        hidden = self._hide_local_modules()
        try:
            restored = RequestModel.deserialize(blob, persistent, compress=False)
        finally:
            self._restore(hidden)
            if saved_line is not None:
                linecache.cache[filename] = saved_line
            else:
                linecache.cache.pop(filename, None)  # let it re-read from disk

        # Run the deserialized block through the original tracer so results push
        # back into the caller's frame (we're in-process; its globals/locals match
        # what was shipped and validated above).
        tracer.info.code = restored.info.code
        tracer.execute(tracer.info.code)

    def _hide_local_modules(self) -> dict:
        """Remove non-installed modules/paths so deserialize can't reach them."""
        hidden: dict = {"modules": {}, "paths": []}

        for path in list(sys.path):
            if "site-packages" in path or "lib/python" in path:
                continue
            if path.endswith("/src") or "/src/nnsight" in path:
                continue
            hidden["paths"].append(path)
            sys.path.remove(path)

        for name in list(sys.modules):
            root = name.split(".")[0]
            if root in _SERVER_MODULES or root in sys.stdlib_module_names:
                continue
            if "nnsight" in name:
                continue
            hidden["modules"][name] = sys.modules.pop(name)

        if self.verbose:
            print(
                f"[local] hid {len(hidden['modules'])} modules, "
                f"{len(hidden['paths'])} paths"
            )
        return hidden

    def _restore(self, hidden: dict) -> None:
        sys.modules.update(hidden["modules"])
        sys.path.extend(hidden["paths"])
