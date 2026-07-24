from __future__ import annotations

from typing import Any

import torch

from ...intervention.envoy import Envoy


class Loadable(Envoy):
    """An [`Envoy`][nnsight.intervention.envoy.Envoy] that loads its own module.

    Every construction goes through `_load`, which returns the module to wrap.
    The base default returns a ready ``torch.nn.Module`` as-is (so ``Loadable(mod)``
    wraps it directly) and otherwise isn't implemented; subclasses override it to
    build from load arguments and to decide what a pre-loaded module means for them
    (e.g. [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] wraps one in a
    ``transformers.pipeline``). The ``rename`` spec is an Envoy concern, so it is
    kept out of the load path.
    """

    def __init__(
        self,
        *args: Any,
        rename: Any = None,
        envoys: Any = None,
        interleaver: Any = None,
        **kwargs: Any,
    ) -> None:
        # rename/envoys/interleaver are Envoy concerns, not load args — keep
        # them out of _load.
        model = self._load(*args, **kwargs)
        super().__init__(model, rename=rename, envoys=envoys, interleaver=interleaver)

    def _load(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """Build and return the module to wrap.

        Base default: a ready ``torch.nn.Module`` is returned as-is; anything else
        is not implemented. Subclasses override to load the real module from the
        given arguments (e.g. a repo id).
        """
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        raise NotImplementedError
