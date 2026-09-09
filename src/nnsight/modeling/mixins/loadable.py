from __future__ import annotations

import inspect
from typing import Any

import torch

from ...intervention.envoy import Envoy

#: The keyword arguments that belong to [`Envoy`][nnsight.intervention.envoy.Envoy], derived from its
#: signature so the split below can never drift from what Envoy accepts.
_ENVOY_KWARGS = frozenset(
    name
    for name in inspect.signature(Envoy.__init__).parameters
    if name not in ("self", "module")
)


def split_envoy_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Split ``kwargs`` into ``(envoy_kwargs, load_kwargs)``.

    Envoy's own parameters (``interleaver``, ``rename``, ...) route to
    ``Envoy.__init__``; everything else is a load argument for
    `Loadable._load`.
    """
    envoy_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _ENVOY_KWARGS}
    return envoy_kwargs, kwargs


class Loadable(Envoy):
    """An [`Envoy`][nnsight.intervention.envoy.Envoy] that loads its own module.

    The constructor accepts either a ready ``torch.nn.Module`` or load
    arguments (e.g. a repo id) and dispatches accordingly: a ready module goes
    to `_wrap` (base: build the tree over it as-is), anything else to `_load`
    (build the module from the arguments). Subclasses override whichever paths
    they support; an override never receives the other path's argument kind.

    Keyword arguments split by `split_envoy_kwargs`: Envoy's parameters go to
    ``Envoy.__init__``, the rest to the load path.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        envoy_kwargs, load_kwargs = split_envoy_kwargs(kwargs)
        if args and isinstance(args[0], torch.nn.Module):
            model = self._wrap(args[0], *args[1:], **load_kwargs)
        else:
            model = self._load(*args, **load_kwargs)
        super().__init__(model, **envoy_kwargs)

    def _wrap(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """The module to build the tree over, given a ready module.

        Base: the module itself. A subclass overrides this when a pre-loaded
        module needs surrounding machinery (e.g.
        [`TransformersModel`][nnsight.modeling.transformers.TransformersModel] builds a
        ``transformers.pipeline`` around it).
        """
        return module

    def _load(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """Build and return the module from load arguments (e.g. a repo id)."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot build a module from {args!r}; "
            f"pass a torch.nn.Module or use a subclass that implements _load."
        )
