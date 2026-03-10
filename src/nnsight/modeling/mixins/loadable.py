from typing import Dict, Optional

import torch

from ..base import NNsight


class LoadableMixin(NNsight):
    """Mixin that adds model loading from an identifier (e.g. a repo ID or path).

    Extends :class:`NNsight` so the first argument can be either a
    ``torch.nn.Module`` (wrapped directly) or an identifier that is
    resolved by :meth:`_load`.

    Subclasses must implement :meth:`_load` to define how an identifier
    is converted into a ``torch.nn.Module``.

    Args:
        *args: If the first argument is a ``torch.nn.Module`` it is
            wrapped directly. Otherwise all arguments are forwarded
            to :meth:`_load`.
        rename (Optional[Dict[str, str]]): Module path aliases.
            See :class:`Envoy` for details.
        **kwargs: Forwarded to :meth:`_load` when loading from an
            identifier.
    """

    def __init__(self, *args, rename: Optional[Dict[str,str]] = None,**kwargs) -> None:

        if not isinstance(args[0], torch.nn.Module):

            model = self._load(*args, **kwargs)

        else:

            model = args[0]

        super().__init__(model, rename=rename)

    def _load(self, *args, **kwargs) -> torch.nn.Module:
        """Load and return a ``torch.nn.Module`` from the given identifier.

        Must be implemented by subclasses.

        Returns:
            torch.nn.Module: The loaded model.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """

        raise NotImplementedError()
