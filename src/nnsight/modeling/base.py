from typing import Dict, Optional, Type, Union

import torch

from ..intervention.envoy import Envoy


class NNsight(Envoy):
    """Root :class:`Envoy` that wraps a full ``torch.nn.Module`` tree.

    ``NNsight`` is the **base / root envoy** — the top of an envoy tree
    that mirrors a PyTorch model's module hierarchy. Constructing one
    recursively wraps every child ``torch.nn.Module`` in its own
    :class:`Envoy` (or a user-specified subclass, see ``envoys`` below),
    giving each module NNsight's intervention capabilities: access to and
    modification of intermediate activations during execution via the
    tracing context (``.trace`` / ``.generate`` / ``.scan`` / ``.edit`` /
    ``.session``).

    This is the simplest entry point for wrapping arbitrary PyTorch
    models. Higher-level wrappers (``LanguageModel``, ``VLLM``,
    ``DiffusionModel``, …) are themselves :class:`NNsight` subclasses and
    serve as specialized root envoys — they add model-specific loading,
    tokenization, and batching on top of the same root-envoy behavior.

    Example::

        import torch
        from nnsight import NNsight

        net = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 2),
        )
        model = NNsight(net)  # root envoy; children are auto-wrapped

        with model.trace(torch.rand(1, 5)):
            hidden = model[0].output.save()

    Customizing descendant Envoy classes
    ------------------------------------

    As the root envoy, ``NNsight`` is also where the ``envoys``
    configuration is introduced for the whole tree. The value is
    forwarded to :class:`Envoy` and propagated to every descendant. It
    can be:

    - ``None`` (default) — every descendant is a plain :class:`Envoy`.
    - An :class:`Envoy` subclass — used for every descendant.
    - A ``Dict[Type[torch.nn.Module], Type[Envoy]]`` — each descendant
      is wrapped with the first :class:`Envoy` subclass whose key
      appears in the module's MRO; unmatched modules fall back to
      :class:`Envoy`.

    Subclasses may set ``envoys`` as a class attribute to provide a
    default for all instances; users can still override it per-instance
    via the ``envoys=`` constructor kwarg (pass ``envoys=None`` to opt
    out of a subclass default)::

        class MyModel(NNsight):
            envoys = {torch.nn.Linear: MyLinearEnvoy}

    Args:
        *args: Positional arguments forwarded to :class:`Envoy`.
            The first argument should be a ``torch.nn.Module``.
        **kwargs: Keyword arguments forwarded to :class:`Envoy`.

    Class Attributes:
        envoys: Default ``envoys`` configuration for descendant modules.
            ``None`` on the base class. Subclasses can set this to a
            class or dict to apply throughout the tree by default.
    """

    envoys: Optional[
        Union[Type[Envoy], Dict[Type[torch.nn.Module], Type[Envoy]]]
    ] = None

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("envoys", type(self).envoys)

        super().__init__(*args, **kwargs)

        # TODO: legacy
        self.__dict__["_model"] = self._module

    def __getstate__(self):
        state = super().__getstate__()
        state["_model"] = self._module
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__["_model"] = state["_model"]
