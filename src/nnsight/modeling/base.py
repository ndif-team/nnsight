from __future__ import annotations

from ..intervention.envoy import Envoy


class NNsight(Envoy):
    """Wrap an arbitrary ``torch.nn.Module`` for tracing and intervention.

    The simplest entry point into nnsight: ``NNsight(module)`` builds a root
    [`Envoy`][nnsight.intervention.envoy.Envoy] mirroring the module's tree, so
    every submodule exposes its activations inside a ``with model.trace(...):``
    block (read them, edit them, capture gradients, ...). It is a thin, named
    [`Envoy`][nnsight.intervention.envoy.Envoy] — ``Envoy`` is the node type the tree is built from; ``NNsight``
    is the conventional name for wrapping a whole model, and the higher-level
    wrappers (``TransformersModel``, ``DiffusionModel``, ...) are specialized
    envoys that add loading/tokenization on top of the same behavior.

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
        print(hidden)
    """
