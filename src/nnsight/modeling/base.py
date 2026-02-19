from ..intervention.envoy import Envoy


class NNsight(Envoy):
    """Base NNsight wrapper around any ``torch.nn.Module``.

    Wraps a PyTorch module with NNsight's intervention capabilities,
    enabling access to and modification of intermediate activations
    during model execution via the tracing context.

    This is the simplest entry point for wrapping arbitrary models.
    For HuggingFace language models, use :class:`LanguageModel` instead.

    Example::

        import torch
        from nnsight import NNsight

        net = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 2),
        )
        model = NNsight(net)

        with model.trace(torch.rand(1, 5)):
            hidden = model[0].output.save()

    Args:
        *args: Positional arguments forwarded to :class:`Envoy`.
            The first argument should be a ``torch.nn.Module``.
        **kwargs: Keyword arguments forwarded to :class:`Envoy`.
    """

    def __init__(self, *args, **kwargs):
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
