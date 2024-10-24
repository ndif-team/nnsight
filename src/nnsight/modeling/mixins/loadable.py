import torch

from ...intervention import NNsight


class LoadableMixin(NNsight):

    def __init__(self, *args, **kwargs) -> None:

        if not isinstance(args[0], torch.nn.Module):

            model = self._load(*args, **kwargs)

        else:

            model = args[0]

        super().__init__(model)

    def _load(self, *args, **kwargs) -> torch.nn.Module:

        raise NotImplementedError()
