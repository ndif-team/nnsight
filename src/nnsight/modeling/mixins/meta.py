import torch
from accelerate import init_empty_weights

from ...intervention import NNsight
from .loadable import LoadableMixin


class MetaMixin(LoadableMixin):

    def __init__(
        self, *args, dispatch: bool = False, meta_buffers: bool = True, **kwargs
    ) -> None:

        self.dispatched = dispatch
        
        if isinstance(args[0], torch.nn.Module) or dispatch:

            super().__init__(*args, **kwargs)

        else:

            with init_empty_weights(include_buffers=meta_buffers):

                model = self._load_meta(*args, **kwargs)

            NNsight.__init__(self, model)

        self.args = args
        self.kwargs = kwargs

    def _load_meta(self, *args, **kwargs) -> torch.nn.Module:

        raise NotImplementedError()

    def dispatch(self) -> None:

        self._model = self._load(*self.args, **self.kwargs)
        self._envoy._update(self._model)

        self.dispatched = True

    def interleave(self, *args, **kwargs):

        if not self.dispatched:
            self.dispatch()

        return super().interleave(*args, **kwargs)
