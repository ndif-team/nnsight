from typing import Dict, Optional

import torch
from accelerate import init_empty_weights

from ...intervention import NNsight
from .loadable import LoadableMixin


class MetaMixin(LoadableMixin):

    def __init__(
        self, *args, dispatch: bool = False, meta_buffers: bool = True, rename: Optional[Dict[str,str]] = None, **kwargs
    ) -> None:

        self.dispatched = False
        
        if isinstance(args[0], torch.nn.Module) or dispatch:
            
            self.dispatched = True

            super().__init__(*args, rename=rename, **kwargs)

        else:

            with init_empty_weights(include_buffers=meta_buffers):

                model = self._load_meta(*args, **kwargs)

            NNsight.__init__(self, model, rename=rename)

        self.args = args
        self.kwargs = kwargs

    def _load_meta(self, *args, **kwargs) -> torch.nn.Module:

        raise NotImplementedError()

    def dispatch(self) -> None:

        model = self._load(*self.args, **self.kwargs)
        self._envoy._update(model)
        self._model = model

        self.dispatched = True

    def interleave(self, *args, **kwargs):

        if not self.dispatched:
            self.dispatch()

        return super().interleave(*args, **kwargs)
