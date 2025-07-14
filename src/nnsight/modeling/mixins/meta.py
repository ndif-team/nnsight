from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from ...intervention.tracing.tracer import ScanningTracer
from .. import NNsight
from .loadable import LoadableMixin

if TYPE_CHECKING:
    from ...intervention.interleaver import Interleaver
else:
    Interleaver = Any


class MetaMixin(LoadableMixin):

    def __init__(
        self,
        *args,
        dispatch: bool = False,
        meta_buffers: bool = True,
        rename: Optional[Dict[str, str]] = None,
        **kwargs,
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
        self._update(model)
        # TODO legacy
        self.__dict__["_model"] = self._module

        self.dispatched = True

    def interleave(self,  fn: Callable, *args, **kwargs):

        if not self.dispatched and not isinstance(self._interleaver.tracer, ScanningTracer):
            self.dispatch()

            if isinstance(fn, torch.nn.Module):
                fn = self._module
            elif isinstance(fn, MethodType) and fn.__self__ is not None:
                # Unbind using __func__, then bind to new_instance using __get__

                new_self = (
                    self._module if isinstance(fn.__self__, torch.nn.Module) else self
                )

                fn = fn.__func__.__get__(new_self, type(new_self))

        return super().interleave(fn, *args, **kwargs)